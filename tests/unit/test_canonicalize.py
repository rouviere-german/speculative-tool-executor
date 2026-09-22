"""Argument canonicalization: the equivalence relation speculation matches calls with.

Two properties carry the whole design:

1. *Equal spellings get equal keys* -- otherwise a correct speculation is thrown away.
2. *Equal keys execute equally* -- otherwise a speculation is committed for a call that should
   have produced something else. This one is the safety property, and it is established
   structurally rather than statistically: the registry runs the **normalized** arguments, so two
   calls sharing a key are literally the same call by the time a handler sees them. The test
   below checks that structural claim end to end.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import JsonValue

from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.core.canonical import canonical_json
from speculative_tool_executor.spec.canonicalize import (
    AliasNormalizer,
    ArgRule,
    ToolArgumentCanonicalizer,
    collapse_whitespace,
    normalize_path,
    normalize_url,
    sorted_values,
    to_lowercase,
)
from speculative_tool_executor.workload.corpus import WorkspaceError, build_corpus
from speculative_tool_executor.workload.tools import build_registry

pytestmark = pytest.mark.unit

_SEGMENT = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_."),
    min_size=1,
    max_size=8,
)
_PATHS = st.lists(_SEGMENT, min_size=1, max_size=5).map("/".join)
_HOSTS = st.sampled_from(["example.test", "EXAMPLE.test", "docs.internal.test"])
_TEXT = st.text(min_size=0, max_size=40)

NORMALIZERS = (normalize_path, normalize_url, to_lowercase, collapse_whitespace)


@pytest.mark.parametrize("normalizer", NORMALIZERS, ids=lambda f: f.__name__)
@given(value=st.one_of(_PATHS, _TEXT, st.integers(), st.none(), st.booleans()))
def test_every_normalizer_is_idempotent(normalizer: object, value: JsonValue) -> None:
    """A canonical form must be a fixed point, or re-canonicalising a stored key would move it."""
    once = normalizer(value)  # type: ignore[operator]
    assert normalizer(once) == once  # type: ignore[operator]


@given(items=st.lists(st.text(max_size=5), max_size=8))
def test_sorted_values_is_idempotent_order_insensitive_and_keeps_multiplicity(
    items: list[str],
) -> None:
    values: JsonValue = list(items)
    once = sorted_values(values)
    assert sorted_values(once) == once
    reversed_values: JsonValue = list(reversed(items))
    assert sorted_values(reversed_values) == once
    assert isinstance(once, list)
    assert sorted(str(item) for item in once) == sorted(items), "no member dropped or added"


@given(path=_PATHS, repeats=st.integers(min_value=1, max_value=3))
def test_equivalent_path_spellings_normalize_together(path: str, repeats: int) -> None:
    """Redundant ``./`` segments, doubled slashes and backslashes all name the same file."""
    assume(not path.endswith("/"))
    decorated = ("./" * repeats) + path.replace("/", "/" * repeats).replace("/", "\\", 1)
    assert normalize_path(decorated) == normalize_path(path)


@given(path=_PATHS)
def test_parent_segments_are_preserved_not_clamped(path: str) -> None:
    """``../`` must survive normalization so the workspace jail can still refuse it.

    Rewriting ``../../etc/passwd`` into ``etc/passwd`` would launder a traversal attempt into a
    legal-looking path, and the handler would then happily serve it.
    """
    escaped = normalize_path(f"../../{path}")
    assert isinstance(escaped, str)
    assert escaped.startswith("../../")


@given(host=_HOSTS, path=_PATHS)
def test_equivalent_url_spellings_normalize_together(host: str, path: str) -> None:
    plain = f"https://{host}/{path}"
    fancy = f"HTTPS://{host.upper()}:443/{path}"
    assert normalize_url(plain) == normalize_url(fancy)


def test_query_parameter_order_is_not_significant() -> None:
    left = normalize_url("https://a.test/x?b=2&a=1")
    right = normalize_url("https://a.test/x?a=1&b=2")
    assert left == right == "https://a.test/x?a=1&b=2"


def test_different_paths_keep_different_keys() -> None:
    """Normalization must merge spellings, never distinct calls."""
    canonicalizer = ToolArgumentCanonicalizer(
        {"read": [ArgRule(field="path", normalizers=(normalize_path,))]}
    )
    seen = {
        canonical_json(canonicalizer.normalize("read", {"path": path}))
        for path in ("a/b", "a/c", "b/a", "a/b/c", "../a/b")
    }
    assert len(seen) == 5


def test_aliases_must_point_directly_at_their_representative() -> None:
    AliasNormalizer({"index": "readme", "home": "readme"})
    with pytest.raises(ValueError, match="itself an alias"):
        AliasNormalizer({"index": "home", "home": "readme"})


def test_a_tool_cannot_declare_two_rules_for_one_field() -> None:
    with pytest.raises(ValueError, match="two rules for the same field"):
        ToolArgumentCanonicalizer(
            {
                "read": [
                    ArgRule(field="path", normalizers=(normalize_path,)),
                    ArgRule(field="path", normalizers=(to_lowercase,)),
                ]
            }
        )


def test_tools_without_rules_pass_through_unchanged() -> None:
    canonicalizer = ToolArgumentCanonicalizer()
    assert canonicalizer.normalize("anything", {"a": " B "}) == {"a": " B "}
    assert canonicalizer.tools == ()


@pytest.mark.parametrize(
    ("tool", "left", "right"),
    [
        ("read_file", {"path": "src/auth/api.py"}, {"path": "./src//auth/../auth/api.py"}),
        ("list_dir", {"path": "src/auth"}, {"path": "src\\auth"}),
        ("search_code", {"query": "auth_api"}, {"query": "  AUTH_API  "}),
        (
            "fetch_doc",
            {"url": "https://docs.internal.test/guide/auth"},
            {"url": "HTTPS://DOCS.INTERNAL.TEST:443/guide/./auth"},
        ),
        (
            "tally_items",
            {"items": ["b", "a", "a"]},
            {"items": ["a", "a", "b"]},
        ),
    ],
)
def test_equal_keys_imply_equal_execution(
    tool: str, left: dict[str, JsonValue], right: dict[str, JsonValue]
) -> None:
    """The safety property: same key, same handler input, same result.

    This is what lets a result computed for one spelling be handed to a call written another way.
    """
    registry, _ = build_registry(build_corpus())
    first = registry.validate_call(ToolUseBlock(id="a", name=tool, arguments=left))
    second = registry.validate_call(ToolUseBlock(id="b", name=tool, arguments=right))

    assert first.call_hash == second.call_hash
    assert first.canonical_args == second.canonical_args
    assert canonical_json(first.run()) == canonical_json(second.run())


def test_a_tally_keeps_multiplicity_out_of_its_equivalence_class() -> None:
    """``tally_items`` counts a multiset: order is insignificant to it, repetition is not.

    De-duplicating in the canonicalizer would make the rule *sound* -- equal keys would still
    imply equal execution -- by making the handler unable to observe a duplicate at all, which
    would leave the tool's own frequency count unreachable. So the declared equivalence sorts
    and nothing more, and a repeated item is a different call with a different key.
    """
    registry, _ = build_registry(build_corpus())
    repeated = registry.validate_call(
        ToolUseBlock(id="a", name="tally_items", arguments={"items": ["b", "a", "a"]})
    )
    distinct = registry.validate_call(
        ToolUseBlock(id="b", name="tally_items", arguments={"items": ["a", "b"]})
    )

    assert repeated.canonical_args == {"items": ["a", "a", "b"]}
    assert repeated.call_hash != distinct.call_hash
    assert repeated.run() == {"distinct": 2, "total": 3, "top": ["a", "b"]}
    assert distinct.run() == {"distinct": 2, "total": 2, "top": ["a", "b"]}


def test_without_the_declared_rules_an_equivalent_spelling_is_a_different_call() -> None:
    """What canonicalization is worth, stated where it can actually be established.

    The benchmark cannot show this: the task generator emits arguments already in canonical
    form, so an ablation that switches the rules off measures the generator, not the design.
    Here the alternate spelling is constructed on purpose, and without the rules it both hashes
    differently (so no speculation could ever match it) and fails outright, because the handler
    receives the raw string and the workspace has no such file.
    """
    plain = {"path": "src/auth/api.py"}
    spelled = {"path": "./src//auth/api.py"}
    traversal_looking = {"path": "src/auth/../auth/api.py"}

    with_rules, _ = build_registry(build_corpus(), canonicalize=True)
    left = with_rules.validate_call(ToolUseBlock(id="a", name="read_file", arguments=plain))
    right = with_rules.validate_call(ToolUseBlock(id="b", name="read_file", arguments=spelled))
    assert left.call_hash == right.call_hash
    assert right.run() == left.run()

    without_rules, _ = build_registry(build_corpus(), canonicalize=False)
    bare_left = without_rules.validate_call(ToolUseBlock(id="a", name="read_file", arguments=plain))
    bare_right = without_rules.validate_call(
        ToolUseBlock(id="b", name="read_file", arguments=spelled)
    )
    assert bare_left.call_hash != bare_right.call_hash
    with pytest.raises(WorkspaceError, match="file not found"):
        bare_right.run()

    # Worse than a miss: an ordinary spelling containing a resolved ".." segment is refused by
    # the jail, because nothing resolved it first.
    bare_traversal = without_rules.validate_call(
        ToolUseBlock(id="c", name="read_file", arguments=traversal_looking)
    )
    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        bare_traversal.run()
    assert (
        with_rules.validate_call(
            ToolUseBlock(id="c", name="read_file", arguments=traversal_looking)
        ).run()
        == left.run()
    )


def test_different_calls_keep_different_hashes_through_the_registry() -> None:
    registry, _ = build_registry(build_corpus())
    hashes = {
        registry.validate_call(
            ToolUseBlock(id="x", name="read_file", arguments={"path": path})
        ).call_hash
        for path in ("src/auth/api.py", "src/auth/utils.py", "src/billing/api.py")
    }
    assert len(hashes) == 3


def test_percent_encoding_is_normalized_per_rfc_3986() -> None:
    """Unreserved characters are decoded; everything else keeps an upper-case escape."""
    # %61 is "a", which is unreserved and must be decoded.
    assert normalize_path("%61/b") == "a/b"
    # A space (%20) is reserved: the escape stays, normalized to upper case.
    assert normalize_url("https://a.test/x%2fy%3Fz") == "https://a.test/x%2Fy%3Fz"
    assert normalize_url("https://a.test/a%20b") == "https://a.test/a%20b"


def test_userinfo_and_non_default_ports_survive_normalization() -> None:
    """Dropping either would merge two URLs that address different things."""
    assert normalize_url("https://user@a.test/x") == "https://user@a.test/x"
    assert normalize_url("https://user:pw@a.test/x") == "https://user:pw@a.test/x"
    assert normalize_url("https://a.test:8443/x") == "https://a.test:8443/x"
    # The default port for the scheme is redundant and is dropped.
    assert normalize_url("https://a.test:443/x") == normalize_url("https://a.test/x")
    assert normalize_url("http://a.test:80/x") == normalize_url("http://a.test/x")


def test_an_empty_url_path_becomes_a_root_path() -> None:
    """RFC 3986 6.2.3: a missing path and "/" name the same resource, so they must share a key."""
    assert normalize_url("https://a.test") == "https://a.test/"
    assert normalize_url("https://a.test") == normalize_url("https://a.test/")
    assert normalize_url("https://a.test/.") == normalize_url("https://a.test/")


def test_non_string_and_non_list_values_pass_through_every_normalizer() -> None:
    """Normalizers are declared per field, so they must ignore a value of the wrong shape."""
    values: list[JsonValue] = [42, True, None, {"a": 1}]
    for value in values:
        assert normalize_path(value) == value
        assert normalize_url(value) == value
        assert to_lowercase(value) == value
        assert collapse_whitespace(value) == value
        assert sorted_values(value) == value


def test_aliases_ignore_values_that_are_not_strings() -> None:
    normalizer = AliasNormalizer({"index": "readme"})
    assert normalizer("index") == "readme"
    assert normalizer("other") == "other"
    assert normalizer(7) == 7


def test_declared_rules_are_inspectable() -> None:
    rule = ArgRule(field="path", normalizers=(normalize_path, to_lowercase))
    canonicalizer = ToolArgumentCanonicalizer({"read": [rule]})
    assert canonicalizer.tools == ("read",)
    assert canonicalizer.rules_for("read") == (rule,)
    assert canonicalizer.rules_for("unknown") == ()
    # Chained normalizers apply left to right.
    assert canonicalizer.normalize("read", {"path": "./A/B"}) == {"path": "a/b"}
    # A field the call did not supply is not invented.
    assert canonicalizer.normalize("read", {"other": 1}) == {"other": 1}
