"""Argument canonicalization: one representative per equivalence class of tool arguments.

Speculation is only sound if "the call the model actually made" can be recognised in a result
that was computed before the model made it. Recognition happens by hashing arguments, so two
spellings of the same call must hash the same, and two different calls must not.

The rules here are deliberately *declared per tool and per field* rather than inferred. A
normalizer that guesses (say, case-folding every string) would merge calls that a tool
distinguishes, and a speculative result would then be committed for a call that should have
produced something else. Every normalizer is:

- **idempotent** -- ``f(f(x)) == f(x)``, checked by a property test, so a canonical form is a
  fixed point and re-canonicalizing a stored key never changes it;
- **semantics preserving by construction** -- the registry executes the *normalized* arguments,
  never the raw ones, so "equal keys imply equal execution" holds for any rule set, and a wrong
  rule can only make the tool see a different (still declared) argument, not make two calls with
  one key run differently.

Path normalization keeps leading ``..`` segments instead of clamping them at the root. Rewriting
``../../etc/passwd`` into ``etc/passwd`` would turn a traversal attempt into a legal-looking
path; keeping it lets the handler's workspace jail reject it.

References: RFC 3986 section 6 (URI normalization and comparison) for the syntax-based rules --
scheme and host case, default ports, percent-encoding case and unreserved decoding, and dot
segment removal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from pydantic import JsonValue

from speculative_tool_executor.core.canonical import canonical_json

__all__ = [
    "AliasNormalizer",
    "ArgRule",
    "Normalizer",
    "ToolArgumentCanonicalizer",
    "collapse_whitespace",
    "normalize_path",
    "normalize_url",
    "sorted_values",
    "to_lowercase",
]

_DEFAULT_PORTS = {"http": "80", "https": "443"}
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")
_WHITESPACE = re.compile(r"\s+")


@runtime_checkable
class Normalizer(Protocol):
    """Maps one argument value onto the representative of its equivalence class."""

    def __call__(self, value: JsonValue) -> JsonValue: ...


def _normalize_percent_encoding(text: str) -> str:
    """Decode percent escapes of unreserved characters, upper-case the rest (RFC 3986 6.2.2)."""

    def replace(match: re.Match[str]) -> str:
        code = int(match.group(1), 16)
        character = chr(code)
        if character in _UNRESERVED:
            return character
        return f"%{match.group(1).upper()}"

    return _PERCENT.sub(replace, text)


def _remove_dot_segments(path: str, *, keep_leading_parents: bool) -> str:
    """Resolve ``.`` and ``..`` inside ``path`` without reordering anything else."""
    absolute = path.startswith("/")
    trailing = path.endswith("/") and len(path) > 1
    stack: list[str] = []
    for segment in path.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if stack and stack[-1] != "..":
                stack.pop()
            elif not absolute and keep_leading_parents:
                stack.append("..")
            continue
        stack.append(segment)
    joined = "/".join(stack)
    if absolute:
        return "/" + joined
    if not joined:
        return "."
    return joined + "/" if trailing else joined


def normalize_path(value: JsonValue) -> JsonValue:
    """Normalize a workspace-relative path: ``./a//b/../c`` and ``a/c`` become ``a/c``.

    Backslashes are folded to ``/`` so Windows-style spellings do not create a second key for the
    same file. Leading ``..`` segments are preserved for the handler's jail to reject.
    """
    if not isinstance(value, str):
        return value
    text = _normalize_percent_encoding(value.replace("\\", "/"))
    return _remove_dot_segments(text, keep_leading_parents=True)


def normalize_url(value: JsonValue) -> JsonValue:
    """Syntax-based URI normalization (RFC 3986 section 6.2.2) plus sorted query parameters."""
    if not isinstance(value, str):
        return value
    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    port = parts.port
    netloc = host
    if parts.username:
        credentials = parts.username
        if parts.password:
            credentials = f"{credentials}:{parts.password}"
        netloc = f"{credentials}@{netloc}"
    if port is not None and _DEFAULT_PORTS.get(scheme) != str(port):
        netloc = f"{netloc}:{port}"
    # An empty URL path is the root, not the relative path ".". Running it through dot-segment
    # removal (which answers "where is here?" with ".") would produce "https://host/." and give
    # "https://host" and "https://host/" two different keys for the same resource -- RFC 3986
    # section 6.2.3 says they are equivalent.
    path = _normalize_percent_encoding(parts.path)
    path = _remove_dot_segments(path, keep_leading_parents=False) if path else ""
    if not path and (scheme or netloc):
        path = "/"
    pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(pairs, quote_via=quote)
    fragment = _normalize_percent_encoding(parts.fragment)
    return urlunsplit((scheme, netloc, path, query, fragment))


def to_lowercase(value: JsonValue) -> JsonValue:
    """Case-fold an identifier whose tool treats case as insignificant."""
    return value.lower() if isinstance(value, str) else value


def collapse_whitespace(value: JsonValue) -> JsonValue:
    """Strip and collapse runs of whitespace in a free-text argument."""
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else value


def sorted_values(value: JsonValue) -> JsonValue:
    """Sort a list argument whose tool ignores the order of its members but not their number.

    De-duplicating here as well would be sound -- equal keys would still imply equal execution
    -- but only because the handler would then be unable to observe a duplicate at all. That is
    a rule that erases part of the tool's input rather than one that describes what the tool
    ignores, so multiplicity is left alone and a repeated member stays a different call.
    """
    if not isinstance(value, list):
        return value
    return sorted(value, key=canonical_json)


@dataclass(frozen=True, slots=True)
class AliasNormalizer:
    """A declared equivalence between argument values, for example ``index`` and ``readme``.

    Aliases are resolved once, not transitively: the mapping must already point every alias at
    its representative, which is checked on construction and keeps the rule idempotent.
    """

    aliases: Mapping[str, str]

    def __post_init__(self) -> None:
        for alias, representative in self.aliases.items():
            target = self.aliases.get(representative, representative)
            if target != representative:
                raise ValueError(
                    f"alias {alias!r} points at {representative!r}, which is itself an alias of "
                    f"{target!r}; map every alias directly to its representative"
                )

    def __call__(self, value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return self.aliases.get(value, value)
        return value


@dataclass(frozen=True, slots=True)
class ArgRule:
    """One field's normalization: a chain of normalizers applied left to right."""

    field: str
    normalizers: tuple[Normalizer, ...]

    def apply(self, value: JsonValue) -> JsonValue:
        for normalizer in self.normalizers:
            value = normalizer(value)
        return value


class ToolArgumentCanonicalizer:
    """Applies the declared per-tool, per-field rules. Tools without rules pass through.

    Passing through is the safe default: an unnormalized argument produces a *more* specific key,
    so at worst a speculation misses. Only a declared rule can merge two spellings.
    """

    def __init__(self, rules: Mapping[str, Sequence[ArgRule]] | None = None) -> None:
        self._rules: dict[str, tuple[ArgRule, ...]] = {
            tool: tuple(entries) for tool, entries in (rules or {}).items()
        }
        for tool, entries in self._rules.items():
            fields = [rule.field for rule in entries]
            if len(set(fields)) != len(fields):
                raise ValueError(f"tool {tool} declares two rules for the same field")

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))

    def rules_for(self, tool_name: str) -> tuple[ArgRule, ...]:
        return self._rules.get(tool_name, ())

    def normalize(self, tool_name: str, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        normalized = dict(args)
        for rule in self._rules.get(tool_name, ()):
            if rule.field in normalized:
                normalized[rule.field] = rule.apply(normalized[rule.field])
        return normalized
