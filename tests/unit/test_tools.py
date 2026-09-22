from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_serializer,
)

from speculative_tool_executor.agentic.effects import (
    SPECULATABLE_EFFECTS,
    Effect,
    effect_rank,
)
from speculative_tool_executor.agentic.errors import ToolArgumentError, ToolNotAllowedError
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
    tool_call_hash,
)

pytestmark = pytest.mark.unit


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=20)
    count: int = 1


class WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class OpenArgs(BaseModel):
    path: str


def echo(args: EchoArgs) -> JsonValue:
    return args.text * args.count


def write(args: WriteArgs) -> JsonValue:
    return f"wrote {args.path}"


ECHO = ToolDefinition(
    name="echo",
    description="Echo text.",
    args_model=EchoArgs,
    effect=Effect.PURE,
    capabilities=frozenset({Capability.COMPUTE}),
    handler=echo,
)
WRITE = ToolDefinition(
    name="write_file",
    description="Write a file.",
    args_model=WriteArgs,
    effect=Effect.WRITE,
    capabilities=frozenset({Capability.WRITE_DATA}),
    handler=write,
    freshness_seconds=30.0,
)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry({Capability.COMPUTE, Capability.WRITE_DATA})
    registry.register(ECHO)
    registry.register(WRITE)
    return registry


def call(name: str, **arguments: JsonValue) -> ToolUseBlock:
    return ToolUseBlock(id="call-1", name=name, arguments=arguments)


def test_registration_requires_granted_capabilities() -> None:
    registry = ToolRegistry({Capability.COMPUTE})
    with pytest.raises(ToolNotAllowedError, match="write_data"):
        registry.register(WRITE)


def test_duplicate_registration_is_rejected() -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ECHO)


def test_a_tool_cannot_be_defined_without_an_effect_class() -> None:
    with pytest.raises(TypeError, match="effect"):
        ToolDefinition(  # type: ignore[call-arg]
            name="no_effect",
            description="x",
            args_model=EchoArgs,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=echo,
        )


def test_effect_constrains_which_capabilities_a_tool_may_hold() -> None:
    """A tool cannot call itself a read while holding a capability that writes."""
    with pytest.raises(ValueError, match=r"effect read .* does not allow"):
        ToolDefinition(
            name="sneaky_read",
            description="Claims to read, writes.",
            args_model=WriteArgs,
            effect=Effect.READ,
            capabilities=frozenset({Capability.READ_DATA, Capability.WRITE_DATA}),
            handler=write,
            freshness_seconds=5.0,
        )


def test_non_pure_tools_must_bound_how_long_a_speculative_result_lives() -> None:
    with pytest.raises(ValueError, match="freshness_seconds"):
        ToolDefinition(
            name="unbounded_read",
            description="x",
            args_model=WriteArgs,
            effect=Effect.READ,
            capabilities=frozenset({Capability.READ_DATA}),
            handler=write,
        )


def test_tool_definitions_are_validated() -> None:
    with pytest.raises(ValueError, match="extra='forbid'"):
        ToolDefinition(
            name="open",
            description="Open.",
            args_model=OpenArgs,
            effect=Effect.PURE,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=lambda args: args.path,
        )
    with pytest.raises(ValueError, match="invalid tool name"):
        ToolDefinition(
            name="bad name",
            description="x",
            args_model=EchoArgs,
            effect=Effect.PURE,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=echo,
        )
    with pytest.raises(ValueError, match="at least one capability"):
        ToolDefinition(
            name="none",
            description="x",
            args_model=EchoArgs,
            effect=Effect.PURE,
            capabilities=frozenset(),
            handler=echo,
        )


def test_irreversible_capabilities_cannot_opt_out_of_approval() -> None:
    """A contained write may; anything that leaves the system may not."""
    contained = ToolDefinition(
        name="store_write",
        description="Buffered write to the versioned store.",
        args_model=WriteArgs,
        effect=Effect.WRITE,
        capabilities=frozenset({Capability.WRITE_DATA}),
        handler=write,
        requires_approval=False,
        freshness_seconds=30.0,
    )
    assert contained.needs_approval is False

    for effect, capability in (
        (Effect.EXTERNAL, Capability.SEND_MESSAGE),
        (Effect.EXTERNAL, Capability.NETWORK_EGRESS),
        (Effect.EXTERNAL, Capability.EXECUTE_CODE),
    ):
        with pytest.raises(ValueError, match="outside this system"):
            ToolDefinition(
                name="escapes",
                description="x",
                args_model=WriteArgs,
                effect=effect,
                capabilities=frozenset({capability}),
                handler=write,
                requires_approval=False,
                freshness_seconds=30.0,
            )


def test_approval_defaults_follow_effect_and_capabilities() -> None:
    assert ECHO.needs_approval is False
    assert WRITE.needs_approval is True  # WRITE_DATA defaults to needing approval
    cautious = ToolDefinition(
        name="echo_reviewed",
        description="Echo with review.",
        args_model=EchoArgs,
        effect=Effect.PURE,
        capabilities=frozenset({Capability.COMPUTE}),
        handler=echo,
        requires_approval=True,
    )
    assert cautious.needs_approval is True
    assert cautious.speculatable is False


def test_speculatable_follows_the_effect_lattice() -> None:
    assert effect_rank(Effect.PURE) < effect_rank(Effect.READ) < effect_rank(Effect.WRITE)
    assert effect_rank(Effect.WRITE) < effect_rank(Effect.EXTERNAL)
    assert Effect.EXTERNAL not in SPECULATABLE_EFFECTS
    external = ToolDefinition(
        name="send",
        description="Send.",
        args_model=WriteArgs,
        effect=Effect.EXTERNAL,
        capabilities=frozenset({Capability.SEND_MESSAGE}),
        handler=write,
        freshness_seconds=30.0,
    )
    assert external.needs_approval is True
    assert external.speculatable is False
    assert ECHO.speculatable is True


def test_unknown_tools_are_not_allowed() -> None:
    with pytest.raises(ToolNotAllowedError, match="delete_all"):
        make_registry().validate_call(call("delete_all"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": "hi", "extra": "SECRETVALUE"},
        {"text": "hi", "count": "3"},
        {"text": "SECRETVALUE" * 3},
        {},
    ],
)
def test_invalid_arguments_are_rejected_without_echoing_values(
    arguments: dict[str, JsonValue],
) -> None:
    with pytest.raises(ToolArgumentError) as info:
        make_registry().validate_call(call("echo", **arguments))
    assert "SECRETVALUE" not in str(info.value)


def test_valid_calls_are_canonicalized_and_runnable() -> None:
    registry = make_registry()
    validated = registry.validate_call(call("echo", text="hi"))
    assert validated.canonical_args == {"text": "hi", "count": 1}
    assert validated.run() == "hi"
    assert validated.call_hash == tool_call_hash("echo", {"count": 1, "text": "hi"})
    assert validated.effect is Effect.PURE, "the effect travels with the validated call"
    assert registry.canonicalizer is None, "this registry declares no equivalences"


def test_running_with_a_stale_authorization_is_refused() -> None:
    validated = make_registry().validate_call(call("echo", text="hi"))
    with pytest.raises(ToolArgumentError, match="after authorization"):
        validated.run(authorized_hash="0" * 64)


def test_scoped_registry_is_a_narrower_allowlist() -> None:
    registry = make_registry()
    scoped = registry.scoped(["echo"])
    assert scoped.names == ("echo",)
    assert scoped.granted == frozenset({Capability.COMPUTE})
    with pytest.raises(ToolNotAllowedError):
        scoped.validate_call(call("write_file", path="x"))
    with pytest.raises(ValueError, match="unknown tools"):
        registry.scoped(["echo", "missing"])


def test_specs_are_sorted_and_schemas_are_closed() -> None:
    specs = make_registry().specs()
    assert [spec.name for spec in specs] == ["echo", "write_file"]
    assert all(spec.input_schema.get("additionalProperties") is False for spec in specs)


class TokenArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: SecretStr


def test_argument_models_that_do_not_round_trip_are_refused_without_echoing_values() -> None:
    registry = ToolRegistry({Capability.COMPUTE})
    registry.register(
        ToolDefinition(
            name="use_token",
            description="Use a token.",
            args_model=TokenArgs,
            effect=Effect.PURE,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=lambda args: "ok",
        )
    )
    # SecretStr dumps as "**********", so every token would share one call hash.
    with pytest.raises(ToolArgumentError, match="canonical") as info:
        registry.validate_call(call("use_token", token="s3cr3t-value"))
    assert "s3cr3t-value" not in str(info.value)


# -- the canonical form has to be a fixed point --------------------------------------------------


class _Canonicalizer:
    """A deliberately broken canonicalizer, so the registry's own guard can be observed."""

    def __init__(self, replacement: dict[str, JsonValue]) -> None:
        self._replacement = replacement

    def normalize(self, tool_name: str, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        del tool_name, args
        return dict(self._replacement)


class TrimmedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, AfterValidator(str.strip)]


TRIM = ToolDefinition(
    name="trim",
    description="Echo trimmed text.",
    args_model=TrimmedArgs,
    effect=Effect.PURE,
    capabilities=frozenset({Capability.COMPUTE}),
    handler=lambda args: args.text,
)


def _registry_with(canonicalizer: _Canonicalizer, tool: ToolDefinition[Any] = TRIM) -> ToolRegistry:
    registry = ToolRegistry({Capability.COMPUTE}, canonicalizer=canonicalizer)
    registry.register(tool)
    return registry


def test_a_canonicalizer_whose_output_fails_the_schema_is_refused() -> None:
    """The canonical form is what gets hashed *and* executed, so it must still be valid."""
    registry = _registry_with(_Canonicalizer({"text": 123}))
    with pytest.raises(ToolArgumentError, match="canonical"):
        registry.validate_call(call("trim", text="hi"))


def test_a_canonicalizer_whose_output_is_not_a_fixed_point_is_refused() -> None:
    """Two spellings may share a call hash only if they also execute identically.

    ``TrimmedArgs`` strips on validation, so a canonicalizer emitting padded text produces
    something the model re-serializes differently: the key would say ``" hi "`` while the handler
    received ``"hi"``, and a second call spelled ``"hi"`` would then be served that branch.
    """
    registry = _registry_with(_Canonicalizer({"text": "  hi  "}))
    with pytest.raises(ToolArgumentError, match="round-trip"):
        registry.validate_call(call("trim", text="hi"))


def test_a_canonicalizer_that_is_a_fixed_point_is_accepted() -> None:
    """The positive control: the guard refuses broken normalizers, not normalization."""
    registry = _registry_with(_Canonicalizer({"text": "HI"}))
    validated = registry.validate_call(call("trim", text="hi"))
    assert validated.canonical_args == {"text": "HI"}
    assert validated.run() == "HI"


# -- guards that only a dynamically assembled tool can reach --------------------------------------


def test_an_effect_that_is_not_an_effect_class_is_refused_at_definition() -> None:
    """A plain string is the shape a hand-built dict or a loose config would produce."""
    # Typed as Any on purpose: this is the value a loosely typed caller would hand in.
    stringly_typed_effect: Any = "read"
    with pytest.raises(TypeError, match="must declare an effect class"):
        ToolDefinition(
            name="stringly_typed",
            description="x",
            args_model=EchoArgs,
            effect=stringly_typed_effect,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=echo,
        )


def test_registration_re_checks_the_effect_of_a_tool_assembled_behind_the_constructor() -> None:
    """``__post_init__`` is not the only way in, so ``register`` checks the effect itself."""
    smuggled = ToolDefinition(
        name="smuggled",
        description="x",
        args_model=EchoArgs,
        effect=Effect.PURE,
        capabilities=frozenset({Capability.COMPUTE}),
        handler=echo,
    )
    object.__setattr__(smuggled, "effect", "pure")

    registry = ToolRegistry({Capability.COMPUTE})
    with pytest.raises(ToolNotAllowedError, match="did not declare an effect class"):
        registry.register(smuggled)
    assert registry.names == ()


def test_a_non_positive_freshness_window_is_refused() -> None:
    for window in (0.0, -1.0):
        with pytest.raises(ValueError, match="freshness_seconds must be positive"):
            ToolDefinition(
                name="stale",
                description="x",
                args_model=WriteArgs,
                effect=Effect.READ,
                capabilities=frozenset({Capability.READ_DATA}),
                handler=write,
                freshness_seconds=window,
            )


def test_negative_expected_latency_or_cost_is_refused() -> None:
    for latency, cost in ((-0.1, 0.0), (0.0, -0.1)):
        with pytest.raises(ValueError, match="negative expected latency or cost"):
            ToolDefinition(
                name="impossible",
                description="x",
                args_model=EchoArgs,
                effect=Effect.PURE,
                capabilities=frozenset({Capability.COMPUTE}),
                handler=echo,
                expected_latency_seconds=latency,
                expected_cost_usd=cost,
            )


class SkewedArgs(BaseModel):
    """An args model whose dump does not re-validate: the shape the round-trip check is for."""

    model_config = ConfigDict(extra="forbid")
    value: int

    @field_serializer("value")
    def _as_text(self, value: int) -> str:
        return str(value)


def test_an_args_model_whose_dump_does_not_re_validate_is_refused() -> None:
    """A lossy serializer would put the handler's input outside the call hash."""
    registry = ToolRegistry({Capability.COMPUTE})
    registry.register(
        ToolDefinition(
            name="skewed",
            description="x",
            args_model=SkewedArgs,
            effect=Effect.PURE,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=lambda args: args.value,
        )
    )
    with pytest.raises(ToolArgumentError, match="round-trip"):
        registry.validate_call(call("skewed", value=7))
