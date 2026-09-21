"""Least-privilege, effect-typed tool registry with strict argument schemas.

Two things make this registry different from a plain allowlist:

1. **Every tool declares an effect class** (:mod:`speculative_tool_executor.agentic.effects`).
   There is no default, so a tool cannot be registered without one, and the declared effect must
   be consistent with the capabilities the tool asks for. The speculator refuses to run anything
   whose effect is not speculatable, so the effect declaration is the boundary that keeps
   speculation safe.
2. **Arguments are normalized before they are hashed or executed.** A registry may be given an
   :class:`ArgumentCanonicalizer`; the normalized form is validated again, the handler receives
   *that* object, and the call hash is taken over it. Two spellings that normalize to the same
   arguments therefore always execute identically, which is what makes a speculative result safe
   to reuse for the authoritative call.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, JsonValue, ValidationError

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.agentic.errors import ToolArgumentError, ToolNotAllowedError
from speculative_tool_executor.agentic.providers.base import (
    TOOL_NAME_PATTERN,
    ToolSpec,
    ToolUseBlock,
)
from speculative_tool_executor.core.canonical import canonical_json, sha256_hex

__all__ = [
    "EFFECT_CAPABILITIES",
    "IRREVERSIBLE_CAPABILITIES",
    "SIDE_EFFECT_CAPABILITIES",
    "ArgumentCanonicalizer",
    "Capability",
    "ToolDefinition",
    "ToolRegistry",
    "ValidatedCall",
    "tool_call_hash",
]


class Capability(StrEnum):
    COMPUTE = "compute"
    READ_DATA = "read_data"
    NETWORK_READ = "network_read"
    WRITE_DATA = "write_data"
    NETWORK_EGRESS = "network_egress"
    SEND_MESSAGE = "send_message"
    EXECUTE_CODE = "execute_code"


#: Capabilities that change something. A tool declaring one needs approval *by default*, so a
#: tool author has to say out loud that a side effect is safe to run without a human.
SIDE_EFFECT_CAPABILITIES = frozenset(
    {
        Capability.WRITE_DATA,
        Capability.NETWORK_EGRESS,
        Capability.SEND_MESSAGE,
        Capability.EXECUTE_CODE,
    }
)

#: Capabilities whose effects leave this system and cannot be taken back. A tool holding one can
#: never opt out of approval, whatever it declares.
#:
#: ``WRITE_DATA`` is deliberately *not* here, and that is the distinction this repo turns on.
#: A write to the versioned store is buffered in a branch overlay, validated against the versions
#: it read and applied in one transaction, so it can be discarded without a trace right up to the
#: moment it commits. Requiring a human for every such write would make speculation on write tools
#: impossible, which would reduce the design to "speculate reads only". An effect that has already
#: reached a third party has no equivalent undo, so the approval requirement stays absolute there.
#: See docs/adr/0002-effect-classes-and-the-approval-boundary.md.
IRREVERSIBLE_CAPABILITIES = frozenset(
    {
        Capability.NETWORK_EGRESS,
        Capability.SEND_MESSAGE,
        Capability.EXECUTE_CODE,
    }
)

#: Capabilities each effect class may declare. ``EXTERNAL`` may declare anything, because it is
#: never speculated and always needs an approval; the narrower classes are constrained so that a
#: tool cannot claim to be a harmless read while holding a capability that writes or sends.
EFFECT_CAPABILITIES: Mapping[Effect, frozenset[Capability]] = {
    Effect.PURE: frozenset({Capability.COMPUTE}),
    Effect.READ: frozenset({Capability.COMPUTE, Capability.READ_DATA, Capability.NETWORK_READ}),
    Effect.WRITE: frozenset(
        {
            Capability.COMPUTE,
            Capability.READ_DATA,
            Capability.NETWORK_READ,
            Capability.WRITE_DATA,
        }
    ),
    Effect.EXTERNAL: frozenset(Capability),
}

_NAME = re.compile(TOOL_NAME_PATTERN)


def tool_call_hash(tool_name: str, canonical_args: Mapping[str, JsonValue]) -> str:
    """SHA-256 over the canonical JSON of ``{"v": 1, "tool": name, "args": args}``."""
    return sha256_hex(canonical_json({"v": 1, "tool": tool_name, "args": dict(canonical_args)}))


@runtime_checkable
class ArgumentCanonicalizer(Protocol):
    """Maps a tool's dumped arguments onto one representative of their equivalence class."""

    def normalize(self, tool_name: str, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition[ArgsT: BaseModel]:
    """A tool: name, argument model, effect class, capabilities and handler.

    ``effect`` has no default: a tool that does not declare one cannot be constructed, and
    :meth:`ToolRegistry.register` checks it again so a tool assembled dynamically is refused too.

    ``requires_approval`` defaults to True whenever the effect is ``EXTERNAL`` or any side-effect
    capability is declared. The argument model must forbid extra fields, so the schema sent to
    the model is closed. Its JSON dump must also round-trip (no ``exclude=True`` fields,
    ``SecretStr`` or lossy serializers): the dump is what approvals and speculation keys hash, so
    it has to be exactly what the handler receives.

    ``freshness_seconds`` bounds how long a speculative result for this tool may be reused. It
    applies to ``READ`` tools, whose observations can go stale for reasons the versioned store
    does not see (a workspace file, a bundled service). ``None`` means "no time bound", which is
    only correct for ``PURE`` tools.
    """

    name: str
    description: str
    args_model: type[ArgsT]
    effect: Effect
    capabilities: frozenset[Capability]
    handler: Callable[[ArgsT], JsonValue]
    requires_approval: bool | None = None
    freshness_seconds: float | None = None
    expected_latency_seconds: float = 0.0
    expected_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")
        if not isinstance(self.effect, Effect):
            raise TypeError(f"tool {self.name} must declare an effect class")
        if not self.capabilities:
            raise ValueError(f"tool {self.name} must declare at least one capability")
        excess = self.capabilities - EFFECT_CAPABILITIES[self.effect]
        if excess:
            names = ", ".join(sorted(excess))
            raise ValueError(
                f"tool {self.name} declares effect {self.effect.value} but needs capabilities "
                f"that effect does not allow: {names}"
            )
        if self.args_model.model_config.get("extra") != "forbid":
            raise ValueError(f"tool {self.name} args model must set extra='forbid'")
        if self.requires_approval is False and (
            self.effect is Effect.EXTERNAL or self.capabilities & IRREVERSIBLE_CAPABILITIES
        ):
            raise ValueError(
                f"tool {self.name} has effects outside this system and cannot opt out of approval"
            )
        if self.freshness_seconds is not None and self.freshness_seconds <= 0:
            raise ValueError(f"tool {self.name} freshness_seconds must be positive")
        if self.effect is not Effect.PURE and self.freshness_seconds is None:
            raise ValueError(
                f"tool {self.name} has effect {self.effect.value} and must bound how long a "
                "speculative result stays usable (freshness_seconds)"
            )
        if self.expected_latency_seconds < 0 or self.expected_cost_usd < 0:
            raise ValueError(f"tool {self.name} cannot have negative expected latency or cost")

    @property
    def needs_approval(self) -> bool:
        if self.requires_approval is not None:
            return self.requires_approval
        return self.effect is Effect.EXTERNAL or bool(self.capabilities & SIDE_EFFECT_CAPABILITIES)

    @property
    def speculatable(self) -> bool:
        """Whether the effect class permits running this tool ahead of the model at all."""
        return self.effect is not Effect.EXTERNAL and not self.needs_approval

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )


@dataclass(frozen=True, slots=True)
class ValidatedCall:
    """A resolved, strictly validated call whose ``canonical_args`` are exactly what runs.

    ``ToolRegistry.validate_call`` proves that ``args`` is reconstructed from ``canonical_args``,
    so :attr:`call_hash` stands for the call in approvals, speculation keys and audit records.
    Mutating either after validation is detected before authorization and before the handler runs.
    """

    tool: ToolDefinition[Any]
    args: BaseModel
    canonical_args: dict[str, JsonValue]

    @property
    def call_hash(self) -> str:
        return tool_call_hash(self.tool.name, self.canonical_args)

    @property
    def effect(self) -> Effect:
        return self.tool.effect

    def ensure_unchanged(self) -> None:
        """Raise ToolArgumentError if ``args`` no longer matches ``canonical_args``."""
        current = canonical_json(self.args.model_dump(mode="json"))
        if current != canonical_json(self.canonical_args):
            raise ToolArgumentError(f"arguments for {self.tool.name} changed after validation")

    def run(self, *, authorized_hash: str | None = None) -> JsonValue:
        """Run the handler; with ``authorized_hash``, only if the call still hashes to it."""
        self.ensure_unchanged()
        if authorized_hash is not None and self.call_hash != authorized_hash:
            raise ToolArgumentError(f"arguments for {self.tool.name} changed after authorization")
        return self.tool.handler(self.args)


def _round_trip_error(tool_name: str) -> ToolArgumentError:
    return ToolArgumentError(
        f"arguments for {tool_name} do not round-trip through their canonical form; the args "
        "model must not exclude, mask or transform fields when dumped"
    )


class ToolRegistry:
    """Tools available to one agent. Registration fails if a tool needs ungranted capabilities.

    A ``canonicalizer`` makes the registry map every call onto one representative of its
    equivalence class. Because both the speculator and the authoritative path validate through
    this one object, a prediction can never reach a handler by a route that skips the allowlist,
    the schema or the normalization the real call goes through.
    """

    def __init__(
        self,
        granted: Iterable[Capability],
        *,
        canonicalizer: ArgumentCanonicalizer | None = None,
    ) -> None:
        self._granted = frozenset(granted)
        self._canonicalizer = canonicalizer
        self._tools: dict[str, ToolDefinition[Any]] = {}

    @property
    def granted(self) -> frozenset[Capability]:
        return self._granted

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def canonicalizer(self) -> ArgumentCanonicalizer | None:
        return self._canonicalizer

    def register(self, tool: ToolDefinition[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} is already registered")
        if not isinstance(tool.effect, Effect):
            raise ToolNotAllowedError(f"tool {tool.name} did not declare an effect class")
        missing = tool.capabilities - self._granted
        if missing:
            needed = ", ".join(sorted(missing))
            raise ToolNotAllowedError(f"tool {tool.name} needs capabilities not granted: {needed}")
        self._tools[tool.name] = tool

    def scoped(self, allowed: Iterable[str]) -> ToolRegistry:
        """A narrower registry holding only the ``allowed`` tools (per-workflow allowlist)."""
        names = set(allowed)
        unknown = names - self._tools.keys()
        if unknown:
            raise ValueError(f"unknown tools in allowlist: {', '.join(sorted(unknown))}")
        tools = [self._tools[name] for name in sorted(names)]
        granted: set[Capability] = set()
        for tool in tools:
            granted |= tool.capabilities
        child = ToolRegistry(granted, canonicalizer=self._canonicalizer)
        for tool in tools:
            child.register(tool)
        return child

    def specs(self) -> tuple[ToolSpec, ...]:
        """Tool specs in a stable order, so prompts (and their hashes) are deterministic."""
        return tuple(self._tools[name].spec() for name in self.names)

    def get(self, name: str) -> ToolDefinition[Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotAllowedError(f"tool {name} is not available to this agent")
        return tool

    def validate_call(self, call: ToolUseBlock) -> ValidatedCall:
        """Resolve, strictly validate and canonicalize a proposed call. Nothing is executed here.

        There are two separate properties to establish, and conflating them loses one of them.

        **The dump must represent the arguments faithfully.** Re-validating the model's own JSON
        dump has to reproduce an *equal object*. An args model that masks or drops a field
        (``SecretStr`` renders as ``**********``, ``exclude=True`` omits it, a lossy serializer
        rounds it) passes a dump-to-dump comparison trivially -- both sides are equally masked --
        while leaving a value that the call hash never covered. Approvals and speculation keys are
        that hash, so such a value could be swapped after approval. This is checked on the raw
        dump, before any normalization, because it is a property of the args model alone.

        **The canonical form must be a fixed point.** After the canonicalizer runs, re-validating
        the normalized JSON must dump back to exactly that JSON. A canonicalizer that is not
        idempotent, or that produces something the schema re-serializes differently, would let two
        calls share a key and still execute differently, which is precisely what speculation must
        never allow.
        """
        tool = self.get(call.name)
        args = self._validate_args(tool, call.arguments)
        raw: dict[str, JsonValue] = args.model_dump(mode="json")
        try:
            echoed = tool.args_model.model_validate_json(canonical_json(raw), strict=True)
        except ValidationError as exc:
            raise _round_trip_error(tool.name) from exc
        if echoed != args or canonical_json(echoed.model_dump(mode="json")) != canonical_json(raw):
            raise _round_trip_error(tool.name)

        canonical = raw
        if self._canonicalizer is not None:
            canonical = self._canonicalizer.normalize(tool.name, raw)
        if canonical_json(canonical) == canonical_json(raw):
            return ValidatedCall(tool=tool, args=echoed, canonical_args=raw)
        try:
            bound = tool.args_model.model_validate_json(canonical_json(canonical), strict=True)
        except ValidationError as exc:
            raise _round_trip_error(tool.name) from exc
        if canonical_json(bound.model_dump(mode="json")) != canonical_json(canonical):
            raise _round_trip_error(tool.name)
        return ValidatedCall(tool=tool, args=bound, canonical_args=canonical)

    @staticmethod
    def _validate_args(tool: ToolDefinition[Any], arguments: Mapping[str, JsonValue]) -> BaseModel:
        try:
            validated: BaseModel = tool.args_model.model_validate_json(
                canonical_json(dict(arguments)), strict=True
            )
        except ValidationError as exc:
            locations = sorted(
                {".".join(str(p) for p in err["loc"]) or "$" for err in exc.errors()}
            )
            raise ToolArgumentError(
                f"invalid arguments for {tool.name} at: {', '.join(locations)}"
            ) from exc
        return validated
