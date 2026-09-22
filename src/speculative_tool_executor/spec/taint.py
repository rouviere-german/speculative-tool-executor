"""Taint tracking for speculative arguments.

Speculation widens what a system does *before* the model has asked for it, so the arguments it
runs with need their own provenance rule, independent of whatever produced them. The rule here:

    a speculative call may only use string values the main model has already seen, or constants
    the tool's own schema declares.

"Already seen" means: a value the operator put in the task, a value in a tool result that has
already been returned to the model, or a value in an argument the model itself has already sent.
Numbers and booleans are exempt -- they cannot name a file, a host or a recipient, and treating
them as tainted would reject ordinary paging arguments for no gain.

What this does and does not buy is worth being precise about. It does **not** make the agent
injection-proof: text inside a tool result is content the model has seen, so an injected string
is a legal speculative argument. What it prevents is the *predictor* -- including the optional
LLM drafter, whose output is untrusted by construction -- from widening the agent's reach on its
own: a drafter cannot invent a path, host or identifier that never appeared anywhere and have it
executed ahead of the model. Containment of injected instructions is the job of the effect
classes, the allowlist and the approval gate; this rule bounds the blast radius of a bad
prediction.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from speculative_tool_executor.core.canonical import canonical_json

__all__ = ["TaintReport", "TaintTracker", "schema_constants"]

_MAX_VALUE_LENGTH = 4096
_MAX_WALK_NODES = 8192
_MAX_SCHEMA_NODES = 4096


@dataclass(frozen=True, slots=True)
class TaintReport:
    """Which argument fields failed the provenance rule, and with what value shape."""

    fields: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.fields

    def reason(self) -> str:
        return "" if self.ok else f"untainted arguments: {', '.join(self.fields)}"


def schema_constants(schema: Mapping[str, JsonValue]) -> frozenset[str]:
    """Every literal a JSON Schema fixes: ``enum`` members, ``const`` values and defaults.

    These are values the tool itself declares, so using one cannot expose anything the tool was
    not already willing to be called with.
    """
    found: set[str] = set()
    stack: list[JsonValue] = [dict(schema)]
    seen_nodes = 0
    while stack and seen_nodes < _MAX_SCHEMA_NODES:
        node = stack.pop()
        seen_nodes += 1
        if isinstance(node, list):
            stack.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        for keyword in ("enum", "const", "default"):
            if keyword not in node:
                continue
            value = node[keyword]
            values = value if keyword == "enum" and isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str):
                    found.add(canonical_json(item))
        stack.extend(node.values())
    return frozenset(found)


class TaintTracker:
    """A bounded set of the string values the main model has been shown."""

    def __init__(self, *, max_values: int = 20_000) -> None:
        if max_values < 1:
            raise ValueError("max_values must be at least 1")
        self._max = max_values
        self._values: OrderedDict[str, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(self._values)

    def _add(self, text: str) -> None:
        if len(text) > _MAX_VALUE_LENGTH:
            return
        self._values.pop(text, None)
        self._values[text] = None
        while len(self._values) > self._max:
            self._values.popitem(last=False)

    def observe(self, value: JsonValue) -> None:
        """Record every string reachable in ``value`` (and, for text, its whitespace tokens)."""
        stack: list[JsonValue] = [value]
        visited = 0
        while stack and visited < _MAX_WALK_NODES:
            current = stack.pop()
            visited += 1
            if isinstance(current, dict):
                for key, item in current.items():
                    self._add(canonical_json(key))
                    stack.append(item)
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, str):
                self._add(canonical_json(current))
                for token in current.split():
                    if token != current:
                        self._add(canonical_json(token))

    def observe_all(self, values: Iterable[JsonValue]) -> None:
        for value in values:
            self.observe(value)

    def seen(self, value: JsonValue) -> bool:
        return canonical_json(value) in self._values

    def check(
        self, arguments: Mapping[str, JsonValue], *, constants: frozenset[str] = frozenset()
    ) -> TaintReport:
        """Report argument fields holding a string the model has not seen and no schema declares."""
        offending: list[str] = []
        for name in sorted(arguments):
            if self._violations(arguments[name], constants):
                offending.append(name)
        return TaintReport(fields=tuple(offending))

    def _violations(self, value: JsonValue, constants: frozenset[str]) -> bool:
        """Whether ``value`` contains a string the model has not seen and no schema declares.

        The walk is bounded so a pathological argument cannot stall the speculator, and the
        bound fails **closed**: an argument too large to finish checking is reported as a
        violation. Anything else would make padding the argument until the budget runs out the
        cheapest way to get an arbitrary string past the rule.
        """
        stack: list[JsonValue] = [value]
        visited = 0
        while stack:
            if visited >= _MAX_WALK_NODES:
                return True
            current = stack.pop()
            visited += 1
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, str):
                encoded = canonical_json(current)
                if encoded not in self._values and encoded not in constants:
                    return True
        return False
