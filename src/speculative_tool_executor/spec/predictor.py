"""Draft predictors: what the agent is likely to call next, while the model is still thinking.

The default predictor is a back-off n-gram (Markov) model over canonicalized call sequences. It
is trained offline on reference trajectories and, at prediction time, produces *concrete* calls
rather than tool names, which is the part that actually matters: speculating "a read comes next"
is worthless unless the arguments are right too.

Arguments are learned as a **derivation scheme** rather than as constants. For every training
transition the model records, per argument field, either

- ``OBSERVED`` -- the value appeared in the previous tool's output, stored as the JSON path where
  it was found, so the rule generalizes to a different directory listing or search result; or
- ``PRIOR`` -- the value did not appear in the output and is remembered as a literal.

At prediction time an ``OBSERVED`` field is re-resolved against the *actual* previous output. A
template whose path does not resolve is dropped instead of guessed, so a prediction is never
assembled from arguments nobody has seen.

That is also where the taint rule bites (:mod:`speculative_tool_executor.spec.taint`): a
prediction may only run if every string in its arguments is something the main model has already
seen or a constant the tool's own schema declares. A predictor -- including the optional LLM
drafter, whose output is untrusted by construction -- therefore cannot widen the agent's reach by
inventing a path, a URL or a recipient.

Reference: back-off smoothing follows the standard Katz-style idea (Katz, "Estimation of
probabilities from sparse data for the language model component of a speech recognizer", 1987) in
its simplest form -- longer contexts first, each shorter context discounted.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from speculative_tool_executor.core.canonical import canonical_json

__all__ = [
    "ArgOrigin",
    "ArgScheme",
    "CallTemplate",
    "MarkovPredictor",
    "NullPredictor",
    "ObservedCall",
    "Prediction",
    "PredictionContext",
    "Predictor",
    "call_signature",
    "index_scalars",
    "resolve_path",
]

PathToken = str | int
JsonPath = tuple[PathToken, ...]
_MISSING = object()


class ArgOrigin(StrEnum):
    """Where a predicted argument value came from."""

    OBSERVED = "observed"
    PRIOR = "prior"


@dataclass(frozen=True, slots=True)
class ArgScheme:
    """How one argument field of a predicted call is produced."""

    origin: ArgOrigin
    path: JsonPath = ()
    literal: str = ""

    def resolve(self, output: JsonValue) -> JsonValue | object:
        if self.origin is ArgOrigin.PRIOR:
            decoded: JsonValue = _decode(self.literal)
            return decoded
        return resolve_path(output, self.path)


@dataclass(frozen=True, slots=True)
class CallTemplate:
    """A tool plus a derivation scheme per argument field."""

    tool: str
    fields: tuple[tuple[str, ArgScheme], ...]

    def instantiate(self, output: JsonValue) -> tuple[dict[str, JsonValue], dict[str, ArgOrigin]]:
        """Build concrete arguments, or raise ``LookupError`` if a path no longer resolves."""
        arguments: dict[str, JsonValue] = {}
        origins: dict[str, ArgOrigin] = {}
        for name, scheme in self.fields:
            value = scheme.resolve(output)
            if value is _MISSING:
                raise LookupError(f"{self.tool}.{name}: derivation path did not resolve")
            arguments[name] = value  # type: ignore[assignment]
            origins[name] = scheme.origin
        return arguments, origins


@dataclass(frozen=True, slots=True)
class Prediction:
    """One candidate next call with the probability the predictor assigned to it."""

    tool: str
    arguments: dict[str, JsonValue]
    probability: float
    source: str
    origins: Mapping[str, ArgOrigin] = field(default_factory=dict)

    def sort_key(self) -> tuple[float, str, str]:
        return (-self.probability, self.tool, canonical_json(self.arguments))


@dataclass(frozen=True, slots=True)
class ObservedCall:
    """One executed call in a trajectory, as the predictor sees it."""

    tool: str
    arguments: Mapping[str, JsonValue]
    output: JsonValue = None


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """What the predictor may look at: the recent call signatures and the last output."""

    history: tuple[str, ...] = ()
    last_output: JsonValue = None

    def tail(self, order: int) -> tuple[str, ...]:
        return self.history[-order:] if order else ()


@runtime_checkable
class Predictor(Protocol):
    name: str

    async def predict(self, context: PredictionContext, *, top_k: int) -> list[Prediction]: ...


def call_signature(tool: str, arguments: Mapping[str, JsonValue]) -> str:
    """The symbol the n-gram model works over: tool name plus its argument shape.

    Values are excluded on purpose. They are handled by the derivation scheme, and including them
    would make every directory and every query its own context, which no amount of training data
    would cover.
    """
    return f"{tool}({','.join(sorted(arguments))})"


def resolve_path(value: JsonValue, path: JsonPath) -> JsonValue | object:
    """Follow a recorded JSON path, returning the sentinel when it no longer exists."""
    current: JsonValue = value
    for token in path:
        if isinstance(token, str):
            if not isinstance(current, dict) or token not in current:
                return _MISSING
            current = current[token]
        else:
            if not isinstance(current, list) or not -len(current) <= token < len(current):
                return _MISSING
            current = current[token]
    return current


def index_scalars(value: JsonValue, *, max_nodes: int = 4096) -> dict[str, JsonPath]:
    """Map every scalar and list in ``value`` to the first path where it occurs.

    "First" is the lexicographically smallest path, so training is deterministic and two runs
    over the same output learn the same derivation.
    """
    found: dict[str, JsonPath] = {}
    stack: list[tuple[JsonValue, JsonPath]] = [(value, ())]
    visited = 0
    while stack and visited < max_nodes:
        current, path = stack.pop()
        visited += 1
        if isinstance(current, dict):
            for key in sorted(current, reverse=True):
                stack.append((current[key], (*path, key)))
            continue
        if isinstance(current, list):
            encoded = _encode(current)
            if encoded not in found or path < found[encoded]:
                found[encoded] = path
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], (*path, index)))
            continue
        encoded = _encode(current)
        if encoded not in found or path < found[encoded]:
            found[encoded] = path
    return found


def _encode(value: JsonValue) -> str:
    return canonical_json(value)


def _decode(text: str) -> JsonValue:
    import json  # noqa: PLC0415 - only needed on the literal path

    decoded: JsonValue = json.loads(text)
    return decoded


class NullPredictor:
    """Predicts nothing.

    The ablation baseline, and the safe default when there is no training data.
    """

    name = "none"

    async def predict(self, context: PredictionContext, *, top_k: int) -> list[Prediction]:
        del context, top_k
        return []


class MarkovPredictor:
    """Back-off n-gram model over call signatures, with derived arguments.

    ``order`` is the longest context considered. Prediction tries the longest context first and
    falls back to shorter ones, discounting each fallback level by ``backoff``, so a specific
    match always outranks a generic one of the same raw frequency.
    """

    name = "markov"

    def __init__(self, *, order: int = 2, backoff: float = 0.4, min_count: int = 1) -> None:
        if order < 0:
            raise ValueError("order must be non-negative")
        if not 0.0 < backoff <= 1.0:
            raise ValueError("backoff must be in (0, 1]")
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        self.order = order
        self.backoff = backoff
        self.min_count = min_count
        self._counts: dict[tuple[str, ...], Counter[CallTemplate]] = defaultdict(Counter)
        self._totals: Counter[tuple[str, ...]] = Counter()
        self._trained_transitions = 0

    @property
    def contexts(self) -> int:
        return len(self._counts)

    @property
    def transitions(self) -> int:
        return self._trained_transitions

    def train(self, trajectories: Sequence[Sequence[ObservedCall]]) -> MarkovPredictor:
        """Count transitions in the given trajectories. Training is additive and deterministic."""
        for trajectory in trajectories:
            history: list[str] = []
            for index, call in enumerate(trajectory):
                if index:
                    previous = trajectory[index - 1]
                    template = self._template_for(call, previous.output)
                    for length in range(min(self.order, len(history)) + 1):
                        context = tuple(history[len(history) - length :]) if length else ()
                        self._counts[context][template] += 1
                        self._totals[context] += 1
                    self._trained_transitions += 1
                history.append(call_signature(call.tool, call.arguments))
        return self

    @staticmethod
    def _template_for(call: ObservedCall, previous_output: JsonValue) -> CallTemplate:
        index = index_scalars(previous_output)
        fields: list[tuple[str, ArgScheme]] = []
        for name in sorted(call.arguments):
            value = call.arguments[name]
            encoded = _encode(value)
            path = index.get(encoded)
            if path is not None:
                fields.append((name, ArgScheme(origin=ArgOrigin.OBSERVED, path=path)))
            else:
                fields.append((name, ArgScheme(origin=ArgOrigin.PRIOR, literal=encoded)))
        return CallTemplate(tool=call.tool, fields=tuple(fields))

    async def predict(self, context: PredictionContext, *, top_k: int) -> list[Prediction]:
        if top_k < 1:
            return []
        scores: dict[tuple[str, str], Prediction] = {}
        longest = min(self.order, len(context.history))
        for length in range(longest, -1, -1):
            key = context.tail(length)
            total = self._totals.get(key, 0)
            if total == 0:
                continue
            discount = math.pow(self.backoff, longest - length)
            for template, count in self._counts[key].items():
                if count < self.min_count:
                    continue
                try:
                    arguments, origins = template.instantiate(context.last_output)
                except LookupError:
                    continue
                probability = (count / total) * discount
                identity = (template.tool, canonical_json(arguments))
                previous = scores.get(identity)
                if previous is None or probability > previous.probability:
                    scores[identity] = Prediction(
                        tool=template.tool,
                        arguments=arguments,
                        probability=probability,
                        source=self.name,
                        origins=origins,
                    )
        ranked = sorted(scores.values(), key=Prediction.sort_key)
        return ranked[:top_k]
