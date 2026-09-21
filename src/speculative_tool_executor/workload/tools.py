"""The benchmark agent's tools: one effect class each, strict schemas, declared latencies.

Every tool here exists to exercise one part of the speculation boundary:

``outline_text`` / ``tally_items`` (``PURE``)
    Depend on nothing but their arguments, so a speculative result never expires and can be
    reused by more than one caller.
``list_dir`` / ``read_file`` / ``search_code`` / ``fetch_doc`` (``READ``)
    Observe the workspace and the bundled documentation service. Neither is versioned, so their
    speculative results are bounded by a freshness window rather than by a read-set check.
``get_note`` (``READ``)
    Observes the *versioned* store through the branch overlay, so a speculative read that another
    writer invalidates is caught by optimistic concurrency control rather than by a timer.
``save_note`` / ``bump_index`` (``WRITE``)
    Mutate the versioned store. ``bump_index`` is a read-modify-write, which is the case where
    speculating without a read set would silently lose an update.
``send_report`` (``EXTERNAL``)
    Leaves the system. It is never speculated under any configuration, always needs an approval
    bound to the canonical argument hash, and asserts both facts itself rather than trusting the
    speculator to have checked.

The per-tool latency profiles live here too, next to the tools they describe. They are inputs to
a simulation, not measurements of anything real, and the README says so wherever a number derived
from them appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
)
from speculative_tool_executor.spec.canonicalize import (
    ArgRule,
    ToolArgumentCanonicalizer,
    collapse_whitespace,
    normalize_path,
    normalize_url,
    sorted_unique,
    to_lowercase,
)
from speculative_tool_executor.spec.context import current_execution
from speculative_tool_executor.spec.latency import LatencyProfile
from speculative_tool_executor.workload.corpus import (
    Corpus,
    WorkspaceError,
    outline_text,
    tally_items,
)

__all__ = [
    "ALL_TOOLS",
    "DEFAULT_TOOL_PROFILE",
    "GRANTED_CAPABILITIES",
    "MODEL_TURN_PROFILE",
    "RECIPIENTS",
    "TOOL_PROFILES",
    "MessageSink",
    "SpeculatedExternalEffectError",
    "build_canonicalizer",
    "build_registry",
]

#: Latency profiles for the simulated tools, in seconds. These are *assumptions*: a fast local
#: read, a slower search, a network-shaped document fetch, a durable write.
TOOL_PROFILES: dict[str, LatencyProfile] = {
    "list_dir": LatencyProfile(median_seconds=0.12, sigma=0.35),
    "read_file": LatencyProfile(median_seconds=0.18, sigma=0.45),
    "search_code": LatencyProfile(median_seconds=0.55, sigma=0.60),
    "fetch_doc": LatencyProfile(median_seconds=0.70, sigma=0.70),
    "get_note": LatencyProfile(median_seconds=0.09, sigma=0.30),
    "save_note": LatencyProfile(median_seconds=0.25, sigma=0.40),
    "bump_index": LatencyProfile(median_seconds=0.28, sigma=0.40),
    "outline_text": LatencyProfile(median_seconds=0.02, sigma=0.20),
    "tally_items": LatencyProfile(median_seconds=0.02, sigma=0.20),
    "send_report": LatencyProfile(median_seconds=1.10, sigma=0.55),
}
DEFAULT_TOOL_PROFILE = LatencyProfile(median_seconds=0.20, sigma=0.50)

#: The model turn. The median is ~4x the median tool, which is the regime where speculation can
#: pay: the model is the bottleneck and the tool has time to finish inside its shadow.
MODEL_TURN_PROFILE = LatencyProfile(median_seconds=0.80, sigma=0.35)

#: Assumed marginal cost of running one tool call, in USD. Speculation is charged this when it
#: launches and refunded nothing when it misses, which is what the waste ledger measures.
_TOOL_COST_USD: dict[str, float] = {
    "list_dir": 0.00004,
    "read_file": 0.00006,
    "search_code": 0.00020,
    "fetch_doc": 0.00030,
    "get_note": 0.00003,
    "save_note": 0.00010,
    "bump_index": 0.00012,
    "outline_text": 0.00001,
    "tally_items": 0.00001,
    "send_report": 0.00200,
}

#: Recipients ``send_report`` is allowed to name. Declared in the schema, so they are constants
#: the taint rule accepts without the model having had to utter them.
RECIPIENTS = ("platform-rota", "docs-team", "security")

GRANTED_CAPABILITIES = frozenset(
    {
        Capability.COMPUTE,
        Capability.READ_DATA,
        Capability.NETWORK_READ,
        Capability.WRITE_DATA,
        Capability.SEND_MESSAGE,
    }
)

_PATH = Field(min_length=1, max_length=200)
_KEY = Field(min_length=1, max_length=200)


class SpeculatedExternalEffectError(RuntimeError):
    """An external tool found itself running on a speculative branch. This must never happen."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathArgs(_Args):
    path: str = _PATH


class QueryArgs(_Args):
    query: str = Field(min_length=1, max_length=200)


class UrlArgs(_Args):
    url: str = Field(min_length=1, max_length=400)


class NoteKeyArgs(_Args):
    key: str = _KEY


class SaveNoteArgs(_Args):
    key: str = _KEY
    text: str = Field(min_length=1, max_length=4000)


class TextArgs(_Args):
    text: str = Field(min_length=1, max_length=4000)


class ItemsArgs(_Args):
    items: list[str] = Field(min_length=1, max_length=200)


class SendReportArgs(_Args):
    recipient: Literal["platform-rota", "docs-team", "security"]
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)


@dataclass(slots=True)
class MessageSink:
    """Where ``send_report`` delivers. Tests assert this stays empty under mispredictions."""

    sent: list[dict[str, str]] = field(default_factory=list)

    def deliver(self, recipient: str, subject: str, body: str) -> dict[str, JsonValue]:
        self.sent.append({"recipient": recipient, "subject": subject, "body": body})
        return {"delivered": True, "recipient": recipient, "sequence": len(self.sent)}

    @property
    def count(self) -> int:
        return len(self.sent)


def build_canonicalizer() -> ToolArgumentCanonicalizer:
    """The declared equivalences. Each rule is a claim about what its tool ignores.

    ``search_code`` lower-cases and collapses whitespace because its handler does; ``tally_items``
    sorts and de-duplicates because it counts a set; paths and URLs follow the syntax-based rules
    in :mod:`speculative_tool_executor.spec.canonicalize`. Nothing else is normalized, so any
    other spelling difference simply misses instead of merging two different calls.
    """
    return ToolArgumentCanonicalizer(
        {
            "list_dir": [ArgRule(field="path", normalizers=(normalize_path,))],
            "read_file": [ArgRule(field="path", normalizers=(normalize_path,))],
            "search_code": [
                ArgRule(field="query", normalizers=(collapse_whitespace, to_lowercase))
            ],
            "fetch_doc": [ArgRule(field="url", normalizers=(normalize_url,))],
            "get_note": [ArgRule(field="key", normalizers=(normalize_path,))],
            "save_note": [ArgRule(field="key", normalizers=(normalize_path,))],
            "bump_index": [ArgRule(field="key", normalizers=(normalize_path,))],
            "outline_text": [ArgRule(field="text", normalizers=(collapse_whitespace,))],
            "tally_items": [ArgRule(field="items", normalizers=(sorted_unique,))],
        }
    )


ALL_TOOLS = (
    "list_dir",
    "read_file",
    "search_code",
    "fetch_doc",
    "get_note",
    "save_note",
    "bump_index",
    "outline_text",
    "tally_items",
    "send_report",
)


def _definitions(
    corpus: Corpus, sink: MessageSink
) -> tuple[
    ToolDefinition[PathArgs]
    | ToolDefinition[QueryArgs]
    | ToolDefinition[UrlArgs]
    | ToolDefinition[NoteKeyArgs]
    | ToolDefinition[SaveNoteArgs]
    | ToolDefinition[TextArgs]
    | ToolDefinition[ItemsArgs]
    | ToolDefinition[SendReportArgs],
    ...,
]:
    """Build every tool definition, closing over the corpus and the message sink."""

    def list_dir(args: PathArgs) -> JsonValue:
        return corpus.list_dir(args.path)

    def read_file(args: PathArgs) -> JsonValue:
        return corpus.read_file(args.path)

    def search_code(args: QueryArgs) -> JsonValue:
        return corpus.search_code(args.query)

    def fetch_doc(args: UrlArgs) -> JsonValue:
        return corpus.fetch_doc(args.url)

    def get_note(args: NoteKeyArgs) -> JsonValue:
        """Read the versioned store through the branch, pinning the version for the commit check."""
        overlay = current_execution().overlay
        value = overlay.read(args.key)
        return {"key": args.key, "present": value is not None, "text": value}

    def save_note(args: SaveNoteArgs) -> JsonValue:
        """Read-then-write, so an overwrite of an existing note is a versioned dependency."""
        overlay = current_execution().overlay
        previous = overlay.read(args.key)
        overlay.write(args.key, args.text)
        return {
            "key": args.key,
            "stored": True,
            "replaced": previous is not None,
            "length": len(args.text),
        }

    def bump_index(args: NoteKeyArgs) -> JsonValue:
        """Read-modify-write: the case a read set has to protect or the increment is lost."""
        overlay = current_execution().overlay
        current = overlay.read(args.key)
        count = int(current) + 1 if current is not None and current.isdigit() else 1
        overlay.write(args.key, str(count))
        return {"key": args.key, "count": count}

    def outline(args: TextArgs) -> JsonValue:
        return outline_text(args.text)

    def tally(args: ItemsArgs) -> JsonValue:
        return tally_items(args.items)

    def send_report(args: SendReportArgs) -> JsonValue:
        """Refuse to run on a speculative branch, whatever the speculator believed."""
        execution = current_execution()
        if execution.speculative:
            raise SpeculatedExternalEffectError(
                f"send_report reached branch {execution.branch_id}, which is speculative"
            )
        return sink.deliver(args.recipient, args.subject, args.body)

    read_caps = frozenset({Capability.READ_DATA})
    return (
        ToolDefinition(
            name="list_dir",
            description="List the files under a workspace directory, newest listing first.",
            args_model=PathArgs,
            effect=Effect.READ,
            capabilities=read_caps,
            handler=list_dir,
            freshness_seconds=30.0,
            expected_latency_seconds=TOOL_PROFILES["list_dir"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["list_dir"],
        ),
        ToolDefinition(
            name="read_file",
            description="Read one workspace file and report its lines, symbols and note key.",
            args_model=PathArgs,
            effect=Effect.READ,
            capabilities=read_caps,
            handler=read_file,
            freshness_seconds=30.0,
            expected_latency_seconds=TOOL_PROFILES["read_file"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["read_file"],
        ),
        ToolDefinition(
            name="search_code",
            description="Search the workspace for a substring and report the matching paths.",
            args_model=QueryArgs,
            effect=Effect.READ,
            capabilities=read_caps,
            handler=search_code,
            freshness_seconds=30.0,
            expected_latency_seconds=TOOL_PROFILES["search_code"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["search_code"],
        ),
        ToolDefinition(
            name="fetch_doc",
            description="Fetch a page from the bundled documentation service over HTTPS.",
            args_model=UrlArgs,
            effect=Effect.READ,
            capabilities=frozenset({Capability.NETWORK_READ}),
            handler=fetch_doc,
            freshness_seconds=20.0,
            expected_latency_seconds=TOOL_PROFILES["fetch_doc"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["fetch_doc"],
        ),
        ToolDefinition(
            name="get_note",
            description="Read a stored note from the versioned store.",
            args_model=NoteKeyArgs,
            effect=Effect.READ,
            capabilities=read_caps,
            handler=get_note,
            freshness_seconds=60.0,
            expected_latency_seconds=TOOL_PROFILES["get_note"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["get_note"],
        ),
        ToolDefinition(
            name="save_note",
            description="Store a note in the versioned store under the given key.",
            args_model=SaveNoteArgs,
            effect=Effect.WRITE,
            capabilities=frozenset({Capability.READ_DATA, Capability.WRITE_DATA}),
            handler=save_note,
            requires_approval=False,
            freshness_seconds=60.0,
            expected_latency_seconds=TOOL_PROFILES["save_note"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["save_note"],
        ),
        ToolDefinition(
            name="bump_index",
            description="Increment the visit counter stored under an index key.",
            args_model=NoteKeyArgs,
            effect=Effect.WRITE,
            capabilities=frozenset({Capability.READ_DATA, Capability.WRITE_DATA}),
            handler=bump_index,
            requires_approval=False,
            freshness_seconds=60.0,
            expected_latency_seconds=TOOL_PROFILES["bump_index"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["bump_index"],
        ),
        ToolDefinition(
            name="outline_text",
            description="Outline a passage of text into bullets. Pure: no I/O, no state.",
            args_model=TextArgs,
            effect=Effect.PURE,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=outline,
            expected_latency_seconds=TOOL_PROFILES["outline_text"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["outline_text"],
        ),
        ToolDefinition(
            name="tally_items",
            description="Count distinct items in a list treated as a set. Pure: no I/O, no state.",
            args_model=ItemsArgs,
            effect=Effect.PURE,
            capabilities=frozenset({Capability.COMPUTE}),
            handler=tally,
            expected_latency_seconds=TOOL_PROFILES["tally_items"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["tally_items"],
        ),
        ToolDefinition(
            name="send_report",
            description="Send a report to a named internal recipient. Irreversible.",
            args_model=SendReportArgs,
            effect=Effect.EXTERNAL,
            capabilities=frozenset({Capability.SEND_MESSAGE}),
            handler=send_report,
            freshness_seconds=60.0,
            expected_latency_seconds=TOOL_PROFILES["send_report"].median_seconds,
            expected_cost_usd=_TOOL_COST_USD["send_report"],
        ),
    )


def build_registry(
    corpus: Corpus,
    sink: MessageSink | None = None,
    *,
    allowed: Sequence[str] | None = None,
    canonicalize: bool = True,
) -> tuple[ToolRegistry, MessageSink]:
    """Build the workload registry. ``allowed`` narrows it to a per-task allowlist.

    ``canonicalize=False`` exists for the ablation that measures what the declared equivalences
    are worth: without them, two spellings of the same call hash differently and speculation
    misses.
    """
    delivery = sink if sink is not None else MessageSink()
    registry = ToolRegistry(
        GRANTED_CAPABILITIES,
        canonicalizer=build_canonicalizer() if canonicalize else None,
    )
    for definition in _definitions(corpus, delivery):
        registry.register(definition)
    if allowed is not None:
        registry = registry.scoped(allowed)
    return registry, delivery


def workspace_error() -> type[WorkspaceError]:
    """Re-exported so callers can catch workspace refusals without importing the corpus."""
    return WorkspaceError
