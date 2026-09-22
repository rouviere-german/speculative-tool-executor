"""One simulated agent run: the loop that speculation overlaps with.

The loop is the ordinary serial one -- model turn, tool call, model turn -- with exactly one
addition: before the model turn's latency is spent, the session is asked to predict and launch.
That single line is the whole latency argument. Everything else in this module exists to make the
two arms of the A/B comparable:

- the model is a :class:`~...providers.fake.FakeProvider` scripted to the task's reference
  trajectory, so both arms issue *identical* calls and no measured difference can come from the
  model behaving differently;
- model latency is drawn from the request hash and tool latency from the call hash, so both arms
  draw the *same* numbers for the same work (:mod:`...spec.latency`);
- every run records the digest of its outputs, its committed write sequence and the store digest,
  which is what the differential equivalence test compares.

Tool results reach the model inside untrusted-data envelopes, and the values in them are what
feeds the taint tracker: a prediction may only name something that has already been shown to the
model, which is checked here by observing the output *before* the next turn is predicted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from speculative_tool_executor.agentic.approval import ApprovalGate, ApprovalStore
from speculative_tool_executor.agentic.budget import BudgetLedger, BudgetLimits
from speculative_tool_executor.agentic.envelope import (
    UNTRUSTED_DATA_POLICY,
    wrap_tool_result,
    wrap_user_content,
)
from speculative_tool_executor.agentic.errors import (
    ApprovalRequiredError,
    BudgetExceededError,
    ProviderError,
    ToolArgumentError,
    ToolNotAllowedError,
)
from speculative_tool_executor.agentic.pricing import cost_usd_micros
from speculative_tool_executor.agentic.providers.base import (
    ChatMessage,
    ContentBlock,
    LLMRequest,
    LLMResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from speculative_tool_executor.agentic.providers.fake import FakeProvider, ScriptedPolicy
from speculative_tool_executor.agentic.tools import ToolRegistry
from speculative_tool_executor.core.canonical import canonical_json, sha256_hex
from speculative_tool_executor.spec.executor import SpeculativeToolExecutor, ToolOutcome
from speculative_tool_executor.spec.predictor import PredictionContext, call_signature
from speculative_tool_executor.spec.speculator import SpeculationSession
from speculative_tool_executor.workload.tasks import Task

__all__ = [
    "SYSTEM_PROMPT",
    "CallRecord",
    "SimulatedAgent",
    "TaskRun",
    "preapprove_external_calls",
    "scripted_policy_for",
]

SYSTEM_PROMPT = (
    "You are a codebase assistant. Use the provided tools to answer the operator's request.\n"
    f"{UNTRUSTED_DATA_POLICY}"
)

#: The scripted model is free, but a run's real cost is overwhelmingly its tokens, and the waste
#: ratio is only meaningful against the whole bill. Token counts are the ones the fake provider
#: actually estimated from the prompt; this constant only says what they would be priced at. It is
#: an assumption of the simulation, stated here and repeated wherever a cost number is reported.
MODEL_PRICING_REFERENCE = "claude-sonnet-5"


@dataclass(frozen=True, slots=True)
class CallRecord:
    """What one authoritative call did, reduced to the facts equivalence is judged on."""

    tool: str
    call_hash: str
    mode: str
    ok: bool
    output_sha256: str
    committed_writes: tuple[str, ...]
    waited_seconds: float
    latency_saved_seconds: float
    started_at: float = 0.0
    error: str | None = None

    @property
    def finished_at(self) -> float:
        return self.started_at + self.waited_seconds

    @classmethod
    def of(cls, outcome: ToolOutcome, started_at: float = 0.0) -> CallRecord:
        return cls(
            started_at=started_at,
            tool=outcome.tool,
            call_hash=outcome.call_hash,
            mode=outcome.mode,
            ok=outcome.ok,
            output_sha256=outcome.output_sha256,
            committed_writes=outcome.committed_writes,
            waited_seconds=outcome.waited_seconds,
            latency_saved_seconds=outcome.latency_saved_seconds,
            error=outcome.error,
        )


@dataclass(frozen=True, slots=True)
class TaskRun:
    """The result of one task, in the two currencies that matter: time and what it did."""

    task_id: str
    shape: str
    completed: bool
    final_text: str
    error: str | None
    elapsed_seconds: float
    model_seconds: float
    tool_wait_seconds: float
    latency_saved_seconds: float
    turns: int
    calls: tuple[CallRecord, ...] = field(default_factory=tuple)
    store_digest: str = ""
    messages_sent: int = 0

    @property
    def outputs_digest(self) -> str:
        """Digest of the ordered (tool, outcome, output) triples: the run's observable answer."""
        return sha256_hex(
            canonical_json(
                [
                    [record.tool, record.ok, record.output_sha256, record.error]
                    for record in self.calls
                ]
            )
        )

    @property
    def write_sequence(self) -> tuple[tuple[str, ...], ...]:
        """The committed writes, in commit order. Speculation must not reorder or drop one."""
        return tuple(record.committed_writes for record in self.calls if record.committed_writes)

    @property
    def equivalence_key(self) -> str:
        """Everything a speculative run must reproduce exactly. Latency is deliberately absent."""
        return sha256_hex(
            canonical_json(
                {
                    "outputs": self.outputs_digest,
                    "writes": [list(group) for group in self.write_sequence],
                    "final_text": self.final_text,
                    "store": self.store_digest,
                    "completed": self.completed,
                    "messages_sent": self.messages_sent,
                }
            )
        )

    @property
    def hits(self) -> int:
        return sum(1 for record in self.calls if record.mode == "speculative-committed")

    @property
    def speculatable_calls(self) -> int:
        """Calls that could in principle have been served speculatively (not the first one)."""
        return max(0, len(self.calls) - 1)


def _echoed(response: LLMResponse) -> tuple[ContentBlock, ...]:
    """The assistant content to send back next turn.

    A provider that returns explicit ``content`` (Anthropic does, because thinking blocks must be
    echoed verbatim) is taken at its word. One that returns only ``text`` and ``tool_calls`` --
    the fakes, and any provider without opaque blocks -- has its content reconstructed in the
    order the protocol expects, because an empty assistant turn is not a valid message.
    """
    if response.content:
        return response.content
    blocks: list[ContentBlock] = []
    if response.text:
        blocks.append(TextBlock(text=response.text))
    blocks.extend(response.tool_calls)
    return tuple(blocks)


@dataclass(slots=True)
class _RunState:
    """Everything one run accumulates. A bag, so the loop reads as a loop and not as bookkeeping."""

    started: float
    turns: int = 0
    messages: list[ChatMessage] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    records: list[CallRecord] = field(default_factory=list)
    last_output: JsonValue = None
    model_seconds: float = 0.0
    tool_wait: float = 0.0
    saved: float = 0.0
    final_text: str = ""
    error: str | None = None
    completed: bool = False


def scripted_policy_for(task: Task) -> ScriptedPolicy:
    """Turn a reference trajectory into a scripted model: one tool call per turn, then a summary."""
    steps: list[LLMResponse] = []
    for index, call in enumerate(task.calls):
        block = ToolUseBlock(id=f"call_{index:03d}", name=call.tool, arguments=dict(call.arguments))
        steps.append(
            LLMResponse(
                model="fake-agent",
                stop_reason="tool_use",
                tool_calls=(block,),
                content=(block,),
            )
        )
    steps.append(LLMResponse(model="fake-agent", stop_reason="end_turn", text=task.final_text))
    return ScriptedPolicy(steps)


def preapprove_external_calls(
    task: Task,
    *,
    registry: ToolRegistry,
    store: ApprovalStore,
    principal: str = "agent",
    approver: str = "operator",
) -> int:
    """Grant approvals for a task's external calls, as a reviewer would have before the run.

    The approval is bound to the canonical hash of the exact call, so this cannot pre-authorise
    anything but the reference call itself. A speculative or mutated variant hashes differently
    and still has to stop at the gate.
    """
    granted = 0
    for index, call in enumerate(task.calls):
        block = ToolUseBlock(id=f"call_{index:03d}", name=call.tool, arguments=dict(call.arguments))
        validated = registry.validate_call(block)
        if not validated.tool.needs_approval:
            continue
        record = store.request(validated, requested_by=principal)
        store.decide(record.id, approver=approver, approve=True)
        granted += 1
    return granted


class SimulatedAgent:
    """Runs one task against a :class:`SpeculativeToolExecutor` on an injected clock."""

    def __init__(
        self,
        *,
        executor: SpeculativeToolExecutor,
        registry: ToolRegistry,
        gate: ApprovalGate,
        budget: BudgetLedger | None = None,
        max_turns: int = 24,
        speculate: bool = True,
        before_tool_call: Callable[[int], None] | None = None,
        provider_factory: Callable[[Task], FakeProvider] | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._executor = executor
        self._registry = registry
        self._gate = gate
        self._before_tool_call = before_tool_call
        self._provider_factory = provider_factory
        self._budget = budget or BudgetLedger(
            BudgetLimits.of(steps=max_turns + 1, tokens=10_000_000, usd=1.0)
        )
        self._max_turns = max_turns
        self._speculate = speculate

    @property
    def executor(self) -> SpeculativeToolExecutor:
        return self._executor

    async def run(self, task: Task) -> TaskRun:
        """Drive the scripted model through one task, returning what it did and how long it took."""
        clock = self._executor.clock
        latency = self._executor.runner.latency
        provider = (
            self._provider_factory(task)
            if self._provider_factory is not None
            else FakeProvider(scripted_policy_for(task), seed=0)
        )
        self._executor.taint.observe(task.prompt)

        state = _RunState(started=clock.now)
        state.messages.append(
            ChatMessage(
                role="user",
                content=(TextBlock(text=wrap_user_content("task", task.prompt).render()),),
            )
        )

        with clock.participant():
            async with self._executor.session() as session:
                for _ in range(self._max_turns):
                    state.turns += 1
                    request = LLMRequest(
                        messages=tuple(state.messages),
                        system=SYSTEM_PROMPT,
                        tools=self._registry.specs(),
                        max_tokens=1024,
                    )
                    # The one line the whole latency argument rests on: predict and launch
                    # *before* the model turn's latency is spent.
                    if self._speculate and state.history:
                        session.start_turn(
                            PredictionContext(
                                history=tuple(state.history), last_output=state.last_output
                            )
                        )
                    try:
                        self._budget.reserve_step()
                    except BudgetExceededError as exc:
                        state.error = type(exc).__name__
                        break

                    think = latency.model_latency(request.sha256())
                    state.model_seconds += think
                    await clock.sleep(think)

                    try:
                        response = provider.complete(request)
                    except ProviderError as exc:
                        state.error = type(exc).__name__
                        break
                    self._charge(response)

                    if response.stop_reason != "tool_use":
                        state.final_text = response.text
                        state.completed = True
                        break

                    state.messages.append(ChatMessage(role="assistant", content=_echoed(response)))
                    results = await self._run_tool_calls(response, session=session, state=state)
                    state.messages.append(ChatMessage(role="user", content=results))
                else:
                    state.error = "MaxTurnsExceeded"

        return TaskRun(
            task_id=task.task_id,
            shape=task.shape,
            completed=state.completed,
            final_text=state.final_text,
            error=state.error,
            elapsed_seconds=clock.now - state.started,
            model_seconds=state.model_seconds,
            tool_wait_seconds=state.tool_wait,
            latency_saved_seconds=state.saved,
            turns=state.turns,
            calls=tuple(state.records),
            store_digest=self._executor.store.digest(),
            messages_sent=sum(1 for record in state.records if record.tool == "send_report"),
        )

    def _charge(self, response: LLMResponse) -> None:
        """Charge the turn to the agent budget and to the run's would-have-cost baseline.

        The waste ratio divides by the baseline, so it has to include the model turns. Tokens
        are the ones the provider actually reported; ``MODEL_PRICING_REFERENCE`` only says what
        they are priced at.
        """
        self._budget.record_usage(model=response.model, usage=response.usage)
        self._executor.controller.record_baseline_spend(
            cost_usd_micros(
                MODEL_PRICING_REFERENCE,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            / 1_000_000
        )

    async def _run_tool_calls(
        self, response: LLMResponse, *, session: SpeculationSession, state: _RunState
    ) -> tuple[ToolResultBlock, ...]:
        """Execute this turn's calls, recording what each one did and what the model may see."""
        clock = self._executor.clock
        results: list[ToolResultBlock] = []
        for call in response.tool_calls:
            if self._before_tool_call is not None:
                # Fires after the model turn and before the claim, which is exactly the window
                # in which another writer can invalidate a finished branch.
                self._before_tool_call(len(state.records))
            call_started = clock.now - state.started
            outcome, block = await self._execute(call, session=session)
            results.append(block)
            if outcome is None:
                continue
            state.records.append(CallRecord.of(outcome, started_at=call_started))
            state.tool_wait += outcome.waited_seconds
            state.saved += outcome.latency_saved_seconds
            state.history.append(call_signature(outcome.tool, call.arguments))
            state.last_output = outcome.output
            # Observed before the next turn is predicted, so a prediction may name it.
            self._executor.taint.observe(outcome.output)
        return tuple(results)

    async def _execute(
        self, call: ToolUseBlock, *, session: SpeculationSession
    ) -> tuple[ToolOutcome | None, ToolResultBlock]:
        """Execute one call, turning a refusal into an error result the model can read."""
        try:
            outcome = await self._executor.execute(call, session=session)
        except ApprovalRequiredError as exc:
            return None, ToolResultBlock(
                tool_use_id=call.id,
                content=f"approval required (request {exc.request_id})",
                is_error=True,
            )
        except (ToolNotAllowedError, ToolArgumentError) as exc:
            return None, ToolResultBlock(
                tool_use_id=call.id, content=type(exc).__name__, is_error=True
            )
        envelope = wrap_tool_result(outcome.tool, outcome.output)
        return outcome, ToolResultBlock(
            tool_use_id=call.id,
            content=envelope.render() if outcome.ok else (outcome.error or "tool failed"),
            is_error=not outcome.ok,
        )


async def run_tasks(
    agent: SimulatedAgent, tasks: Sequence[Task], *, timeout_seconds: float = 300.0
) -> list[TaskRun]:
    """Run tasks one after another under a wall-clock guard on the simulation itself."""
    runs: list[TaskRun] = []
    async with asyncio.timeout(timeout_seconds):
        for task in tasks:
            runs.append(await agent.run(task))
    return runs
