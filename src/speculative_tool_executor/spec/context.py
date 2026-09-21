"""The ambient execution context a tool handler runs in.

A handler's signature is ``(args) -> result``: it takes no runtime object, so that the schema the
model sees and the arguments the approval hashes are the whole interface. The branch a call runs
against therefore has to arrive out of band, and a :class:`~contextvars.ContextVar` is the right
carrier -- asyncio copies the context into each task, so concurrent speculative branches cannot
see each other's overlay, and nothing leaks back into the caller when a task is cancelled.

Handlers read :func:`current_execution` to reach their branch. They must not branch on
``speculative``: a speculative run that behaved differently from the authoritative one would
break the equivalence the whole design rests on. The flag exists for metrics, tracing and for
guards that *refuse* to run (an external tool asserting it is never speculated), not for
alternative behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from speculative_tool_executor.spec.clock import Clock
from speculative_tool_executor.spec.overlay import WriteOverlay

__all__ = ["ExecutionContext", "current_execution", "use_execution"]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """What a handler may depend on besides its arguments."""

    overlay: WriteOverlay
    clock: Clock
    branch_id: str
    speculative: bool


_CURRENT: ContextVar[ExecutionContext | None] = ContextVar("spec_execution", default=None)


def current_execution() -> ExecutionContext:
    """The context of the call being executed. Raises outside an execution."""
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("no execution context: tool handlers run through the executor")
    return context


@contextmanager
def use_execution(context: ExecutionContext) -> Iterator[ExecutionContext]:
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)
