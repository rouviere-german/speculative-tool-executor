"""The task generator: multi-step tasks with reference trajectories.

Every reference argument is produced by *running* the read tools against the corpus, never by
string-building what a tool "would" return. That matters for honesty: if a trajectory contained an
argument no tool output could have produced, the predictor would be learning from data the real
agent could never see, and the measured hit rate would be fiction.

The shapes below deliberately share prefixes. After ``list_dir`` then ``read_file`` three
different calls can follow, so a top-2 speculator cannot cover the context perfectly and the
benchmark reports a hit rate well under 100%. A workload where every continuation is forced would
make speculation look free; this one makes it pay for its mistakes.

The first call of a task is never predictable -- nothing has happened yet -- so the ceiling on
saved tool time for a ``k``-call task is ``(k-1)/k``. The README states that alongside the
measured figure.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue

from speculative_tool_executor.spec.predictor import ObservedCall
from speculative_tool_executor.workload.corpus import Corpus, build_corpus
from speculative_tool_executor.workload.tools import RECIPIENTS

__all__ = [
    "DEFAULT_TASK_COUNT",
    "SHAPE_WEIGHTS",
    "ReferenceCall",
    "Task",
    "bullets_of",
    "entries_of",
    "generate_tasks",
    "paths_of",
    "pick_entry",
    "shape_counts",
    "text_of",
    "training_trajectories",
]

DEFAULT_TASK_COUNT: Final = 300

#: Relative frequency of each task shape. The weights, not the shapes, are what make the
#: benchmark neither trivially predictable nor hopeless.
SHAPE_WEIGHTS: Final[Mapping[str, int]] = {
    "audit_module": 30,
    "search_outline": 20,
    "doc_note": 15,
    "index_bump": 12,
    "note_lookup": 10,
    "tally_dir": 8,
    "doc_report": 5,
}


@dataclass(frozen=True, slots=True)
class ReferenceCall:
    """One call in a reference trajectory, with the output the corpus produced for it."""

    tool: str
    arguments: dict[str, JsonValue]
    output: JsonValue = None

    def observed(self) -> ObservedCall:
        return ObservedCall(tool=self.tool, arguments=self.arguments, output=self.output)


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark task: what the operator asked, and the calls a correct run makes."""

    task_id: str
    shape: str
    prompt: str
    calls: tuple[ReferenceCall, ...]
    final_text: str

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(call.tool for call in self.calls)

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """The per-task allowlist: the least privilege this task can be run with."""
        return tuple(sorted(set(self.tools)))

    @property
    def external_calls(self) -> tuple[ReferenceCall, ...]:
        return tuple(call for call in self.calls if call.tool == "send_report")

    def trajectory(self) -> tuple[ObservedCall, ...]:
        return tuple(call.observed() for call in self.calls)


def text_of(value: JsonValue, field: str) -> str:
    """Pull a string field out of a tool output, failing loudly if the shape changed.

    Public because tests read tool outputs the same way the generator does, and duplicating
    the narrowing in each test file would be three more places to get it subtly wrong.
    """
    found = value.get(field) if isinstance(value, dict) else None
    if not isinstance(found, str):
        raise TypeError(f"tool output has no string field {field!r}")
    return found


def entries_of(value: JsonValue) -> list[str]:
    found = value.get("entries") if isinstance(value, dict) else None
    if not isinstance(found, list):
        raise TypeError("list_dir output has no entries list")
    return [str(item) for item in found]


def paths_of(value: JsonValue) -> list[str]:
    found = value.get("paths") if isinstance(value, dict) else None
    if not isinstance(found, list):
        raise TypeError("search_code output has no paths list")
    return [str(item) for item in found]


def bullets_of(value: JsonValue) -> list[str]:
    found = value.get("bullets") if isinstance(value, dict) else None
    if not isinstance(found, list):
        raise TypeError("outline output has no bullets list")
    return [str(item) for item in found]


def pick_entry(entries: Sequence[str], rng: random.Random, bias: float) -> str:
    """Choose which listed file the agent reads next.

    ``bias`` is the exponent of a rank-biased (Zipf-like) weighting over the listing: 0 draws
    uniformly, larger values concentrate on the entries a relevance-ordered listing puts first.

    This single number is what the whole benchmark's predictability rests on, so it is a declared
    parameter rather than a constant. At ``bias = 0`` the argument of a read-after-list is
    unpredictable *by construction* -- no model of the call history can recover a coin flip -- and
    the benchmark would be measuring the workload's entropy rather than the speculator. At high
    bias it would be measuring a foregone conclusion. The default is 0, the least flattering
    setting, and ``bench/bench_speculation.py`` sweeps it so the dependence is visible instead of
    baked in.
    """
    if bias <= 0 or len(entries) == 1:
        return entries[rng.randrange(len(entries))]
    weights = [1.0 / ((rank + 1) ** bias) for rank in range(len(entries))]
    return rng.choices(list(entries), weights=weights, k=1)[0]


class _Builder:
    """Builds one task by running the corpus tools and recording what they returned."""

    def __init__(self, corpus: Corpus, rng: random.Random, read_bias: float = 0.0) -> None:
        self.corpus = corpus
        self.rng = rng
        self.read_bias = read_bias
        self.calls: list[ReferenceCall] = []

    def pick(self, entries: Sequence[str]) -> str:
        return pick_entry(entries, self.rng, self.read_bias)

    def _record(self, tool: str, arguments: dict[str, JsonValue], output: JsonValue) -> JsonValue:
        self.calls.append(ReferenceCall(tool=tool, arguments=arguments, output=output))
        return output

    def list_dir(self, module: str) -> JsonValue:
        path = f"src/{module}"
        return self._record("list_dir", {"path": path}, self.corpus.list_dir(path))

    def read_file(self, path: str) -> JsonValue:
        return self._record("read_file", {"path": path}, self.corpus.read_file(path))

    def search_code(self, query: str) -> JsonValue:
        return self._record("search_code", {"query": query}, self.corpus.search_code(query))

    def fetch_doc(self, url: str) -> JsonValue:
        return self._record("fetch_doc", {"url": url}, self.corpus.fetch_doc(url))

    def outline(self, text: str) -> JsonValue:
        from speculative_tool_executor.workload.corpus import outline_text  # noqa: PLC0415

        return self._record("outline_text", {"text": text}, outline_text(text))

    def tally(self, items: Sequence[str]) -> JsonValue:
        from speculative_tool_executor.workload.corpus import tally_items  # noqa: PLC0415

        return self._record("tally_items", {"items": list(items)}, tally_items(items))

    def save_note(self, key: str, text: str) -> None:
        """Writes have no reference output: nothing in these shapes follows one."""
        self._record("save_note", {"key": key, "text": text}, None)

    def bump_index(self, key: str) -> None:
        self._record("bump_index", {"key": key}, None)

    def get_note(self, key: str) -> None:
        self._record("get_note", {"key": key}, None)

    def send_report(self, recipient: str, subject: str, body: str) -> None:
        self._record(
            "send_report",
            {"recipient": recipient, "subject": subject, "body": body},
            None,
        )


def _build(  # noqa: PLR0911 - one branch per task shape reads better than a dispatch table
    shape: str, builder: _Builder, modules: Sequence[str]
) -> str:
    """Run one shape, returning the operator prompt that would have produced it."""
    rng = builder.rng
    module = rng.choice(list(modules))

    if shape == "audit_module":
        listing = builder.list_dir(module)
        entries = entries_of(listing)
        target = builder.pick(entries)
        read = builder.read_file(target)
        builder.save_note(text_of(read, "note_key"), text_of(read, "summary"))
        return f"Audit the {module} module: list its files, read {target}, and file a note."

    if shape == "search_outline":
        symbol = rng.choice(sorted(s for s in builder.corpus.symbols if s.startswith(module)))
        found = builder.search_code(symbol)
        paths = paths_of(found)
        read = builder.read_file(paths[0])
        builder.outline(text_of(read, "summary"))
        return f"Find {symbol} in the codebase, read where it lives, and outline it."

    if shape == "doc_note":
        document = builder.corpus.documents[module]
        fetched = builder.fetch_doc(document.url)
        builder.save_note(text_of(fetched, "note_key"), text_of(fetched, "summary"))
        return f"Read the {module} service guide and file a note about it."

    if shape == "index_bump":
        listing = builder.list_dir(module)
        entries = entries_of(listing)
        target = builder.pick(entries)
        read = builder.read_file(target)
        builder.bump_index(text_of(read, "index_key"))
        return f"Record a visit to the {module} module after reading {target}."

    if shape == "note_lookup":
        listing = builder.list_dir(module)
        entries = entries_of(listing)
        target = builder.pick(entries)
        read = builder.read_file(target)
        builder.get_note(text_of(read, "note_key"))
        return f"Check whether {target} in {module} already has a note."

    if shape == "tally_dir":
        listing = builder.list_dir(module)
        builder.tally(entries_of(listing))
        return f"How many distinct files does the {module} module have?"

    if shape == "doc_report":
        document = builder.corpus.documents[module]
        fetched = builder.fetch_doc(document.url)
        outline = builder.outline(text_of(fetched, "summary"))
        builder.send_report(
            rng.choice(list(RECIPIENTS)),
            f"{module} guide summary"[:200],
            "\n".join(bullets_of(outline))[:4000] or text_of(fetched, "summary"),
        )
        return f"Summarise the {module} guide and report it to the rota."

    raise ValueError(f"unknown task shape {shape!r}")


def _shape_sequence(count: int, rng: random.Random) -> list[str]:
    """A deterministic shape multiset matching ``SHAPE_WEIGHTS``, then shuffled.

    Sampling independently would let the realised mix drift from the declared one, which would
    make two runs of "the same" benchmark measure slightly different workloads.
    """
    total = sum(SHAPE_WEIGHTS.values())
    shapes: list[str] = []
    for shape, weight in SHAPE_WEIGHTS.items():
        shapes.extend([shape] * ((count * weight) // total))
    for shape in _round_robin(count - len(shapes)):
        shapes.append(shape)
    rng.shuffle(shapes)
    return shapes


def _round_robin(missing: int) -> list[str]:
    """Hand out the remainder of the integer division to the heaviest shapes, in order."""
    order = sorted(SHAPE_WEIGHTS, key=lambda name: (-SHAPE_WEIGHTS[name], name))
    return [order[index % len(order)] for index in range(max(0, missing))]


def generate_tasks(
    *,
    count: int = DEFAULT_TASK_COUNT,
    seed: int = 20260101,
    corpus: Corpus | None = None,
    prefix: str = "task",
    read_bias: float = 0.0,
) -> tuple[Task, ...]:
    """Generate ``count`` tasks with reference trajectories. Deterministic in ``seed``.

    ``read_bias`` controls how concentrated the choice of file to read is (see :func:`pick_entry`)
    and is therefore the knob that sets how predictable the workload is at all.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if read_bias < 0:
        raise ValueError("read_bias must be non-negative")
    workspace = corpus if corpus is not None else build_corpus()
    modules = sorted({path.split("/")[1] for path in workspace.files})
    # Workload generation, not cryptography.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    tasks: list[Task] = []
    for index, shape in enumerate(_shape_sequence(count, rng)):
        builder = _Builder(
            workspace,
            random.Random(f"{seed}:{index}"),  # noqa: S311  # nosec B311
            read_bias=read_bias,
        )
        prompt = _build(shape, builder, modules)
        tasks.append(
            Task(
                task_id=f"{prefix}-{index:04d}",
                shape=shape,
                prompt=prompt,
                calls=tuple(builder.calls),
                final_text=f"Completed {shape} in {len(builder.calls)} steps.",
            )
        )
    return tuple(tasks)


def training_trajectories(
    *,
    count: int = 120,
    seed: int = 777,
    corpus: Corpus | None = None,
    read_bias: float = 0.0,
) -> tuple[tuple[ObservedCall, ...], ...]:
    """Trajectories for training the predictor, from tasks disjoint from the evaluation set.

    The corpus is shared on purpose: the realistic deployment is an agent that has seen its own
    history over the *same* workspace. What the predictor must not see is the evaluation tasks,
    and a different seed guarantees that.
    """
    return tuple(
        task.trajectory()
        for task in generate_tasks(
            count=count, seed=seed, corpus=corpus, prefix="train", read_bias=read_bias
        )
    )


def shape_counts(tasks: Sequence[Task]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.shape] = counts.get(task.shape, 0) + 1
    return dict(sorted(counts.items()))
