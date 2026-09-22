"""The synthetic workspace the benchmark agent works over: a small codebase and a docs site.

Everything is generated from a seed and held in memory. That is deliberate: the benchmark has to
be reproducible on any machine with no fixtures to download, and the point of the workload is the
*shape* of the call sequences (list then read, search then read then note), not the realism of
the code being read.

Two properties matter for the rest of the system:

- **Outputs carry the arguments of the calls that usually follow them.** A directory listing
  returns full relative paths, a search returns paths, a file read returns the key a note about
  it would use. That is what lets the predictor derive a concrete next call instead of guessing
  one, and it is how a well-designed real tool behaves too.
- **Reads are jailed.** Paths are resolved against the workspace root and anything that escapes
  it is refused, after canonicalization rather than before, so ``a/../../etc`` is rejected rather
  than quietly rewritten.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import JsonValue

from speculative_tool_executor.core.canonical import sha256_hex

__all__ = [
    "DOCS_HOST",
    "Corpus",
    "WorkspaceError",
    "build_corpus",
    "note_key_for",
    "outline_text",
    "tally_items",
]

MODULES = (
    "auth",
    "billing",
    "ingest",
    "render",
    "scheduler",
    "search",
    "storage",
    "telemetry",
)
STEMS = ("api", "config", "handlers", "models", "service", "utils")
DOCS_HOST = "docs.internal.test"
DOCS_BASE = f"https://{DOCS_HOST}/guide"
_MAX_PATH = 200


class WorkspaceError(ValueError):
    """A path or URL left the workspace or the bundled documentation service."""


def _as_json_list(values: Sequence[str]) -> list[JsonValue]:
    """``JsonValue``'s list arm is invariant, so a ``list[str]`` is not a ``list[JsonValue]``."""
    return list(values)


def _slug(text: str) -> str:
    return sha256_hex(text)[:12]


def note_key_for(path: str) -> str:
    """The note key a summary of ``path`` is stored under. Tools return it so agents need not
    build one from parts, which also keeps speculative arguments derivable."""
    return f"notes/{path.replace('/', '.')}"


def outline_text(text: str) -> dict[str, JsonValue]:
    """Pure text outline: deterministic, no I/O, safe to speculate without limit."""
    sentences = [part.strip() for part in text.replace(";", ".").split(".") if part.strip()]
    bullets: list[JsonValue] = [f"- {sentence}" for sentence in sentences[:5]]
    words = len(text.split())
    return {
        "slug": f"outline/{_slug(text)}",
        "bullets": bullets,
        "words": words,
        "summary": sentences[0] if sentences else text.strip(),
    }


def tally_items(items: Sequence[str]) -> dict[str, JsonValue]:
    """Pure frequency count: how many members, how many distinct, and the three commonest.

    Order is insignificant (the declared canonicalization sorts), repetition is not.
    """
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return {
        "distinct": len(counts),
        "total": len(items),
        "top": _as_json_list(sorted(counts, key=lambda key: (-counts[key], key))[:3]),
    }


@dataclass(frozen=True, slots=True)
class Document:
    slug: str
    title: str
    sections: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"{DOCS_BASE}/{self.slug}"

    @property
    def summary(self) -> str:
        return f"{self.title}. {self.sections[0]}" if self.sections else self.title


@dataclass(frozen=True, slots=True)
class Corpus:
    """An in-memory workspace plus the bundled documentation service."""

    files: Mapping[str, tuple[str, ...]]
    documents: Mapping[str, Document]
    symbols: Mapping[str, str]

    # -- read tools ---------------------------------------------------------------------------

    @staticmethod
    def _check_path(path: str) -> str:
        if not path or len(path) > _MAX_PATH:
            raise WorkspaceError("path must be between 1 and 200 characters")
        if path.startswith("/") or path.startswith(".."):
            raise WorkspaceError("path escapes the workspace")
        if any(segment == ".." for segment in path.split("/")):
            raise WorkspaceError("path escapes the workspace")
        return path

    def list_dir(self, path: str) -> dict[str, JsonValue]:
        directory = self._check_path(path).rstrip("/")
        prefix = f"{directory}/"
        entries = _as_json_list(sorted(name for name in self.files if name.startswith(prefix)))
        if not entries:
            raise WorkspaceError("directory not found")
        return {"path": directory, "entries": entries, "count": len(entries)}

    def read_file(self, path: str) -> dict[str, JsonValue]:
        name = self._check_path(path)
        lines = self.files.get(name)
        if lines is None:
            raise WorkspaceError("file not found")
        defined = _as_json_list(
            [line.split("(")[0][4:] for line in lines if line.startswith("def ")]
        )
        module = name.split("/")[1] if name.count("/") >= 2 else name.split("/")[0]
        return {
            "path": name,
            "module": module,
            "lines": list(lines),
            "symbols": defined,
            "note_key": note_key_for(name),
            "index_key": f"index/{module}",
            "summary": (
                f"{name} has {len(lines)} lines and defines "
                f"{defined[0] if defined else 'no symbol'}"
            ),
        }

    def search_code(self, query: str) -> dict[str, JsonValue]:
        needle = query.strip().lower()
        if not needle:
            raise WorkspaceError("query must not be empty")
        matches: list[JsonValue] = []
        paths: list[JsonValue] = []
        for name in sorted(self.files):
            for number, line in enumerate(self.files[name], start=1):
                if needle in line.lower():
                    matches.append({"path": name, "line": number, "text": line})
                    if name not in paths:
                        paths.append(name)
                    break
        return {"query": needle, "paths": paths, "matches": matches, "count": len(matches)}

    def fetch_doc(self, url: str) -> dict[str, JsonValue]:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != DOCS_HOST:
            raise WorkspaceError("only the bundled documentation service may be fetched")
        slug = parts.path.rsplit("/", 1)[-1]
        document = self.documents.get(slug)
        if document is None:
            raise WorkspaceError("document not found")
        return {
            "url": document.url,
            "slug": document.slug,
            "title": document.title,
            "sections": _as_json_list(document.sections),
            "note_key": f"notes/doc.{document.slug}",
            "summary": document.summary,
        }


def build_corpus(*, seed: int = 20260101, modules: Sequence[str] = MODULES) -> Corpus:
    """Build a deterministic workspace: a few files per module and one doc page per module."""
    # Fixture generation, not cryptography.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    files: dict[str, tuple[str, ...]] = {}
    symbols: dict[str, str] = {}
    documents: dict[str, Document] = {}
    for module in modules:
        stems = sorted(rng.sample(STEMS, k=rng.randint(3, 4)))
        for stem in stems:
            path = f"src/{module}/{stem}.py"
            symbol = f"{module}_{stem}"
            body = [
                f"# module {module}, unit {stem}",
                "from runtime import dispatch",
                "",
                f"def {symbol}(request):",
                f'    """Handle a {module} {stem} request."""',
                "    return dispatch(request)",
            ]
            for extra in range(rng.randint(0, 3)):
                body.append(f"# note {extra}: {module} {stem} invariant {extra}")
            files[path] = tuple(body)
            symbols[symbol] = path
        documents[module] = Document(
            slug=module,
            title=f"{module.capitalize()} service guide",
            sections=(
                f"The {module} service owns {len(stems)} units",
                f"Operational limits for {module} are reviewed quarterly",
                f"Escalation for {module} goes to the platform rota",
            ),
        )
    return Corpus(files=files, documents=documents, symbols=symbols)
