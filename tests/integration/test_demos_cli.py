from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from speculative_tool_executor import cli
from speculative_tool_executor.agentic.audit import AuditLog
from speculative_tool_executor.core import tracing
from speculative_tool_executor.core.security import hash_api_key
from speculative_tool_executor.sim.demo import run_speculation_demo, span_summary

pytestmark = pytest.mark.integration


def test_demo_is_offline_equivalent_and_contains_the_injection(tmp_path: Path) -> None:
    summary = run_speculation_demo(tmp_path)

    # The latency claim: speculation removed tool wait without changing what the run did.
    assert summary["equivalent"] is True
    assert float(summary["tool_wait_reduction"]) > 0.15  # type: ignore[arg-type]
    assert 0.0 < float(summary["coverage"]) <= 1.0  # type: ignore[arg-type]

    # The safety claims, each one a live check rather than a comment.
    assert summary["external_speculations"] == 0
    assert summary["send_report_speculations"] == 0
    assert summary["messages_sent"] == 0
    assert summary["approval_requests"] == ["send_report"]
    assert summary["model_obeyed_injection"] is True
    assert summary["audit_ok"] is True
    assert summary["tampered_copy_detected"] is True
    assert summary["all_checks_passed"] is True

    # Spans: authoritative and speculative work are told apart in the trace itself.
    spans = summary["spans"]
    assert isinstance(spans, dict)
    assert spans["consistent"] is True
    assert isinstance(spans["speculative_spans"], int)
    assert spans["speculative_spans"] > 0
    assert isinstance(spans["calls_served_from_speculation"], int)
    assert spans["calls_served_from_speculation"] >= 1


def _spans(*spans: tuple[str, dict[str, Any]]) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = tracing.build_tracer_provider("span-check", extra_exporter=exporter)
    tracer = provider.get_tracer("t")
    for name, attributes in spans:
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
    return exporter


def test_the_demo_span_check_passes_only_on_a_faithful_trace() -> None:
    """The live check has to be able to fail, or it is decoration."""
    guess = ("tool.speculate", {"speculative": True, "call.hash": "h1"})
    served = (
        "tool.execute",
        {"speculative": False, "speculation.mode": "speculative-committed", "call.hash": "h1"},
    )
    direct = ("tool.execute", {"speculative": False, "speculation.mode": "direct"})

    good = span_summary(_spans(guess, served, direct), min_authoritative=2)
    assert good == {
        "authoritative_spans": 2,
        "speculative_spans": 1,
        "calls_served_from_speculation": 1,
        "consistent": True,
    }

    unexplained = (
        "tool.execute",
        {"speculative": False, "speculation.mode": "speculative-committed", "call.hash": "h2"},
    )
    assert span_summary(_spans(guess, unexplained), min_authoritative=1)["consistent"] is False
    mislabelled = ("tool.speculate", {"speculative": False, "call.hash": "h1"})
    assert span_summary(_spans(mislabelled, served), min_authoritative=1)["consistent"] is False
    assert span_summary(_spans(guess, served), min_authoritative=2)["consistent"] is False
    assert span_summary(_spans(direct), min_authoritative=1)["consistent"] is False


def test_demo_is_rerunnable_from_a_dirty_workdir(tmp_path: Path) -> None:
    first = run_speculation_demo(tmp_path)
    second = run_speculation_demo(tmp_path)
    assert second["audit_entries"] == first["audit_entries"]
    assert second["p50_reduction"] == first["p50_reduction"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_one_json_summary_per_kit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    decoder = json.JSONDecoder()
    demos: list[str] = []
    index = 0
    while index < len(output):
        if output[index] != "{":
            index += 1
            continue
        obj, index = decoder.raw_decode(output, index)
        demos.append(obj["demo"])
    assert demos == ["speculation"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_skips_removed_kits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert cli.main(["demo", "--kit", "speculation", "--workdir", str(tmp_path)]) == 1


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("agent", "step", {})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["speculative_tool_executor", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("speculative_tool_executor", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
