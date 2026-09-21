"""Prometheus metrics for speculation, on a per-instance registry (tests stay isolated).

The metric set is chosen so that the two questions an operator actually asks can be answered
without a trace: *is speculation paying for itself* (hit ratio against wasted spend and saved
seconds) and *is it safe* (conflicts, cancellations, and refusals broken down by reason).
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["SpeculationMetrics", "create_speculation_metrics"]


@dataclass(frozen=True, slots=True)
class SpeculationMetrics:
    registry: CollectorRegistry
    attempts_total: Counter
    hits_total: Counter
    hit_ratio: Gauge
    latency_saved_seconds: Counter
    wasted_usd_total: Counter
    speculative_spend_usd_total: Counter
    cancellations_total: Counter
    occ_conflicts_total: Counter
    refusals_total: Counter
    calls_total: Counter
    call_latency_seconds: Histogram
    enabled: Gauge

    def observe_call(self, tool: str, mode: str, seconds: float) -> None:
        self.calls_total.labels(tool=tool, mode=mode).inc()
        self.call_latency_seconds.labels(tool=tool, mode=mode).observe(max(0.0, seconds))


def create_speculation_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "app"
) -> SpeculationMetrics:
    reg = registry if registry is not None else CollectorRegistry()
    return SpeculationMetrics(
        registry=reg,
        attempts_total=Counter(
            "speculation_attempts_total",
            "Speculative tool executions started, by tool.",
            ["tool"],
            namespace=namespace,
            registry=reg,
        ),
        hits_total=Counter(
            "speculation_hits_total",
            "Speculative results claimed and committed by the authoritative call, by tool.",
            ["tool"],
            namespace=namespace,
            registry=reg,
        ),
        hit_ratio=Gauge(
            "speculation_hit_ratio",
            "Committed speculations divided by started ones, by tool.",
            ["tool"],
            namespace=namespace,
            registry=reg,
        ),
        latency_saved_seconds=Counter(
            "speculation_latency_saved_seconds",
            "Tool latency the authoritative path did not have to wait for.",
            namespace=namespace,
            registry=reg,
        ),
        wasted_usd_total=Counter(
            "speculation_wasted_usd",
            "Cost of speculative work that was cancelled, expired or discarded.",
            namespace=namespace,
            registry=reg,
        ),
        speculative_spend_usd_total=Counter(
            "speculation_spend_usd",
            "Cost of all speculative work started, whether or not it was used.",
            namespace=namespace,
            registry=reg,
        ),
        cancellations_total=Counter(
            "speculation_cancellations_total",
            "Speculative executions cancelled because the authoritative call did not match.",
            namespace=namespace,
            registry=reg,
        ),
        occ_conflicts_total=Counter(
            "speculation_occ_conflicts_total",
            "Speculative branches discarded because a version they read had changed.",
            namespace=namespace,
            registry=reg,
        ),
        refusals_total=Counter(
            "speculation_refusals_total",
            "Predictions the controller or the safety checks refused, by reason.",
            ["reason"],
            namespace=namespace,
            registry=reg,
        ),
        calls_total=Counter(
            "tool_calls_total",
            "Authoritative tool calls, by tool and how they were served.",
            ["tool", "mode"],
            namespace=namespace,
            registry=reg,
        ),
        call_latency_seconds=Histogram(
            "tool_call_latency_seconds",
            "Wait the authoritative call experienced, by tool and how it was served.",
            ["tool", "mode"],
            namespace=namespace,
            registry=reg,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        ),
        enabled=Gauge(
            "speculation_enabled",
            "1 when speculation is active, 0 when the kill switch or the waste trip is in force.",
            namespace=namespace,
            registry=reg,
        ),
    )
