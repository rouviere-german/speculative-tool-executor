"""Speculation as an operational control: turned off, narrowed, and observable.

The spec calls for a kill switch and per-tool overrides that work from configuration, plus a way
to see what speculation is doing. These are the tests that keep those promises honest: they drive
the settings the way an operator would (environment variables) rather than constructing a policy
object directly, because the environment is the part that has to work during an incident.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from speculative_tool_executor.core.app import create_app
from speculative_tool_executor.core.config import Settings
from speculative_tool_executor.core.security import hash_api_key
from speculative_tool_executor.sim.experiment import Scenario, build_arm
from speculative_tool_executor.spec.controller import (
    SpeculationController,
    policy_from_settings,
)
from speculative_tool_executor.workload.corpus import build_corpus
from speculative_tool_executor.workload.tasks import generate_tasks

pytestmark = pytest.mark.integration

CORPUS = build_corpus()
API_KEY = "test-key-" + "z" * 24
AUTH = {"X-API-Key": API_KEY}


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, api_key_sha256=[hash_api_key(API_KEY)], **overrides)  # type: ignore[arg-type]


def client(**overrides: object) -> TestClient:
    return TestClient(create_app(settings(**overrides)))


# -- configuration to policy ---------------------------------------------------------------------


def test_the_env_kill_switch_reaches_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SPECULATION_ENABLED", "false")
    policy = policy_from_settings(Settings(_env_file=None))
    assert policy.enabled is False
    assert SpeculationController(policy).enabled is False


def test_per_tool_overrides_come_from_a_comma_separated_env_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_SPECULATION_DISABLED_TOOLS", "save_note, bump_index")
    policy = policy_from_settings(Settings(_env_file=None))
    assert policy.disabled_tools == frozenset({"save_note", "bump_index"})


def test_a_malformed_tool_name_is_refused_rather_than_silently_ignored() -> None:
    with pytest.raises(ValueError, match="must be tool names"):
        Settings(_env_file=None, speculation_disabled_tools=["not a tool name!"])


def test_the_rest_of_the_policy_is_configurable_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SPECULATION_TOP_K", "5")
    monkeypatch.setenv("APP_SPECULATION_MAX_IN_FLIGHT", "9")
    monkeypatch.setenv("APP_SPECULATION_BUDGET_USD", "0.25")
    monkeypatch.setenv("APP_SPECULATION_MAX_WASTE_RATIO", "0.33")
    monkeypatch.setenv("APP_SPECULATION_MIN_PROBABILITY", "0.4")
    monkeypatch.setenv("APP_SPECULATION_LATENCY_VALUE_USD_PER_SECOND", "0.05")
    policy = policy_from_settings(Settings(_env_file=None))
    assert policy.top_k == 5
    assert policy.max_in_flight == 9
    assert policy.speculative_budget_usd == 0.25
    assert policy.max_waste_ratio == 0.33
    assert policy.min_probability == 0.4
    assert policy.latency_value_usd_per_second == 0.05


async def test_the_configured_kill_switch_stops_real_speculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: an environment variable, a real agent run, and nothing speculated."""
    monkeypatch.setenv("APP_SPECULATION_ENABLED", "false")
    arm = build_arm(Scenario(name="env-off", read_bias=1.0), corpus=CORPUS, training_count=40)
    try:
        arm.controller.policy = policy_from_settings(Settings(_env_file=None))
        for task in generate_tasks(count=10, corpus=CORPUS, read_bias=1.0):
            await arm.agent.run(task)
        assert arm.controller.stats.attempts == 0
        assert arm.controller.stats.wasted_usd == 0.0
    finally:
        arm.close()


async def test_a_configured_per_tool_override_stops_only_that_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_SPECULATION_DISABLED_TOOLS", "save_note")
    arm = build_arm(Scenario(name="env-scoped", read_bias=1.0), corpus=CORPUS, training_count=80)
    try:
        arm.controller.policy = policy_from_settings(Settings(_env_file=None))
        for task in generate_tasks(count=20, corpus=CORPUS, read_bias=1.0):
            await arm.agent.run(task)
        assert arm.controller.tool_stats("save_note").attempts == 0
        assert arm.controller.tool_stats("read_file").attempts > 0
    finally:
        arm.close()


# -- the snapshot endpoint -----------------------------------------------------------------------


def test_the_snapshot_endpoint_requires_authentication() -> None:
    with client() as http:
        assert http.get("/v1/speculation").status_code == 401
        assert http.get("/v1/speculation", headers={"X-API-Key": "wrong"}).status_code == 401


def test_the_snapshot_reports_both_benefit_and_cost() -> None:
    """A hit ratio without a waste ratio would be marketing, so both must be served."""
    with client() as http:
        body = http.get("/v1/speculation", headers=AUTH).json()
    for field in (
        "enabled",
        "tripped",
        "hit_ratio",
        "waste_ratio",
        "wasted_usd",
        "latency_saved_seconds",
        "occ_conflicts",
        "cancellations",
        "per_tool_hit_ratio",
        "refusals",
    ):
        assert field in body
    assert body["enabled"] is True


def test_the_snapshot_shows_the_kill_switch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SPECULATION_ENABLED", "false")
    with TestClient(create_app(settings(speculation_enabled=False))) as http:
        body = http.get("/v1/speculation", headers=AUTH).json()
    assert body["enabled"] is False


def test_the_snapshot_reflects_a_live_controller() -> None:
    controller = SpeculationController()
    controller.record_baseline_spend(1.0)
    controller.record_launch("read_file", 0.002)
    controller.record_hit("read_file", saved_seconds=0.75)
    app = create_app(settings(), controller=controller)
    with TestClient(app) as http:
        body = http.get("/v1/speculation", headers=AUTH).json()
    assert body["hits"] == 1.0
    assert body["hit_ratio"] == 1.0
    assert body["latency_saved_seconds"] == pytest.approx(0.75)
    assert body["per_tool_hit_ratio"] == {"read_file": 1.0}
