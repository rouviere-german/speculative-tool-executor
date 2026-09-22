"""FastAPI application factory with the baseline hardening applied."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field

from speculative_tool_executor import __version__
from speculative_tool_executor.core.config import Settings, get_settings
from speculative_tool_executor.core.errors import install_error_handlers
from speculative_tool_executor.core.logging import get_logger
from speculative_tool_executor.core.metrics import create_http_metrics, render_metrics
from speculative_tool_executor.core.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from speculative_tool_executor.core.ratelimit import RateLimiter
from speculative_tool_executor.core.redaction import redact_text
from speculative_tool_executor.core.security import (
    ApiKeyVerifier,
    AuthenticatedPrincipal,
    require_api_key,
)
from speculative_tool_executor.core.tracing import build_tracer_provider
from speculative_tool_executor.spec.controller import SpeculationController, policy_from_settings

__all__ = ["ReadinessCheck", "RedactRequest", "RedactResponse", "create_app"]

ReadinessCheck = Callable[[], bool | Awaitable[bool]]
_log = get_logger("speculative_tool_executor.app")


class RedactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=10_000)


class RedactResponse(BaseModel):
    text: str


class SpeculationSnapshot(BaseModel):
    """The controller's ledgers, as served by ``/v1/speculation``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    tripped: bool
    attempts: float
    hits: float
    misses: float
    hit_ratio: float
    cancellations: float
    occ_conflicts: float
    stale_reads: float
    speculative_spend_usd: float
    wasted_usd: float
    baseline_spend_usd: float
    waste_ratio: float
    latency_saved_seconds: float
    per_tool_hit_ratio: dict[str, float]
    refusals: dict[str, int]


async def _run_check(name: str, check: ReadinessCheck) -> bool:
    try:
        result = check()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception as exc:  # noqa: BLE001 - a failing probe must report not-ready, not crash
        _log.warning("readiness_check_failed", check=name, error_type=type(exc).__name__)
        return False


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
    controller: SpeculationController | None = None,
) -> FastAPI:
    """Build the app. All state hangs off ``app.state``; nothing global is mutated."""
    cfg = settings if settings is not None else get_settings()
    checks: dict[str, ReadinessCheck] = dict(readiness_checks or {})
    provider = tracer_provider or build_tracer_provider(cfg.service_name, cfg.tracing_exporter)
    metrics = create_http_metrics(registry)
    verifier = ApiKeyVerifier(cfg.api_key_sha256)
    limiter = RateLimiter(
        cfg.rate_limit_per_second, cfg.rate_limit_burst, max_keys=cfg.rate_limit_max_clients
    )

    app = FastAPI(
        title=cfg.service_name,
        version=__version__,
        docs_url="/docs" if cfg.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.enable_docs else None,
    )
    app.state.settings = cfg
    app.state.metrics = metrics
    app.state.api_key_verifier = verifier
    app.state.rate_limiter = limiter
    app.state.tracer_provider = provider
    # One controller per process: the snapshot endpoint and the agent loop must not disagree
    # about whether speculation is on.
    app.state.speculation = controller or SpeculationController(policy_from_settings(cfg))
    install_error_handlers(app)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is serving requests. Checks no dependencies."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Readiness: every registered dependency check passes. Reports names, not errors."""
        results = {name: await _run_check(name, check) for name, check in checks.items()}
        ready = all(results.values())
        body: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "checks": {name: "ok" if ok else "fail" for name, ok in results.items()},
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    metrics_deps = [Depends(require_api_key)] if cfg.metrics_require_auth else []

    @app.get("/metrics", include_in_schema=False, dependencies=metrics_deps)
    async def metrics_endpoint(request: Request) -> Response:
        payload, content_type = render_metrics(request.app.state.metrics.registry)
        return Response(content=payload, media_type=content_type)

    @app.get("/v1/speculation", response_model=SpeculationSnapshot)
    async def speculation(
        request: Request, principal: AuthenticatedPrincipal
    ) -> SpeculationSnapshot:
        """Whether speculation is paying for itself, and whether it is still on.

        Operators need both numbers in one place during an incident: a hit ratio without the
        waste ratio says nothing about cost, and neither says anything if the waste trip has
        already turned speculation off.
        """
        state: SpeculationController = request.app.state.speculation
        _log.info("speculation_snapshot", key_id=principal.key_id)
        return SpeculationSnapshot.model_validate(state.snapshot())

    @app.post("/v1/redact", response_model=RedactResponse)
    async def redact(body: RedactRequest, principal: AuthenticatedPrincipal) -> RedactResponse:
        """Example protected endpoint: return the input with PII and secrets redacted."""
        _log.info("redact_called", key_id=principal.key_id, chars=len(body.text))
        return RedactResponse(text=redact_text(body.text))

    # add_middleware prepends, so the last one added is the outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=cfg.max_body_bytes, metrics=metrics)
    app.add_middleware(RateLimitMiddleware, limiter=limiter, verifier=verifier, metrics=metrics)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(
        RequestContextMiddleware,
        tracer=provider.get_tracer("speculative_tool_executor.http"),
        metrics=metrics,
        docs_enabled=cfg.enable_docs,
    )
    return app
