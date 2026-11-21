"""Session E, part 2 — FastAPI routing layer (PRD Sections 6.7 and 8).

    ROBOSCRIPT_BACKEND=rules uvicorn roboscript.serve.api:app --port 8000
    ROBOSCRIPT_BACKEND=local uvicorn roboscript.serve.api:app --port 8000
    ROBOSCRIPT_BACKEND=vllm  VLLM_URL=http://localhost:8080/v1 uvicorn roboscript.serve.api:app

Request path for every endpoint:

  telemetry --Pydantic--> Scenario
     |  (emergency only) life-safety fast path: person/forklift or e-stop -> deterministic
     |   halt in ~1 ms, the LLM is never on the critical path for these two categories
     v
  backend.complete(messages, adapter) under a per-endpoint deadline
     |  timeout / transport error / unparsable / schema-invalid -> EMERGENCY_HALT_FALLBACK
     v
  safety.guard(scenario, response)  -- interlock: only ever makes the answer MORE conservative
     v
  RoboResponse + adapter_used, latency_ms, fallback_reason, guard_overrides, trace_url

Endpoints: POST /v1/emergency (dpo), /v1/navigate (sft), /v1/diagnose (sft),
POST /v1/act (auto-routes by the classified fault), GET /v1/health.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

from roboscript.constants import SYSTEM_PROMPT
from roboscript.safety import MANDATORY_HALT, classify, guard, halt_response, reference_action
from roboscript.schemas.payload import EMERGENCY_HALT_FALLBACK, RoboResponse
from roboscript.schemas.telemetry import Scenario
from roboscript.serve.backends import Backend, make_backend

log = logging.getLogger("roboscript.api")

DEADLINES_MS = {
    "emergency": float(os.getenv("ROBOSCRIPT_DEADLINE_MS_EMERGENCY", "300")),
    "navigate": float(os.getenv("ROBOSCRIPT_DEADLINE_MS_NAVIGATE", "500")),
    "diagnose": float(os.getenv("ROBOSCRIPT_DEADLINE_MS_DIAGNOSE", "500")),
}
ADAPTERS = {"emergency": "roboscript-dpo", "navigate": "roboscript-sft", "diagnose": "roboscript-sft"}
FAST_PATH = os.getenv("ROBOSCRIPT_INTERLOCK_FAST_PATH", "1") == "1"


class TelemetryRequest(Scenario):
    session_id: str | None = None


class ActResponse(RoboResponse):
    adapter_used: str
    latency_ms: float
    fallback_reason: str | None = None
    guard_overrides: list[str] = []
    fault_category: int | None = None
    trace_url: str | None = None


class _Stats:
    def __init__(self):
        self.requests = 0
        self.fallbacks = Counter()
        self.overrides = 0
        self.fast_path = 0


def _tracer():
    """LangSmith tracing when LANGCHAIN_API_KEY is set and langsmith is installed."""
    if not os.getenv("LANGCHAIN_API_KEY"):
        return None
    try:
        from langsmith import traceable
        return traceable
    except ImportError:
        return None


def create_app(backend: Backend | None = None) -> FastAPI:
    app = FastAPI(title="RoboScript AI", version="1.0")
    app.state.backend = backend or make_backend()
    app.state.stats = _Stats()
    traceable = _tracer()

    async def call_model(scenario: dict, adapter: str, deadline_ms: float) -> tuple[RoboResponse, str | None]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(scenario)}]
        fn = app.state.backend.complete
        if traceable is not None:
            cid = classify(scenario)
            fn = traceable(run_type="llm", name="roboscript_inference",
                           tags=["roboscript", adapter, os.getenv("ENVIRONMENT", "dev")],
                           metadata={"adapter": adapter, "fault_category": cid})(fn)
        try:
            raw = await asyncio.wait_for(fn(messages, adapter), timeout=deadline_ms / 1000)
        except asyncio.TimeoutError:
            return EMERGENCY_HALT_FALLBACK, "deadline_exceeded"
        except Exception as e:  # noqa: BLE001 - transport errors must never propagate as motion
            log.exception("backend error")
            return EMERGENCY_HALT_FALLBACK, f"backend_error:{type(e).__name__}"
        try:
            return RoboResponse.model_validate_json(raw), None
        except ValidationError:
            log.warning("validation failure; raw=%r", raw[:500])
            return EMERGENCY_HALT_FALLBACK, "validation_failed"

    async def handle(req: TelemetryRequest, route: str) -> ActResponse:
        t0 = time.perf_counter()
        st = app.state.stats
        st.requests += 1
        scenario = req.model_dump(exclude={"session_id"})
        cid = classify(scenario)
        adapter = ADAPTERS[route]

        if route == "emergency" and FAST_PATH and cid in MANDATORY_HALT:
            st.fast_path += 1
            resp, fallback = halt_response(
                f"Interlock fast path: category {cid} mandates halt. Zero velocity commanded."), None
            adapter = "interlock"
        else:
            resp, fallback = await call_model(scenario, adapter, DEADLINES_MS[route])
        if fallback:
            st.fallbacks[fallback] += 1
        resp, overrides = guard(scenario, resp)
        st.overrides += bool(overrides)
        return ActResponse(**resp.model_dump(), adapter_used=adapter,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                           fallback_reason=fallback, guard_overrides=overrides,
                           fault_category=cid)

    @app.post("/v1/emergency", response_model=ActResponse)
    async def emergency(req: TelemetryRequest) -> ActResponse:
        return await handle(req, "emergency")

    @app.post("/v1/navigate", response_model=ActResponse)
    async def navigate(req: TelemetryRequest) -> ActResponse:
        return await handle(req, "navigate")

    @app.post("/v1/diagnose", response_model=ActResponse)
    async def diagnose(req: TelemetryRequest) -> ActResponse:
        return await handle(req, "diagnose")

    @app.post("/v1/act", response_model=ActResponse)
    async def act(req: TelemetryRequest) -> ActResponse:
        """Auto-route: stop-required faults -> emergency (DPO), Nav2 faults -> navigate."""
        scenario = req.model_dump(exclude={"session_id"})
        ref = reference_action(scenario)
        if ref is None or ref.must_stop:
            route = "emergency"
        elif ref.payload_type == "NAV2_ACTION":
            route = "navigate"
        else:
            route = "diagnose"
        return await handle(req, route)

    @app.get("/v1/health")
    async def health() -> dict:
        st = app.state.stats
        n = max(st.requests, 1)
        return {"status": "healthy", "backend": app.state.backend.name,
                "adapters": sorted(set(ADAPTERS.values())),
                "model": os.getenv("BASE_MODEL", "local"),
                "requests": st.requests,
                "validation_fallback_rate_pct": round(100 * sum(st.fallbacks.values()) / n, 2),
                "fallbacks": dict(st.fallbacks), "guard_override_count": st.overrides,
                "interlock_fast_path_count": st.fast_path,
                "deadlines_ms": DEADLINES_MS}

    return app


app = create_app()
