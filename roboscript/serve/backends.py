"""Model backends for the serving layer.

ROBOSCRIPT_BACKEND selects one:

* ``vllm``  — any OpenAI-compatible server (vLLM multi-LoRA, SageMaker proxy, ...).
              The adapter name is passed as the ``model`` field.
* ``local`` — in-process transformers models (CPU or GPU). Adapter names map to
              model directories via ROBOSCRIPT_SFT_PATH / ROBOSCRIPT_DPO_PATH.
* ``rules`` — the deterministic reference policy. No model at all; used for demos,
              CI and ROS 2 integration tests, and as a ground-truth stand-in.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Protocol

from roboscript.data.procedural import render_response
from roboscript.eval.generate import generate, load_for_inference


class Backend(Protocol):
    name: str

    async def complete(self, messages: list[dict], adapter: str) -> str: ...


class RulesBackend:
    name = "rules"

    async def complete(self, messages: list[dict], adapter: str) -> str:
        import random
        scenario = json.loads(messages[-1]["content"])
        try:
            return json.dumps(render_response(scenario, random.Random(0)))
        except AssertionError:  # nominal scenario: nothing to do
            return json.dumps({
                "decision": "CONTINUE", "reasoning": "No active fault; continuing mission.",
                "payload_type": "FAULT_REPORT_ONLY", "ros2": None,
                "aws_telemetry": {"incident_severity": "LEVEL_0", "request_human_audit": False}})


class LocalHFBackend:
    name = "local"

    def __init__(self, paths: dict[str, str] | None = None):
        self.paths = paths or {
            "roboscript-sft": os.getenv("ROBOSCRIPT_SFT_PATH", "outputs/roboscript-sft"),
            "roboscript-dpo": os.getenv("ROBOSCRIPT_DPO_PATH", "outputs/roboscript-dpo-merged"),
            "roboscript-edge": os.getenv("ROBOSCRIPT_EDGE_PATH", "outputs/roboscript-edge"),
        }
        self._models: dict[str, tuple] = {}
        self._lock = asyncio.Lock()

    def _get(self, adapter: str):
        if adapter not in self._models:
            self._models[adapter] = load_for_inference(self.paths[adapter])
        return self._models[adapter]

    async def complete(self, messages: list[dict], adapter: str) -> str:
        scenario = json.loads(messages[-1]["content"])
        async with self._lock:  # one generate at a time on CPU
            model, tok = await asyncio.to_thread(self._get, adapter)
            (text, _), = await asyncio.to_thread(generate, model, tok, [scenario], 1)
        return text


class OpenAICompatBackend:
    name = "vllm"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("VLLM_URL", "http://localhost:8080/v1"),
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"))

    async def complete(self, messages: list[dict], adapter: str) -> str:
        r = await self.client.chat.completions.create(
            model=adapter, messages=messages, temperature=0.0, max_tokens=320,
            response_format={"type": "json_object"})
        return r.choices[0].message.content or ""


def make_backend(kind: str | None = None) -> Backend:
    kind = (kind or os.getenv("ROBOSCRIPT_BACKEND", "rules")).lower()
    if kind == "rules":
        return RulesBackend()
    if kind == "local":
        return LocalHFBackend()
    if kind == "vllm":
        return OpenAICompatBackend()
    raise ValueError(f"unknown backend {kind!r}")
