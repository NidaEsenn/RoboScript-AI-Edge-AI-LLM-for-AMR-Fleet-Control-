import asyncio
import json

import pytest
import torch
from fastapi.testclient import TestClient

from roboscript.distillation.distill import distillation_loss
from roboscript.eval.evaluator import score_set
from roboscript.eval.generate import parse_json
from roboscript.serve.api import create_app
from roboscript.serve.backends import RulesBackend
from roboscript.serve.vllm_server import build_command
from tests.conftest import BATTERY_CRIT, BATTERY_LOW, ESTOP, PERSON


class FixedBackend:
    name = "fixed"

    def __init__(self, text: str, delay: float = 0.0):
        self.text, self.delay, self.calls = text, delay, 0

    async def complete(self, messages, adapter):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.text


UNSAFE = json.dumps({"decision": "CONTINUE", "reasoning": "Keep going, fault is transient.",
                     "payload_type": "CMD_VEL_DIRECT",
                     "ros2": {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist",
                              "cmd_vel": {"linear": {"x": 0.6, "y": 0, "z": 0},
                                          "angular": {"x": 0, "y": 0, "z": 0}}},
                     "aws_telemetry": {"incident_severity": "LEVEL_0"}})


def test_endpoints_return_expected_payload_types():
    c = TestClient(create_app(RulesBackend()))
    assert c.post("/v1/emergency", json=ESTOP).json()["payload_type"] == "CMD_VEL_DIRECT"
    assert c.post("/v1/navigate", json=BATTERY_CRIT).json()["payload_type"] == "NAV2_ACTION"
    assert c.post("/v1/diagnose", json=BATTERY_LOW).json()["payload_type"] == "FAULT_REPORT_ONLY"
    h = c.get("/v1/health").json()
    assert h["status"] == "healthy" and h["requests"] == 3


def test_invalid_json_falls_back_to_halt():
    c = TestClient(create_app(FixedBackend("{this is not json")))
    r = c.post("/v1/navigate", json=BATTERY_CRIT).json()
    assert r["decision"] == "EMERGENCY_HALT" and r["fallback_reason"] == "validation_failed"
    assert r["ros2"]["cmd_vel"]["linear"]["x"] == 0.0
    assert c.get("/v1/health").json()["validation_fallback_rate_pct"] == 100.0


def test_deadline_exceeded_falls_back_to_halt(monkeypatch):
    import roboscript.serve.api as api
    monkeypatch.setitem(api.DEADLINES_MS, "navigate", 20)
    c = TestClient(create_app(FixedBackend("{}", delay=0.5)))
    r = c.post("/v1/navigate", json=BATTERY_CRIT).json()
    assert r["fallback_reason"] == "deadline_exceeded" and r["decision"] == "EMERGENCY_HALT"


def test_guard_overrides_unsafe_model_output():
    c = TestClient(create_app(FixedBackend(UNSAFE)))
    r = c.post("/v1/act", json=ESTOP | {"session_id": "amr-047"}).json()
    assert r["decision"] == "EMERGENCY_HALT"
    assert r["ros2"]["cmd_vel"]["linear"]["x"] == 0.0


def test_unsafe_motion_blocked_even_when_not_stop_category():
    c = TestClient(create_app(FixedBackend(UNSAFE)))
    r = c.post("/v1/diagnose", json=BATTERY_LOW).json()
    assert r["decision"] == "EMERGENCY_HALT" and r["guard_overrides"]


def test_life_safety_fast_path_skips_model():
    b = FixedBackend(UNSAFE)
    c = TestClient(create_app(b))
    r = c.post("/v1/emergency", json=PERSON).json()
    assert r["adapter_used"] == "interlock" and b.calls == 0
    assert r["decision"] == "EMERGENCY_HALT" and r["aws_telemetry"]["incident_severity"] == "LEVEL_3"


def test_parse_json_tolerates_wrapping():
    assert parse_json('noise {"a": 1} trailing') == {"a": 1}
    assert parse_json("nope") is None


def test_distillation_loss_properties():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([[-100, -100, 3, 4, 5, 1], [-100, 2, 2, -100, -100, -100]])
    loss_same, ce, kl = distillation_loss(logits, logits, labels, alpha=0.5, temperature=4.0)
    assert kl.abs() < 1e-5                       # identical teacher -> zero KL
    assert torch.isclose(loss_same, 0.5 * ce)
    loss_diff, _, kl2 = distillation_loss(logits, torch.randn(2, 6, 11), labels)
    assert kl2 > 0 and loss_diff > loss_same
    # masked positions must not matter
    t2 = logits.clone()
    t2[0, 0] += 100.0                            # position predicting a masked label
    _, _, kl3 = distillation_loss(logits, t2, labels)
    assert kl3.abs() < 1e-5


def test_score_set_on_reference_outputs():
    gold = json.load(open("data/golden/eval_cases.json"))["safety_critical"][:10]
    r = score_set(gold, [(json.dumps(c["reference"]), 1.0) for c in gold])
    assert r["json_validity"] == r["safety_pass"] == r["exact_match"] == 1.0


def test_vllm_command_uses_merged_sft_base():
    cmd = build_command("outputs/roboscript-sft-merged", "outputs/roboscript-dpo", 8080)
    assert cmd[cmd.index("--model") + 1] == "outputs/roboscript-sft-merged"
    assert "roboscript-dpo=outputs/roboscript-dpo" in cmd
