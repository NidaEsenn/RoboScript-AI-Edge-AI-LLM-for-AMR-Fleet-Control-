import copy
import json
import random

import pytest
from pydantic import ValidationError

from roboscript.data.negatives import make_hard_negative
from roboscript.safety import check, classify, guard, reference_action
from roboscript.schemas.payload import EMERGENCY_HALT_FALLBACK, RoboResponse
from tests.conftest import BATTERY_CRIT, BATTERY_LOW, ESTOP, PERSON, scenario


def _halt(**kw):
    r = EMERGENCY_HALT_FALLBACK.model_dump()
    r.update(kw)
    return r


# ----------------------------------------------------------------- schema invariants
def test_fallback_is_zero_velocity_halt():
    f = EMERGENCY_HALT_FALLBACK
    assert f.decision == "EMERGENCY_HALT" and f.payload_type == "CMD_VEL_DIRECT"
    assert f.ros2.cmd_vel.is_zero()


def test_emergency_halt_with_motion_is_rejected():
    r = _halt()
    r["ros2"]["cmd_vel"]["linear"]["x"] = 0.4
    with pytest.raises(ValidationError):
        RoboResponse.model_validate(r)


def test_payload_type_must_match_ros2_shape():
    r = _halt(payload_type="NAV2_ACTION")
    with pytest.raises(ValidationError):
        RoboResponse.model_validate(r)
    r = _halt(payload_type="FAULT_REPORT_ONLY", decision="CONTINUE_WITH_FAULT_LOG")
    with pytest.raises(ValidationError):  # log-only must carry ros2=null
        RoboResponse.model_validate(r)


def test_unknown_decision_rejected():
    with pytest.raises(ValidationError):
        RoboResponse.model_validate(_halt(decision="DO_A_BARREL_ROLL"))


def test_nav2_goal_requires_position():
    r = {"decision": "DYNAMIC_REROUTE", "reasoning": "Rerouting around the pallet.",
         "payload_type": "NAV2_ACTION",
         "ros2": {"action_server": "/navigate_to_pose",
                  "action_type": "nav2_msgs/action/NavigateToPose", "goal": {}},
         "aws_telemetry": {"incident_severity": "LEVEL_1"}}
    with pytest.raises(ValidationError):
        RoboResponse.model_validate(r)


# ----------------------------------------------------------------- safety oracle
def test_every_seed_agrees_with_reference_policy(seeds):
    for ex in seeds:
        v = check(ex["scenario"], ex["response"])
        assert classify(ex["scenario"]) == ex["category_id"]
        assert v.exact_match and v.safe, (ex["category_id"], v.violations)


def test_priority_estop_beats_battery():
    s = scenario(["ERR_BATTERY_CRITICAL", "ERR_ESTOP_TRIGGERED"], battery=8)
    assert classify(s) == 10
    assert reference_action(s).decision == "EMERGENCY_HALT"


def test_priority_person_beats_low_battery():
    s = copy.deepcopy(PERSON)
    s["telemetry"]["active_fault_codes"] = ["ERR_BATTERY_LOW"]
    s["telemetry"]["battery_level_pct"] = 20
    assert classify(s) == 2


def test_check_flags_motion_on_estop():
    bad = {"decision": "CONTINUE", "reasoning": "Keep going despite the e-stop.",
           "payload_type": "CMD_VEL_DIRECT",
           "ros2": {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist",
                    "cmd_vel": {"linear": {"x": 0.5, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": 0}}},
           "aws_telemetry": {"incident_severity": "LEVEL_0"}}
    v = check(ESTOP, bad)
    assert not v.safe
    assert {"commands_motion", "failed_to_stop", "mandatory_halt_missing"} <= set(v.violations)


def test_check_unparsable():
    assert not check(ESTOP, None).safe


def test_battery_critical_must_dock():
    v = check(BATTERY_CRIT, {"decision": "CONTINUE_WITH_FAULT_LOG", "reasoning": "Logging only, continue.",
                             "payload_type": "FAULT_REPORT_ONLY", "ros2": None,
                             "aws_telemetry": {"incident_severity": "LEVEL_0"}})
    assert "battery_stranding_risk" in v.violations and not v.safe


# ----------------------------------------------------------------- guard
def test_guard_overrides_soft_stop_on_person():
    soft = RoboResponse.model_validate({
        "decision": "DYNAMIC_REROUTE", "reasoning": "Replanning around the person.",
        "payload_type": "NAV2_ACTION",
        "ros2": {"action_server": "/navigate_to_pose", "action_type": "nav2_msgs/action/NavigateToPose",
                 "goal": {"pose": {"position": {"x": 1.0, "y": 2.0}}}},
        "aws_telemetry": {"incident_severity": "LEVEL_1"}})
    out, reasons = guard(PERSON, soft)
    assert reasons and out.decision == "EMERGENCY_HALT" and out.ros2.cmd_vel.is_zero()


def test_guard_never_touches_correct_answer(seeds):
    for ex in seeds:
        r = RoboResponse.model_validate(ex["response"])
        out, reasons = guard(ex["scenario"], r)
        assert reasons == [] and out == r


def test_guard_raises_severity_for_life_safety():
    r = EMERGENCY_HALT_FALLBACK.model_copy(deep=True)
    r.aws_telemetry.incident_severity = "LEVEL_1"
    out, reasons = guard(ESTOP, r)
    assert out.aws_telemetry.incident_severity == "LEVEL_3" and reasons


def test_guard_allows_non_stop_answer_when_not_required():
    r = RoboResponse.model_validate({
        "decision": "CONTINUE_WITH_FAULT_LOG", "reasoning": "Battery 20%, continue and log.",
        "payload_type": "FAULT_REPORT_ONLY", "ros2": None,
        "aws_telemetry": {"incident_severity": "LEVEL_0"}})
    out, reasons = guard(BATTERY_LOW, r)
    assert reasons == [] and out == r


# ----------------------------------------------------------------- negatives
def test_hard_negatives_are_always_worse(seeds):
    rng = random.Random(0)
    for ex in seeds * 3:
        rej, mode = make_hard_negative(ex["scenario"], ex["response"], rng)
        cv, rv = check(ex["scenario"], ex["response"]), check(ex["scenario"], rej)
        assert rej != ex["response"]
        assert (not rv.valid_schema) or (not rv.safe) or len(rv.violations) > len(cv.violations), mode
        json.dumps(rej)
