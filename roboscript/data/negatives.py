"""Hard negatives for DPO.

The original pipeline paired every chosen answer with the *same* rejected answer
("CONTINUE at 0.5 m/s"). That pair is trivially separable — the policy learns to
avoid one string, not to reason about safety, and DPO margins saturate in a few
steps. Here each chosen response gets a scenario-aware rejected response drawn
from a family of realistic failure modes:

  forward_motion   drive on via /cmd_vel despite the fault (the canonical hazard)
  soft_stop        a plausible but non-stopping alternative (reroute / log only)
                   when the reference policy requires a halt
  over_halt        for non-stop cases: wrong action (e.g. log-only on critical battery)
  under_report     right motion, but life-safety incident logged as LEVEL_0, no audit
  wrong_shape      payload_type that contradicts the ros2 field (schema-invalid)

Every rejected answer is checked to be *worse* than the chosen one under
``roboscript.safety.check`` before it is used.
"""
from __future__ import annotations

import copy
import random

from roboscript.safety import check, reference_action

_TWIST = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}


def _cmd_vel(x: float = 0.0, z: float = 0.0) -> dict:
    t = copy.deepcopy(_TWIST)
    t["linear"]["x"], t["angular"]["z"] = x, z
    return {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist", "cmd_vel": t}


def _nav(rng: random.Random, bt: str = "navigate_w_replanning_and_recovery.xml") -> dict:
    return {"action_server": "/navigate_to_pose",
            "action_type": "nav2_msgs/action/NavigateToPose",
            "goal": {"pose": {"header": {"frame_id": "map"},
                              "position": {"x": round(rng.uniform(1, 30), 1),
                                           "y": round(rng.uniform(1, 30), 1), "z": 0.0},
                              "orientation": {"w": 1.0}},
                     "behavior_tree": bt}}


def forward_motion(scenario: dict, chosen: dict, rng: random.Random) -> dict:
    speed = scenario["telemetry"].get("current_speed_mps", 0.5) or 0.5
    return {"decision": "CONTINUE",
            "reasoning": "Fault appears transient; maintaining current velocity to meet throughput.",
            "payload_type": "CMD_VEL_DIRECT",
            "ros2": _cmd_vel(round(max(0.3, speed), 2), round(rng.choice([0.0, 0.0, 0.2]), 2)),
            "aws_telemetry": {"incident_severity": "LEVEL_0", "request_human_audit": False}}


def soft_stop(scenario: dict, chosen: dict, rng: random.Random) -> dict:
    if rng.random() < 0.5:
        return {"decision": "DYNAMIC_REROUTE",
                "reasoning": "Obstruction or fault noted; replanning around it without stopping.",
                "payload_type": "NAV2_ACTION", "ros2": _nav(rng),
                "aws_telemetry": {"incident_severity": "LEVEL_1", "request_human_audit": False}}
    return {"decision": "CONTINUE_WITH_FAULT_LOG",
            "reasoning": "Logging the fault for maintenance while the robot keeps moving.",
            "payload_type": "FAULT_REPORT_ONLY", "ros2": None,
            "aws_telemetry": {"incident_severity": "LEVEL_1", "request_human_audit": False}}


def over_halt_or_wrong(scenario: dict, chosen: dict, rng: random.Random) -> dict:
    if chosen["decision"] == "NAVIGATE_TO_CHARGING_DOCK":
        return {"decision": "CONTINUE_WITH_FAULT_LOG",
                "reasoning": "Battery low but task almost done; continuing and logging the fault.",
                "payload_type": "FAULT_REPORT_ONLY", "ros2": None,
                "aws_telemetry": {"incident_severity": "LEVEL_0", "request_human_audit": False}}
    return forward_motion(scenario, chosen, rng)


def under_report(scenario: dict, chosen: dict, rng: random.Random) -> dict:
    r = copy.deepcopy(chosen)
    r["aws_telemetry"] = {"incident_severity": "LEVEL_0", "request_human_audit": False}
    return r


def wrong_shape(scenario: dict, chosen: dict, rng: random.Random) -> dict:
    r = copy.deepcopy(chosen)
    if r["payload_type"] == "CMD_VEL_DIRECT":
        r["payload_type"] = "NAV2_ACTION"           # keeps a cmd_vel body -> mismatch
    elif r["payload_type"] == "NAV2_ACTION":
        r["payload_type"] = "CMD_VEL_DIRECT"
    else:
        r["ros2"] = _cmd_vel(0.4)                   # log-only that also drives
    return r


def make_hard_negative(scenario: dict, chosen: dict, rng: random.Random) -> tuple[dict, str]:
    """Return (rejected_response, failure_mode). Guaranteed to score worse than chosen."""
    ref = reference_action(scenario)
    modes = ["forward_motion", "wrong_shape"]
    if ref is not None and ref.must_stop:
        modes += ["soft_stop", "soft_stop"]
        if ref.severity == "LEVEL_3":
            modes.append("under_report")
    else:
        modes += ["over_halt_or_wrong"]
    fns = {"forward_motion": forward_motion, "soft_stop": soft_stop,
           "over_halt_or_wrong": over_halt_or_wrong, "under_report": under_report,
           "wrong_shape": wrong_shape}
    cv = check(scenario, chosen)
    rng.shuffle(modes)
    for m in modes:
        rej = fns[m](scenario, chosen, rng)
        rv = check(scenario, rej)
        worse = (not rv.valid_schema) or (cv.safe and not rv.safe) or (
            len(rv.violations) > len(cv.violations))
        if worse and rej != chosen:
            return rej, m
    return forward_motion(scenario, chosen, rng), "forward_motion"
