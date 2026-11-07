"""Output-side schemas: the RoboResponse the model must produce (PRD Section 7).

The JSON wire format is unchanged from the PRD, but the models are stricter than the
original sketch so that the README's safety invariants are enforced by Pydantic
itself rather than only by the offline rule-check:

  * ``cmd_vel`` is typed (``Vector3`` of floats) instead of a free ``dict``;
  * ``payload_type`` and the shape of ``ros2`` must agree
    (CMD_VEL_DIRECT <-> /cmd_vel Twist, NAV2_ACTION <-> NavigateToPose goal,
    FAULT_REPORT_ONLY <-> null), so cmd_vel and Nav2 can never co-occur;
  * ``decision`` must come from the fault taxonomy's closed vocabulary;
  * any halt decision carried on /cmd_vel must command exactly zero velocity.

Anything that fails here is replaced by ``EMERGENCY_HALT_FALLBACK`` at serving time.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Decision = Literal[
    "EMERGENCY_HALT",
    "DYNAMIC_REROUTE",
    "HALT_AND_REPORT",
    "NAVIGATE_TO_CHARGING_DOCK",
    "CONTINUE_WITH_FAULT_LOG",
    "THERMAL_COOLDOWN",
    "REDUCE_SPEED",
    "CONTINUE",
    "HALT_AND_RELOCALIZE",
    "DAMAGE_ASSESSMENT",
    "LIDAR_ONLY_MODE",
]

# Decisions whose physical meaning is "the robot stops". When one of these is sent
# over /cmd_vel the Twist must be all zeros.
HALT_DECISIONS = frozenset({"EMERGENCY_HALT", "HALT_AND_REPORT", "HALT_AND_RELOCALIZE"})


class Vector3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @field_validator("x", "y", "z")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("velocity components must be finite")
        return v

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0


class CmdVelTwist(BaseModel):
    model_config = ConfigDict(extra="forbid")
    linear: Vector3 = Field(default_factory=Vector3)
    angular: Vector3 = Field(default_factory=Vector3)

    def is_zero(self) -> bool:
        return self.linear.is_zero() and self.angular.is_zero()


class ROS2CmdVel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: Literal["/cmd_vel"]
    msg_type: Literal["geometry_msgs/msg/Twist"]
    cmd_vel: CmdVelTwist


class ROS2Nav2Goal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_server: Literal["/navigate_to_pose"]
    action_type: Literal["nav2_msgs/action/NavigateToPose"]
    goal: dict

    @field_validator("goal")
    @classmethod
    def _goal_has_pose(cls, g: dict) -> dict:
        pose = g.get("pose")
        if not isinstance(pose, dict) or not isinstance(pose.get("position"), dict):
            raise ValueError("Nav2 goal must contain pose.position")
        pos = pose["position"]
        for k in ("x", "y"):
            if not isinstance(pos.get(k), (int, float)) or isinstance(pos.get(k), bool):
                raise ValueError(f"Nav2 goal pose.position.{k} must be a number")
        return g


class AWSTelemetry(BaseModel):
    incident_severity: Literal["LEVEL_0", "LEVEL_1", "LEVEL_2", "LEVEL_3"]
    zone_update: str | None = None
    request_human_audit: bool = False


class RoboResponse(BaseModel):
    decision: Decision
    reasoning: str = Field(min_length=10, max_length=200)
    payload_type: Literal["CMD_VEL_DIRECT", "NAV2_ACTION", "FAULT_REPORT_ONLY"]
    ros2: ROS2CmdVel | ROS2Nav2Goal | None
    aws_telemetry: AWSTelemetry

    @model_validator(mode="after")
    def _payload_matches_ros2(self) -> "RoboResponse":
        expected = {
            "CMD_VEL_DIRECT": ROS2CmdVel,
            "NAV2_ACTION": ROS2Nav2Goal,
            "FAULT_REPORT_ONLY": type(None),
        }[self.payload_type]
        if not isinstance(self.ros2, expected):
            raise ValueError(
                f"payload_type {self.payload_type} requires ros2 of type {expected.__name__}")
        if self.decision in HALT_DECISIONS and isinstance(self.ros2, ROS2CmdVel):
            if not self.ros2.cmd_vel.is_zero():
                raise ValueError(f"{self.decision} must command zero velocity on /cmd_vel")
        if self.decision == "EMERGENCY_HALT" and self.payload_type != "CMD_VEL_DIRECT":
            raise ValueError("EMERGENCY_HALT must be delivered as CMD_VEL_DIRECT")
        return self


def zero_twist_ros2() -> ROS2CmdVel:
    return ROS2CmdVel(topic="/cmd_vel", msg_type="geometry_msgs/msg/Twist",
                      cmd_vel=CmdVelTwist())


# Fallback constant — always physically safe.
EMERGENCY_HALT_FALLBACK = RoboResponse(
    decision="EMERGENCY_HALT",
    reasoning="Model output validation failed. Defaulting to safe halt.",
    payload_type="CMD_VEL_DIRECT",
    ros2=zero_twist_ros2(),
    aws_telemetry=AWSTelemetry(incident_severity="LEVEL_3", request_human_audit=True),
)
