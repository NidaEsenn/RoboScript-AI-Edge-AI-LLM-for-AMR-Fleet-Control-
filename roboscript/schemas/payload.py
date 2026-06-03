"""Output-side schemas: the RoboResponse the model must produce (PRD Section 7)."""
from pydantic import BaseModel, Field
from typing import Literal


class CmdVelTwist(BaseModel):
    linear:  dict = Field(default={"x": 0.0, "y": 0.0, "z": 0.0})
    angular: dict = Field(default={"x": 0.0, "y": 0.0, "z": 0.0})


class ROS2CmdVel(BaseModel):
    topic:    Literal["/cmd_vel"]
    msg_type: Literal["geometry_msgs/msg/Twist"]
    cmd_vel:  CmdVelTwist


class ROS2Nav2Goal(BaseModel):
    action_server: Literal["/navigate_to_pose"]
    action_type:   Literal["nav2_msgs/action/NavigateToPose"]
    goal:          dict


class AWSTelemetry(BaseModel):
    incident_severity:   Literal["LEVEL_0", "LEVEL_1", "LEVEL_2", "LEVEL_3"]
    zone_update:         str | None = None
    request_human_audit: bool = False


class RoboResponse(BaseModel):
    decision:      str
    reasoning:     str = Field(min_length=10, max_length=200)
    payload_type:  Literal["CMD_VEL_DIRECT", "NAV2_ACTION", "FAULT_REPORT_ONLY"]
    ros2:          ROS2CmdVel | ROS2Nav2Goal | None
    aws_telemetry: AWSTelemetry


# Fallback constant — always physically safe.
EMERGENCY_HALT_FALLBACK = RoboResponse(
    decision="EMERGENCY_HALT",
    reasoning="Model output validation failed. Defaulting to safe halt.",
    payload_type="CMD_VEL_DIRECT",
    ros2=ROS2CmdVel(topic="/cmd_vel", msg_type="geometry_msgs/msg/Twist",
                    cmd_vel=CmdVelTwist()),
    aws_telemetry=AWSTelemetry(incident_severity="LEVEL_3", request_human_audit=True),
)
