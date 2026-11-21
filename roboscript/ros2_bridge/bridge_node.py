"""Session F — ROS 2 bridge node (PRD Section 6.8).

Consumes validated RoboResponse payloads and publishes them to the right ROS 2
interface:

  CMD_VEL_DIRECT    -> /cmd_vel           geometry_msgs/msg/Twist
  NAV2_ACTION       -> /navigate_to_pose  nav2_msgs/action/NavigateToPose (goal)
  FAULT_REPORT_ONLY -> /fault_events      std_msgs/msg/String (aws_telemetry JSON)

plus /fault_events for any LEVEL_2+ incident regardless of payload type.

Safety rules inside the bridge (it is the last hop before the motors):
  * every payload is re-validated with RoboResponse; anything invalid -> zero Twist
  * a Nav2 goal that cannot be delivered (no action server / rejected) -> zero Twist
  * if the API call fails or times out in the telemetry loop -> zero Twist

Run modes:
    python -m roboscript.ros2_bridge.bridge_node              # real ROS 2 if rclpy present
    python -m roboscript.ros2_bridge.bridge_node --demo       # in-process bus demo, no ROS
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from pydantic import ValidationError

from roboscript.ros2_bridge.transport import InProcessBus, Transport, dumps, rclpy_available
from roboscript.schemas.payload import RoboResponse

log = logging.getLogger("roboscript.bridge")

ZERO_TWIST = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}
TELEMETRY_TOPIC = "/roboscript/telemetry"


class RoboScriptBridge:
    def __init__(self, transport: Transport, api_url: str | None = None, http_client=None,
                 nav_timeout_s: float = 2.0):
        self.t = transport
        self.api_url = api_url or os.getenv("FASTAPI_URL", "http://localhost:8000")
        self.http = http_client
        self.nav_timeout_s = nav_timeout_s
        self.dispatched: list[dict] = []

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, payload: dict) -> str:
        """Route one payload. Returns the action taken (for logging/tests)."""
        try:
            resp = RoboResponse.model_validate(payload)
        except ValidationError as e:
            log.error("invalid payload, emitting halt: %s", e.errors()[:1])
            self._halt()
            return "halt:invalid_payload"

        action = resp.payload_type
        if resp.payload_type == "CMD_VEL_DIRECT":
            self.t.publish("/cmd_vel", "geometry_msgs/msg/Twist", resp.ros2.cmd_vel.model_dump())
        elif resp.payload_type == "NAV2_ACTION":
            if not self.t.send_nav_goal(resp.ros2.goal, timeout_s=self.nav_timeout_s):
                log.error("Nav2 goal not accepted; emitting halt")
                self._halt()
                action = "halt:nav2_unavailable"
        if resp.payload_type == "FAULT_REPORT_ONLY" or resp.aws_telemetry.incident_severity in (
                "LEVEL_2", "LEVEL_3"):
            self.t.publish("/fault_events", "std_msgs/msg/String",
                           {"data": dumps({"decision": resp.decision,
                                           **resp.aws_telemetry.model_dump()})})
        self.dispatched.append({"action": action, "decision": resp.decision})
        return action

    def _halt(self) -> None:
        self.t.publish("/cmd_vel", "geometry_msgs/msg/Twist", ZERO_TWIST)

    # ----------------------------------------------------------- telemetry loop
    def on_telemetry(self, msg: dict) -> str:
        """Telemetry (std_msgs/String JSON) -> POST /v1/act -> dispatch."""
        try:
            scenario = json.loads(msg["data"])
            client = self.http
            if client is None:
                import httpx
                client = self.http = httpx.Client(base_url=self.api_url, timeout=1.0)
            r = client.post("/v1/act", json=scenario)
            r.raise_for_status()
            return self.dispatch(r.json())
        except Exception as e:  # noqa: BLE001 - any failure must stop the robot
            log.error("telemetry handling failed (%s); emitting halt", e)
            self._halt()
            return "halt:api_error"

    def listen(self) -> None:
        self.t.subscribe(TELEMETRY_TOPIC, "std_msgs/msg/String", self.on_telemetry)


def _demo() -> None:
    from fastapi.testclient import TestClient
    from roboscript.ros2_bridge.mock_subscriber import MockNav2Server, MockSubscriber
    from roboscript.serve.api import create_app
    from roboscript.serve.backends import RulesBackend

    bus = InProcessBus()
    sub = MockSubscriber(bus)
    nav = MockNav2Server(bus)
    bridge = RoboScriptBridge(bus, http_client=TestClient(create_app(RulesBackend())))
    bridge.listen()
    scenarios = {
        "ESTOP": {"yolo": None, "lidar": {"min_distance_meters": 2.0, "status": "CLEAR"},
                  "telemetry": {"current_speed_mps": 1.0, "battery_level_pct": 65,
                                "active_fault_codes": ["ERR_ESTOP_TRIGGERED"]}},
        "battery critical": {"yolo": None, "lidar": {"min_distance_meters": 3.2, "status": "CLEAR"},
                             "telemetry": {"current_speed_mps": 1.0, "battery_level_pct": 9,
                                           "active_fault_codes": ["ERR_BATTERY_CRITICAL"]}},
        "battery low": {"yolo": None, "lidar": {"min_distance_meters": 4.0, "status": "CLEAR"},
                        "telemetry": {"current_speed_mps": 1.0, "battery_level_pct": 20,
                                      "active_fault_codes": ["ERR_BATTERY_LOW"]}},
    }
    for name, s in scenarios.items():
        n_msgs, n_goals = len(sub.received), len(nav.goals)
        bus.publish(TELEMETRY_TOPIC, "std_msgs/msg/String", {"data": json.dumps(s)})
        print(f"{name}: bridge -> {bridge.dispatched[-1]}")
        for m in sub.received[n_msgs:]:
            print(f"    mock subscriber got {m}")
        for g in nav.goals[n_goals:]:
            print(f"    mock Nav2 server got goal {g['pose']['position']} bt={g.get('behavior_tree')}")


def main() -> None:  # pragma: no cover - entry point
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="in-process bus + rules backend")
    a = ap.parse_args()
    if a.demo or not rclpy_available():
        if not a.demo:
            log.warning("rclpy not available; running the in-process demo instead")
        _demo()
        return
    from roboscript.ros2_bridge.transport import RclpyTransport
    t = RclpyTransport()
    bridge = RoboScriptBridge(t)
    bridge.listen()
    t.node.get_logger().info("RoboScript bridge node ready.")
    while t.rclpy.ok():
        t.spin_once(0.1)


if __name__ == "__main__":
    main()
