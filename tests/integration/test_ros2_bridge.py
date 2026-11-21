"""ROS 2 bridge integration tests (PRD AC-08).

telemetry -> FastAPI (/v1/emergency or /v1/act) -> bridge.dispatch -> mock subscriber.
Runs on the in-process bus everywhere; the ``rclpy`` variants run the same flow on
a real ROS 2 graph when rclpy is importable (e.g. inside docker/Dockerfile.ros2).
"""
import json

import pytest
from fastapi.testclient import TestClient

from roboscript.ros2_bridge.bridge_node import TELEMETRY_TOPIC, RoboScriptBridge
from roboscript.ros2_bridge.mock_subscriber import MockNav2Server, MockSubscriber
from roboscript.ros2_bridge.transport import InProcessBus, rclpy_available
from roboscript.serve.api import create_app
from roboscript.serve.backends import RulesBackend
from tests.conftest import BATTERY_CRIT, BATTERY_LOW, ESTOP, PERSON


@pytest.fixture
def rig():
    bus = InProcessBus()
    sub = MockSubscriber(bus)
    nav = MockNav2Server(bus)
    client = TestClient(create_app(RulesBackend()))
    bridge = RoboScriptBridge(bus, http_client=client)
    return bus, sub, nav, client, bridge


def test_emergency_halt_publishes_zero_velocity(rig):
    bus, sub, nav, client, bridge = rig
    resp = client.post("/v1/emergency", json=ESTOP)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["decision"] == "EMERGENCY_HALT"
    bridge.dispatch(payload)
    twists = sub.on("/cmd_vel")
    assert len(twists) == 1
    assert twists[0]["linear_x"] == 0.0 and twists[0]["angular_z"] == 0.0


def test_person_detection_mandatory_halt(rig):
    bus, sub, nav, client, bridge = rig
    payload = client.post("/v1/emergency", json=PERSON).json()
    assert payload["decision"] == "EMERGENCY_HALT"
    assert bridge.dispatch(payload) == "CMD_VEL_DIRECT"
    assert sub.on("/cmd_vel")[-1] == {"topic": "/cmd_vel", "linear_x": 0.0,
                                      "linear_y": 0.0, "angular_z": 0.0}
    event = json.loads(sub.on("/fault_events")[-1]["data"])
    assert event["incident_severity"] == "LEVEL_3" and event["request_human_audit"] is True


def test_estop_mandatory_halt_via_telemetry_topic(rig):
    bus, sub, nav, client, bridge = rig
    bridge.listen()
    bus.publish(TELEMETRY_TOPIC, "std_msgs/msg/String", {"data": json.dumps(ESTOP)})
    assert sub.on("/cmd_vel")[-1]["linear_x"] == 0.0
    assert bridge.dispatched[-1]["decision"] == "EMERGENCY_HALT"


def test_fault_report_only_logs_without_motion(rig):
    bus, sub, nav, client, bridge = rig
    payload = client.post("/v1/diagnose", json=BATTERY_LOW).json()
    assert bridge.dispatch(payload) == "FAULT_REPORT_ONLY"
    assert sub.on("/cmd_vel") == []
    assert json.loads(sub.on("/fault_events")[0]["data"])["decision"] == "CONTINUE_WITH_FAULT_LOG"


def test_nav2_goal_sent_to_action_server(rig):
    bus, sub, nav, client, bridge = rig
    payload = client.post("/v1/navigate", json=BATTERY_CRIT).json()
    assert bridge.dispatch(payload) == "NAV2_ACTION"
    assert len(nav.goals) == 1 and nav.goals[0]["behavior_tree"] == "navigate_to_dock.xml"


def test_nav2_unavailable_halts(rig):
    bus, sub, nav, client, bridge = rig
    nav.accept = False
    payload = client.post("/v1/navigate", json=BATTERY_CRIT).json()
    assert bridge.dispatch(payload) == "halt:nav2_unavailable"
    assert sub.on("/cmd_vel")[-1]["linear_x"] == 0.0


def test_invalid_payload_never_reaches_motors(rig):
    bus, sub, nav, client, bridge = rig
    bad = {"decision": "EMERGENCY_HALT", "payload_type": "CMD_VEL_DIRECT",
           "ros2": {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist",
                    "cmd_vel": {"linear": {"x": 2.0}, "angular": {}}}}
    assert bridge.dispatch(bad) == "halt:invalid_payload"
    assert [m["linear_x"] for m in sub.on("/cmd_vel")] == [0.0]


def test_api_failure_halts(rig):
    bus, sub, nav, client, bridge = rig

    class Broken:
        def post(self, *a, **k):
            raise ConnectionError("api down")
    bridge.http = Broken()
    bridge.listen()
    bus.publish(TELEMETRY_TOPIC, "std_msgs/msg/String", {"data": json.dumps(BATTERY_LOW)})
    assert sub.on("/cmd_vel")[-1]["linear_x"] == 0.0


@pytest.mark.skipif(not rclpy_available(), reason="rclpy not installed (run inside docker/Dockerfile.ros2)")
def test_rclpy_emergency_halt_publishes_zero_velocity():  # pragma: no cover
    from roboscript.ros2_bridge.transport import RclpyTransport
    pub_side = RclpyTransport("roboscript_bridge_test")
    sub_side = RclpyTransport("mock_subscriber_test")
    sub = MockSubscriber(sub_side)
    bridge = RoboScriptBridge(pub_side)
    payload = TestClient(create_app(RulesBackend())).post("/v1/emergency", json=ESTOP).json()
    for _ in range(20):
        bridge.dispatch(payload)
        sub_side.spin_once(0.1)
        if sub.on("/cmd_vel"):
            break
    assert sub.on("/cmd_vel") and sub.on("/cmd_vel")[0]["linear_x"] == 0.0
