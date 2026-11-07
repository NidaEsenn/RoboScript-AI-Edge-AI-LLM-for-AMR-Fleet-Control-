"""Mock ROS 2 consumers for integration tests (PRD Section 6.8).

MockSubscriber listens on /cmd_vel and /fault_events and records what it receives;
MockNav2Server accepts NavigateToPose goals and records them. Both work on any
``Transport`` (in-process bus or rclpy), so the same assertions run in CI and on a
real ROS 2 graph.
"""
from __future__ import annotations

from roboscript.ros2_bridge.transport import Transport


class MockSubscriber:
    def __init__(self, transport: Transport):
        self.received: list[dict] = []
        transport.subscribe("/cmd_vel", "geometry_msgs/msg/Twist", self._on_cmd_vel)
        transport.subscribe("/fault_events", "std_msgs/msg/String", self._on_fault_event)

    def _on_cmd_vel(self, msg: dict) -> None:
        self.received.append({"topic": "/cmd_vel", "linear_x": msg["linear"]["x"],
                              "linear_y": msg["linear"]["y"], "angular_z": msg["angular"]["z"]})

    def _on_fault_event(self, msg: dict) -> None:
        self.received.append({"topic": "/fault_events", "data": msg["data"]})

    def on(self, topic: str) -> list[dict]:
        return [m for m in self.received if m["topic"] == topic]


class MockNav2Server:
    def __init__(self, bus, accept: bool = True):
        self.goals: list[dict] = []
        self.accept = accept
        bus.register_nav_server(self._handle)

    def _handle(self, goal: dict) -> bool:
        self.goals.append(goal)
        return self.accept
