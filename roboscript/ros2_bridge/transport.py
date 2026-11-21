"""ROS 2 transports for the bridge.

``RclpyTransport`` talks to a real ROS 2 graph (geometry_msgs/Twist on /cmd_vel,
std_msgs/String on /fault_events, nav2_msgs/action/NavigateToPose action client).
``InProcessBus`` implements the same interface in pure Python so the bridge logic
and the integration tests run anywhere (CI, laptops without ROS). The message
dicts mirror the ROS message field layout exactly, so the assertions made against
the in-process bus are the same ones made against the real topics.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from typing import Callable, Protocol


class Transport(Protocol):
    def publish(self, topic: str, msg_type: str, msg: dict) -> None: ...
    def send_nav_goal(self, goal: dict, timeout_s: float = 2.0) -> bool: ...
    def subscribe(self, topic: str, msg_type: str, cb: Callable[[dict], None]) -> None: ...
    def spin_once(self, timeout_s: float = 0.1) -> None: ...


class InProcessBus:
    """Synchronous pub/sub + a pluggable NavigateToPose action server."""

    def __init__(self):
        self._subs: dict[str, list[Callable[[dict], None]]] = defaultdict(list)
        self._nav_server: Callable[[dict], bool] | None = None
        self._lock = threading.Lock()
        self.log: list[tuple[str, dict]] = []

    def publish(self, topic: str, msg_type: str, msg: dict) -> None:
        with self._lock:
            self.log.append((topic, msg))
            subs = list(self._subs[topic])
        for cb in subs:
            cb(msg)

    def subscribe(self, topic: str, msg_type: str, cb: Callable[[dict], None]) -> None:
        self._subs[topic].append(cb)

    def register_nav_server(self, handler: Callable[[dict], bool]) -> None:
        self._nav_server = handler

    def send_nav_goal(self, goal: dict, timeout_s: float = 2.0) -> bool:
        if self._nav_server is None:
            return False                       # == wait_for_server timed out
        return bool(self._nav_server(goal))

    def spin_once(self, timeout_s: float = 0.1) -> None:
        return None                            # delivery is synchronous


class RclpyTransport:  # pragma: no cover - needs a ROS 2 install
    def __init__(self, node_name: str = "roboscript_bridge"):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String
        from nav2_msgs.action import NavigateToPose

        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.node = Node(node_name)
        self._types = {"geometry_msgs/msg/Twist": Twist, "std_msgs/msg/String": String}
        self._pubs: dict = {}
        self._nav = ActionClient(self.node, NavigateToPose, "/navigate_to_pose")
        self._NavigateToPose = NavigateToPose

    def _to_msg(self, msg_type: str, msg: dict):
        cls = self._types[msg_type]
        m = cls()
        if msg_type == "geometry_msgs/msg/Twist":
            for part in ("linear", "angular"):
                for k in ("x", "y", "z"):
                    setattr(getattr(m, part), k, float(msg[part][k]))
        else:
            m.data = msg["data"]
        return m

    def _from_msg(self, msg_type: str, m) -> dict:
        if msg_type == "geometry_msgs/msg/Twist":
            return {p: {k: getattr(getattr(m, p), k) for k in "xyz"} for p in ("linear", "angular")}
        return {"data": m.data}

    def publish(self, topic: str, msg_type: str, msg: dict) -> None:
        if topic not in self._pubs:
            self._pubs[topic] = self.node.create_publisher(self._types[msg_type], topic, 10)
        self._pubs[topic].publish(self._to_msg(msg_type, msg))

    def subscribe(self, topic: str, msg_type: str, cb: Callable[[dict], None]) -> None:
        self.node.create_subscription(self._types[msg_type], topic,
                                      lambda m: cb(self._from_msg(msg_type, m)), 10)

    def send_nav_goal(self, goal: dict, timeout_s: float = 2.0) -> bool:
        if not self._nav.wait_for_server(timeout_sec=timeout_s):
            return False
        g = self._NavigateToPose.Goal()
        pose = goal["pose"]
        g.pose.header.frame_id = pose.get("header", {}).get("frame_id", "map")
        g.pose.pose.position.x = float(pose["position"]["x"])
        g.pose.pose.position.y = float(pose["position"]["y"])
        g.pose.pose.orientation.w = float(pose.get("orientation", {}).get("w", 1.0))
        if "behavior_tree" in goal:
            g.behavior_tree = goal["behavior_tree"]
        fut = self._nav.send_goal_async(g)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout_s)
        handle = fut.result()
        return bool(handle and handle.accepted)

    def spin_once(self, timeout_s: float = 0.1) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_s)


def rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))
