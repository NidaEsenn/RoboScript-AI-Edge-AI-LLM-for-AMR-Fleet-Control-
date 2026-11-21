import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # modules use repo-relative data paths

SEEDS = json.loads((ROOT / "data/seed/examples.json").read_text())


@pytest.fixture
def seeds():
    return SEEDS


def scenario(codes=(), speed=1.0, battery=70, dist=4.0, status="CLEAR", yolo=None):
    return {"yolo": yolo, "lidar": {"min_distance_meters": dist, "status": status},
            "telemetry": {"current_speed_mps": speed, "battery_level_pct": battery,
                          "active_fault_codes": list(codes)}}


ESTOP = scenario(["ERR_ESTOP_TRIGGERED"])
PERSON = scenario(yolo={"cls": "person", "bbox_xywh": [320, 240, 80, 180], "confidence": 0.92},
                  dist=0.38, status="PATH_OBSTRUCTED", speed=1.2)
BATTERY_LOW = scenario(["ERR_BATTERY_LOW"], battery=20)
BATTERY_CRIT = scenario(["ERR_BATTERY_CRITICAL"], battery=9)
