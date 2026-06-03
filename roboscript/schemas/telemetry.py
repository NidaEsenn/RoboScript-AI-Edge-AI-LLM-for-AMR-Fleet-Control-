"""Input-side schemas: the sensor telemetry that describes an AMR scenario."""
from pydantic import BaseModel, Field
from typing import Literal


class YOLODetection(BaseModel):
    cls:        str
    bbox_xywh:  list[float]
    confidence: float = Field(ge=0.0, le=1.0)


class LiDARReading(BaseModel):
    min_distance_meters: float
    status: Literal["CLEAR", "PATH_OBSTRUCTED", "SENSOR_FAULT"]


class FleetTelemetry(BaseModel):
    current_speed_mps:  float
    battery_level_pct:  float
    active_fault_codes: list[str]


class Scenario(BaseModel):
    """A complete AMR sensor snapshot — the user-side input of one SFT example."""
    yolo:      YOLODetection | None = None
    lidar:     LiDARReading
    telemetry: FleetTelemetry
