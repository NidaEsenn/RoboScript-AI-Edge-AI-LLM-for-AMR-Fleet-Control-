"""Procedural (zero-API) dataset generator — scales the 36 hand seeds to thousands.

For each of the 12 fault categories we sample a randomized but physically plausible
telemetry snapshot that exhibits the category's key signal, label it with the
deterministic reference policy in ``roboscript.safety`` and render a varied,
number-grounded ``reasoning`` string. A fraction of scenarios are *compound*
(two faults at once, e.g. low battery + person in path) — the label then follows
the safety priority order, which is exactly what a model has to learn.

Outputs (same format as offline.py / synthetic.py, so pipeline.py consumes them):
    data/processed/sft_train.json   validated {category_id, scenario, response, ...}
    data/processed/dpo_pairs.json   {prompt, chosen, rejected, failure_mode, ...}

    python -m roboscript.data.procedural --per-category 250 --include-seeds

Every example passes RoboResponse validation and ``safety.check`` (exact match +
safe) before it is written; anything else is discarded, never patched.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from roboscript.constants import FAULT_CATEGORIES
from roboscript.data.negatives import make_hard_negative
from roboscript.safety import CATS, check, classify, reference_action
from roboscript.schemas.payload import RoboResponse

OUT_DIR = Path("data/processed")
SEED_PATH = Path("data/seed/examples.json")

STATIC_CLASSES = ["pallet", "box", "tote", "cart", "damaged_pallet", "shrink_wrap"]
ZONES = list("ABCDEFGH")


# ------------------------------------------------------------------ scenario samplers
def _r(rng: random.Random, lo: float, hi: float, nd: int = 2) -> float:
    return round(rng.uniform(lo, hi), nd)


def _bbox(rng: random.Random, tall: bool = False) -> list[float]:
    w = rng.randint(60, 140)
    h = rng.randint(140, 260) if tall else rng.randint(80, 160)
    return [rng.randint(120, 480), rng.randint(140, 300), w, h]


def _base(rng: random.Random) -> dict:
    return {"yolo": None,
            "lidar": {"min_distance_meters": _r(rng, 1.5, 8.0, 1), "status": "CLEAR"},
            "telemetry": {"current_speed_mps": _r(rng, 0.5, 1.5, 1),
                          "battery_level_pct": rng.randint(30, 95),
                          "active_fault_codes": []}}


def _apply(cid: int, s: dict, rng: random.Random) -> None:
    """Mutate scenario ``s`` in place so that it exhibits category ``cid``."""
    t, lid = s["telemetry"], s["lidar"]
    codes = t["active_fault_codes"]
    if cid == 1:
        s["yolo"] = {"cls": rng.choice(STATIC_CLASSES), "bbox_xywh": _bbox(rng),
                     "confidence": _r(rng, 0.65, 0.97)}
        lid.update(min_distance_meters=_r(rng, 0.25, 0.59), status="PATH_OBSTRUCTED")
    elif cid == 2:
        s["yolo"] = {"cls": rng.choice(["person", "person", "forklift"]),
                     "bbox_xywh": _bbox(rng, tall=True), "confidence": _r(rng, 0.81, 0.99)}
        lid.update(min_distance_meters=_r(rng, 0.3, 3.0), status="PATH_OBSTRUCTED")
    elif cid == 3:
        lid.update(min_distance_meters=0.0, status="SENSOR_FAULT")
        codes.append(rng.choice(["ERR_LIDAR_NODATA", "ERR_LIDAR_NAN"]))
        if rng.random() < 0.3:
            t["current_speed_mps"] = 0.0
    elif cid == 4:
        t["battery_level_pct"] = rng.randint(3, 14)
        codes.append("ERR_BATTERY_CRITICAL")
    elif cid == 5:
        t["battery_level_pct"] = rng.randint(15, 25)
        codes.append("ERR_BATTERY_LOW")
    elif cid == 6:
        codes += [rng.choice(["ERR_MOTOR_OVERHEAT_L", "ERR_MOTOR_OVERHEAT_R"]),
                  f"temp_c={rng.randint(86, 105)}"]
    elif cid == 7:
        codes.append("ERR_WHEEL_SLIP_LOW")
    elif cid == 8:
        codes += ["ERR_WHEEL_SLIP_HIGH", "floor=WET"]
    elif cid == 9:
        codes += ["ERR_ENCODER_DRIFT", f"localization={_r(rng, 0.1, 0.39)}"]
        if rng.random() < 0.3:
            t["current_speed_mps"] = 0.0
    elif cid == 10:
        codes.append("ERR_ESTOP_TRIGGERED")
        if rng.random() < 0.2:
            t["current_speed_mps"] = 0.0
    elif cid == 11:
        codes += ["ERR_COLLISION_DETECTED", f"imu_spike={_r(rng, 2.5, 6.0, 1)}g"]
        lid.update(min_distance_meters=_r(rng, 0.1, 0.6), status="PATH_OBSTRUCTED")
        t["current_speed_mps"] = rng.choice([0.0, 0.0, _r(rng, 0.1, 0.4, 1)])
    elif cid == 12:
        s["yolo"] = {"cls": "unknown", "bbox_xywh": _bbox(rng), "confidence": _r(rng, 0.05, 0.29)}
        codes.append("ERR_CAMERA_DEGRADED")


def sample_scenario(cid: int, rng: random.Random, compound_p: float = 0.15) -> dict:
    s = _base(rng)
    _apply(cid, s, rng)
    if rng.random() < compound_p:
        # Add a second fault that does not structurally conflict with the first.
        compatible = {1: [4, 5, 7, 6], 2: [4, 5, 6, 7, 9], 3: [4, 5, 6], 4: [6, 7, 9, 12, 2, 10],
                      5: [1, 2, 6, 7, 8, 9, 10, 12], 6: [4, 5, 7, 1], 7: [4, 5, 6, 10],
                      8: [4, 5, 6], 9: [4, 5, 7], 10: [4, 5, 6, 7], 11: [4, 5], 12: [4, 5, 7, 10]}
        _apply(rng.choice(compatible[cid]), s, rng)
    return s


# ------------------------------------------------------------------ response renderer
def _zone(rng: random.Random, kind: str) -> str:
    return f"{kind}_{rng.choice(ZONES)}{rng.randint(1, 9)}"


def _reason(cid: int, s: dict, decision: str, rng: random.Random) -> str:
    t, lid, y = s["telemetry"], s["lidar"], s.get("yolo") or {}
    v, b, d = t["current_speed_mps"], t["battery_level_pct"], lid["min_distance_meters"]
    codes = t["active_fault_codes"]
    kv = {c.split("=")[0]: c.split("=")[1] for c in codes if "=" in c}
    extra = ""
    others = [c for c in codes if c.startswith("ERR_") and c not in
              ("ERR_ESTOP_TRIGGERED",)]
    if cid in (2, 10) and others:
        extra = f" {others[0]} deferred until area is safe."
    T = {
        10: [f"E-stop triggered at {v} m/s. Mandatory zero-velocity halt, operator reset required.",
             f"ERR_ESTOP_TRIGGERED active. Halting immediately; no auto-recovery."],
        2: [f"{y.get('cls', 'person').capitalize()} detected at {y.get('confidence')} confidence, {d}m ahead. Mandatory halt.",
            f"Human-class object ({y.get('cls')}) in path at {d}m. Emergency halt, no auto-recovery."],
        11: [f"Collision with {kv.get('imu_spike', 'high')} IMU spike. Halting and requesting damage audit.",
             f"Impact detected ({kv.get('imu_spike', '')}). Robot must not move until inspected."]
        if decision == "EMERGENCY_HALT" else
            [f"Low-energy contact ({kv.get('imu_spike', '')}) while stationary. Logging for damage assessment.",
             f"Collision flagged at rest; holding position and reporting for inspection."],
        3: [f"LiDAR fault ({codes[0] if codes else 'no data'}) at {v} m/s. Obstacle data untrusted, halting.",
            f"Primary LiDAR returning {lid['min_distance_meters']}m readings. Stopping and flagging maintenance."]
        if v > 0 else
            [f"LiDAR fault while stationary. Holding position and reporting for diagnostics.",
             f"Sensor fault at rest; logging fault, robot remains stopped."],
        8: [f"High wheel slip on wet floor at {v} m/s. Halting to avoid uncontrolled skid.",
            f"ERR_WHEEL_SLIP_HIGH with floor=WET. Emergency halt until floor is cleared."],
        6: [f"Motor at {kv.get('temp_c', '?')}C exceeds limit at {v} m/s. Halting to prevent damage.",
            f"Overheat ({kv.get('temp_c', '?')}C). Emergency halt and maintenance audit."]
        if decision == "EMERGENCY_HALT" else
            [f"Motor at {kv.get('temp_c', '?')}C, moderate speed. Logging and entering thermal cooldown.",
             f"Mild overheat ({kv.get('temp_c', '?')}C); cooldown cycle requested, fault logged."],
        9: [f"Encoder drift, localization {kv.get('localization', '?')}. Halting to relocalize.",
            f"Localization confidence {kv.get('localization', '?')} below 0.4. Stopping and relocalizing."]
        if v > 0 else
            [f"Encoder drift at rest (loc {kv.get('localization', '?')}). Relocalizing before motion.",
             f"Localization {kv.get('localization', '?')} while stationary; reporting and relocalizing."],
        1: [f"{y.get('cls', 'object')} at {d}m inside stopping distance at {v} m/s. Halting.",
            f"Static obstruction ({y.get('cls')}) {d}m ahead. Emergency halt before replanning."]
        if decision == "EMERGENCY_HALT" else
            [f"{y.get('cls', 'object').capitalize()} blocking lane at {d}m. Rerouting via Nav2 replanner.",
             f"Static {y.get('cls')} at {d}m; alternate path available, delegating to Nav2."],
        4: [f"Battery critical at {b}%. Routing to nearest charging dock via Nav2.",
            f"Battery {b}% below safe margin. Abandoning task, navigating to dock."],
        12: [f"Camera degraded (conf {y.get('confidence')}). Reducing speed, LiDAR primary.",
             f"YOLO confidence {y.get('confidence')} unreliable at {v} m/s. Speed-limited navigation."]
        if decision == "REDUCE_SPEED" else
            [f"Camera degraded (conf {y.get('confidence')}). Switching to LiDAR-only mode, fault logged.",
             f"Vision unreliable; continuing in LiDAR-only mode and logging camera fault."],
        7: [f"Low wheel slip at {v} m/s. Reducing speed via Nav2 speed-limit tree.",
            f"Minor slip detected at {v} m/s; speed-limited navigation engaged."]
        if decision == "REDUCE_SPEED" else
            [f"Minor wheel slip at {v} m/s, no obstruction. Continuing, fault logged.",
             f"Low slip at low speed; safe to continue, logging for maintenance."],
        5: [f"Battery low at {b}%. Continuing task, recharge queued, fault logged.",
            f"Battery {b}% in low band. Logging fault and scheduling recharge."],
    }
    text = rng.choice(T[cid]) + extra
    return text[:200] if len(text) >= 10 else text + " Safe action."


def render_response(s: dict, rng: random.Random) -> dict:
    cid = classify(s)
    a = reference_action(s)
    assert cid is not None and a is not None
    if a.payload_type == "CMD_VEL_DIRECT":
        ros2 = {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist",
                "cmd_vel": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}}
    elif a.payload_type == "NAV2_ACTION":
        dock = a.decision == "NAVIGATE_TO_CHARGING_DOCK"
        ros2 = {"action_server": "/navigate_to_pose",
                "action_type": "nav2_msgs/action/NavigateToPose",
                "goal": {"pose": {"header": {"frame_id": "map"},
                                  "position": {"x": _r(rng, 0.5, 6.0, 1) if dock else _r(rng, 2, 30, 1),
                                               "y": _r(rng, 0.5, 8.0, 1) if dock else _r(rng, 2, 30, 1),
                                               "z": 0.0},
                                  "orientation": {"w": 1.0}},
                         "behavior_tree": a.behavior_tree}}
    else:
        ros2 = None
    tel: dict = {"incident_severity": a.severity, "request_human_audit": a.audit}
    zone_kind = {1: "TEMPORARY_BLOCKAGE_ZONE" if a.decision == "EMERGENCY_HALT" else "ROUTE_MODIFIED",
                 4: "RETURNING_TO_DOCK", 8: "WET_FLOOR_ZONE", 3: "SENSOR_FAULT_HOLD"}.get(cid)
    if zone_kind and rng.random() < 0.7:
        tel["zone_update"] = zone_kind if cid == 3 else _zone(rng, zone_kind)
    return {"decision": a.decision, "reasoning": _reason(cid, s, a.decision, rng),
            "payload_type": a.payload_type, "ros2": ros2, "aws_telemetry": tel}


# ------------------------------------------------------------------------------ main
def generate(per_category: int, seed: int = 0, compound_p: float = 0.15,
             include_seeds: bool = True) -> tuple[list[dict], list[dict], Counter]:
    rng = random.Random(seed)
    rows: list[dict] = []
    if include_seeds and SEED_PATH.exists():
        for ex in json.loads(SEED_PATH.read_text()):
            rows.append({"category_id": ex["category_id"], "scenario": ex["scenario"],
                         "response": ex["response"], "source": "seed"})
    discarded = Counter()
    for cat in FAULT_CATEGORIES:
        made = 0
        while made < per_category:
            s = sample_scenario(cat["id"], rng, compound_p)
            r = render_response(s, rng)
            rows.append({"category_id": classify(s), "scenario": s, "response": r,
                         "source": "procedural", "intended_category": cat["id"]})
            made += 1

    kept, dpo = [], []
    for row in rows:
        try:
            RoboResponse.model_validate(row["response"])
        except Exception:  # noqa: BLE001
            discarded["schema"] += 1
            continue
        v = check(row["scenario"], row["response"])
        if not (v.exact_match and v.safe) or v.category != row["category_id"]:
            discarded["rule"] += 1
            continue
        row["judge_score"] = 5.0  # deterministic judge: passes every rule
        kept.append(row)
        rej, mode = make_hard_negative(row["scenario"], row["response"], rng)
        dpo.append({"prompt": json.dumps(row["scenario"]), "chosen": json.dumps(row["response"]),
                    "rejected": json.dumps(rej), "failure_mode": mode,
                    "category_id": row["category_id"]})
    return kept, dpo, discarded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compound-p", type=float, default=0.15)
    ap.add_argument("--no-seeds", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    kept, dpo, discarded = generate(args.per_category, args.seed, args.compound_p,
                                    not args.no_seeds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft_train.json").write_text(json.dumps(kept, indent=1))
    (out / "dpo_pairs.json").write_text(json.dumps(dpo, indent=1))
    by_cat = Counter(r["category_id"] for r in kept)
    print("per-category (after priority relabel):", dict(sorted(by_cat.items())))
    print("DPO failure modes:", dict(Counter(d["failure_mode"] for d in dpo)))
    print(f"discarded: {dict(discarded)}")
    print(f"Saved {len(kept)} SFT rows and {len(dpo)} DPO pairs -> {out}")


if __name__ == "__main__":
    main()
