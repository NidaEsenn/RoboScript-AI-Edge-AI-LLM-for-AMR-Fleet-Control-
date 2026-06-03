"""No-key offline dataset builder.

Takes hand-authored seed examples (data/seed/examples.json), validates each with
the RoboResponse Pydantic model, applies a deterministic rule-check in place of the
LLM-as-Judge, builds DPO pairs, and writes data/processed/{sft_train,dpo_pairs}.json.

    python -m roboscript.data.offline

Requires no API key.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from roboscript.constants import FAULT_CATEGORIES
from roboscript.schemas.payload import RoboResponse

SEED_PATH = Path("data/seed/examples.json")
OUT_DIR = Path("data/processed")
PASS_SCORE = 4.5      # deterministic stand-in for the LLM judge average
CHOSEN_THRESHOLD = 4.0

CATS = {c["id"]: c for c in FAULT_CATEGORIES}

# payload_type -> the ros2 shape it must carry
_ROS2_SHAPE = {
    "CMD_VEL_DIRECT": "cmd_vel",
    "NAV2_ACTION": "nav2",
    "FAULT_REPORT_ONLY": "none",
}


def make_rejected(cat: dict) -> dict:
    """Synthesize a physically unsafe 'rejected' RoboResponse for DPO.

    Drives the robot forward (linear.x > 0) regardless of the fault — the
    canonical dangerous answer, especially for person-detection / ESTOP.
    """
    return {
        "decision": "CONTINUE",
        "reasoning": "Proceeding at current speed despite the active fault signal.",
        "payload_type": "CMD_VEL_DIRECT",
        "ros2": {
            "topic": "/cmd_vel",
            "msg_type": "geometry_msgs/msg/Twist",
            "cmd_vel": {
                "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        "aws_telemetry": {"incident_severity": "LEVEL_0", "request_human_audit": False},
    }


def rule_check(cat: dict, r: dict) -> tuple[bool, str]:
    """Deterministic judge: returns (passed, reason-if-failed)."""
    if r["decision"] not in cat["decisions"]:
        return False, f"decision {r['decision']} not in {cat['decisions']}"
    if r["payload_type"] not in cat["payloads"]:
        return False, f"payload_type {r['payload_type']} not in {cat['payloads']}"

    # ros2 field must match the payload_type shape
    shape = _ROS2_SHAPE[r["payload_type"]]
    ros2 = r["ros2"]
    if shape == "none" and ros2 is not None:
        return False, "FAULT_REPORT_ONLY must have ros2=null"
    if shape == "cmd_vel" and (ros2 is None or "cmd_vel" not in ros2):
        return False, "CMD_VEL_DIRECT must carry a cmd_vel payload"
    if shape == "nav2" and (ros2 is None or "goal" not in ros2):
        return False, "NAV2_ACTION must carry a navigate_to_pose goal"

    # Mandatory-halt categories (person/forklift, ESTOP) must truly stop the robot.
    if cat["mandatory_halt"]:
        if r["decision"] != "EMERGENCY_HALT" or r["payload_type"] != "CMD_VEL_DIRECT":
            return False, "mandatory-halt category must EMERGENCY_HALT via CMD_VEL_DIRECT"
        lin = ros2["cmd_vel"]["linear"]
        if lin["x"] != 0.0 or ros2["cmd_vel"]["angular"]["z"] != 0.0:
            return False, "mandatory-halt must command zero velocity"

    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=str(SEED_PATH))
    args = ap.parse_args()

    seed = json.loads(Path(args.seed).read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_rows: list[dict] = []
    dpo_rows: list[dict] = []
    kept_by_cat: dict[int, int] = {c["id"]: 0 for c in FAULT_CATEGORIES}
    discarded = 0

    for ex in seed:
        cat = CATS[ex["category_id"]]
        resp = ex["response"]

        # 1. Pydantic validation — discard invalid, never patch.
        try:
            RoboResponse.model_validate(resp)
        except ValidationError as e:
            print(f"  [validate] cat {cat['id']} invalid: {e.error_count()} errs")
            discarded += 1
            continue

        # 2. Deterministic rule-check (judge stand-in).
        ok, why = rule_check(cat, resp)
        if not ok:
            print(f"  [rule] cat {cat['id']} rejected: {why}")
            discarded += 1
            continue

        sft_rows.append({
            "category_id": cat["id"],
            "scenario": ex["scenario"],
            "response": resp,
            "judge_score": PASS_SCORE,
        })
        kept_by_cat[cat["id"]] += 1

        # 3. DPO pair: safe chosen vs unsafe rejected.
        if PASS_SCORE >= CHOSEN_THRESHOLD:
            dpo_rows.append({
                "prompt": json.dumps(ex["scenario"]),
                "chosen": json.dumps(resp),
                "rejected": json.dumps(make_rejected(cat)),
            })

    (OUT_DIR / "sft_train.json").write_text(json.dumps(sft_rows, indent=2))
    (OUT_DIR / "dpo_pairs.json").write_text(json.dumps(dpo_rows, indent=2))

    print("\nper-category kept:", {k: v for k, v in kept_by_cat.items()})
    print(f"discarded: {discarded}")
    print(f"Saved {len(sft_rows)} SFT rows -> {OUT_DIR/'sft_train.json'}")
    print(f"Saved {len(dpo_rows)} DPO pairs -> {OUT_DIR/'dpo_pairs.json'}")


if __name__ == "__main__":
    main()
