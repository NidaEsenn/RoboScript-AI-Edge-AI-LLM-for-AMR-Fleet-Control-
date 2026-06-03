"""Session A — synthetic data pipeline (PRD Section 6.1).

Generates scenario -> RoboResponse SFT examples across the 12 fault categories,
validates each with Pydantic (invalid discarded, never patched), filters with an
LLM-as-Judge (keep average >= 3.5), and builds DPO preference pairs.

Provider is selected via the PROVIDER env var (default: gemini); see PROVIDERS
below for the per-provider base_url, key env var, and default model. All three
expose an OpenAI-compatible API, so the OpenAI SDK drives every one of them.

Run a smoke test (set GEMINI_API_KEY in .env first):
    python -m roboscript.data.synthetic --per-category 2

Full run (3,000 SFT):
    python -m roboscript.data.synthetic --per-category 250
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import ValidationError

load_dotenv()  # read provider keys (GEMINI_API_KEY / GROQ_API_KEY / OPENAI_API_KEY)

from roboscript.constants import SYSTEM_PROMPT, FAULT_CATEGORIES, EVAL_JUDGE_PROMPT
from roboscript.data.offline import make_rejected
from roboscript.schemas.payload import RoboResponse

OUT_DIR = Path("data/processed")
JUDGE_THRESHOLD = 3.5
CHOSEN_THRESHOLD = 4.0

# OpenAI-compatible providers. base_url=None uses OpenAI's own endpoint.
PROVIDERS = {
    "openai": {"base_url": None,
               "key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
    "groq":   {"base_url": "https://api.groq.com/openai/v1",
               "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
}
PROVIDER = os.getenv("PROVIDER", "gemini").lower()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily build the OpenAI-compatible client for the selected provider."""
    global _client
    if _client is None:
        cfg = PROVIDERS[PROVIDER]
        key = os.getenv(cfg["key_env"])
        if not key:
            raise SystemExit(f"PROVIDER={PROVIDER} requires {cfg['key_env']} in .env")
        _client = OpenAI(api_key=key, base_url=cfg["base_url"])
    return _client


class DailyQuotaExhausted(Exception):
    """Raised when the provider's per-day quota is gone — no point retrying today."""


def chat_json(model: str, messages: list[dict], max_tokens: int = 800) -> str:
    """JSON-mode chat call with bounded backoff on per-minute 429s.

    DECISION: treat per-day and per-minute 429s differently.
    WHY: a per-MINUTE limit clears in seconds, so retrying is sensible; a per-DAY
         limit will not clear today, so retrying just burns more of the (already
         exhausted) budget — exactly what stalled our earlier run for 25 minutes.
    TRADEOFF: we parse the error text to tell them apart, which is provider-specific
              and a little brittle; the win is we fail fast instead of thrashing.
    """
    for attempt in range(3):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
                # DECISION: cap completion length.
                # WHY: rate limiters reserve prompt+max_tokens against the per-minute
                #      TOKEN budget. Unbounded max_tokens reserves a huge allowance, so a
                #      single call can drain the 12k/min bucket and 429 instantly.
                # TRADEOFF: a response longer than this is truncated (then fails JSON
                #           parsing and is discarded) — fine, our outputs are well under.
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            msg = str(e)
            # Per-day exhaustion (Gemini: "PerDay"; Groq: "tokens per day (TPD)").
            if "PerDay" in msg or "per_day" in msg or "per day" in msg or "(TPD)" in msg or "(RPD)" in msg:
                raise DailyQuotaExhausted(msg[:250])
            wait = 10 * (attempt + 1)  # 10s, 20s — short, since this is a speed limit
            print(f"  [rate-limit] per-minute; sleeping {wait}s (attempt {attempt + 1}/3)")
            time.sleep(wait)
    raise RuntimeError("giving up after repeated per-minute rate-limit errors")


def build_generation_prompt(cat: dict) -> str:
    """Instruct the model to emit one {scenario, response} example for a category."""
    return f"""Generate one realistic Amazon AMR fault-response training example for this category.

Fault category #{cat['id']}: {cat['name']}
Key sensor signal: {cat['signal']}
Acceptable decisions: {', '.join(cat['decisions'])}
Acceptable payload_type values: {', '.join(cat['payloads'])}

Output ONLY a JSON object with exactly two keys: "scenario" and "response".

"scenario": {{"yolo": {{"cls": str, "bbox_xywh": [x,y,w,h], "confidence": 0.0-1.0}} or null,
"lidar": {{"min_distance_meters": float, "status": "CLEAR"|"PATH_OBSTRUCTED"|"SENSOR_FAULT"}},
"telemetry": {{"current_speed_mps": float, "battery_level_pct": float, "active_fault_codes": [str]}}}}
The scenario MUST exhibit the key sensor signal above.

"response": {{"decision": str (one of the acceptable decisions), "reasoning": str (10-200 chars),
"payload_type": str, "ros2": <see below>, "aws_telemetry": {{"incident_severity":
"LEVEL_0".."LEVEL_3", "zone_update": str or omit, "request_human_audit": bool}}}}

The "ros2" field MUST match payload_type EXACTLY, copying these literal keys/values:
- CMD_VEL_DIRECT: {{"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist", "cmd_vel":
  {{"linear": {{"x": 0.0, "y": 0.0, "z": 0.0}}, "angular": {{"x": 0.0, "y": 0.0, "z": 0.0}}}}}}
  (for any halt, all linear/angular values MUST be 0.0)
- NAV2_ACTION: {{"action_server": "/navigate_to_pose", "action_type": "nav2_msgs/action/NavigateToPose",
  "goal": {{"pose": {{"header": {{"frame_id": "map"}}, "position": {{"x": float, "y": float, "z": 0.0}},
  "orientation": {{"w": 1.0}}}}, "behavior_tree": "navigate_w_replanning_and_recovery.xml"}}}}
- FAULT_REPORT_ONLY: null

The response MUST be the safest valid action for this category."""


def generate_example(cat: dict, model: str) -> dict | None:
    """One generation call -> raw {scenario, response} dict (or None on failure)."""
    try:
        return json.loads(chat_json(model, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_generation_prompt(cat)},
        ]))
    except DailyQuotaExhausted:
        raise  # let main() stop the whole run — retrying today is pointless
    except Exception as e:  # network / JSON / API error -> skip this example
        print(f"  [gen] category {cat['id']} failed: {e}")
        return None


def judge(scenario: dict, response: dict, model: str) -> float:
    """LLM-as-Judge -> average score 1-5 (0.0 on failure so it is discarded)."""
    prompt = EVAL_JUDGE_PROMPT.format(
        scenario=json.dumps(scenario), response=json.dumps(response)
    )
    try:
        return float(json.loads(chat_json(model, [{"role": "user", "content": prompt}], max_tokens=100))["average"])
    except Exception as e:
        print(f"  [judge] failed: {e}")
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=2,
                    help="examples to generate per fault category (2 = smoke test)")
    ap.add_argument("--model", default=None,
                    help="override model; defaults to the provider's model")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to pause between examples to stay under rate limits "
                         "(e.g. 7 for Groq free tier when scaling up)")
    args = ap.parse_args()

    model = args.model or PROVIDERS[PROVIDER]["model"]
    print(f"provider={PROVIDER} model={model} delay={args.delay}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sft_rows: list[dict] = []
    dpo_rows: list[dict] = []

    for cat in FAULT_CATEGORIES:
        kept = 0
        for _ in range(args.per_category):
            # DECISION: pace requests instead of bursting then backing off.
            # WHY: token-per-minute limits stay saturated if we fire as fast as we can;
            #      a steady trickle keeps us under the ceiling so every category completes.
            # TRADEOFF: slower wall-clock time, but a balanced, complete dataset.
            time.sleep(args.delay)
            raw = generate_example(cat, model)
            if raw is None or "scenario" not in raw or "response" not in raw:
                continue

            # Validate the response with Pydantic — discard invalid, never patch.
            try:
                RoboResponse.model_validate(raw["response"])
            except ValidationError as e:
                print(f"  [validate] category {cat['id']} invalid: {e.error_count()} errs")
                continue

            score = judge(raw["scenario"], raw["response"], model)
            if score < JUDGE_THRESHOLD:
                continue

            sft_rows.append({
                "category_id": cat["id"],
                "scenario": raw["scenario"],
                "response": raw["response"],
                "judge_score": score,
            })
            kept += 1

            # DPO pair: keep high-scoring examples as 'chosen' vs an unsafe 'rejected'.
            if score >= CHOSEN_THRESHOLD:
                dpo_rows.append({
                    "prompt": json.dumps(raw["scenario"]),
                    "chosen": json.dumps(raw["response"]),
                    "rejected": json.dumps(make_rejected(cat)),
                })
        print(f"category {cat['id']:>2} ({cat['name']}): kept {kept}/{args.per_category}")

    if not sft_rows:
        raise SystemExit("No examples survived — refusing to overwrite existing output.")

    (OUT_DIR / "sft_train.json").write_text(json.dumps(sft_rows, indent=2))
    (OUT_DIR / "dpo_pairs.json").write_text(json.dumps(dpo_rows, indent=2))
    print(f"\nSaved {len(sft_rows)} SFT rows -> {OUT_DIR/'sft_train.json'}")
    print(f"Saved {len(dpo_rows)} DPO pairs -> {OUT_DIR/'dpo_pairs.json'}")


if __name__ == "__main__":
    try:
        main()
    except DailyQuotaExhausted as e:
        raise SystemExit(f"\nDaily quota exhausted — stopping early. Existing data kept.\n{e}")
