# RoboScript AI — Edge-AI LLM for AMR Fleet Control

A domain-specific LLM pipeline that turns Amazon-style **Autonomous Mobile Robot (AMR)** sensor telemetry into a single, schema-validated **ROS 2 action payload** — with safety as a hard constraint, not an afterthought.

Given a fault scenario (LiDAR distance, YOLO detections, fleet telemetry, fault codes), the model decides the safest action and emits **valid JSON** that a ROS 2 bridge can dispatch directly — halt, reroute, return-to-dock, or log-only.

> Full design and component specifications in [`RoboScript_PRD.md`](RoboScript_PRD.md).

---

## Why this project

Warehouse AMRs must respond to faults (person in path, e-stop, motor overheat, low battery) in **milliseconds** and **never** make an unsafe move. A general LLM produces free-form text; a fleet controller needs **deterministic, schema-valid, safety-first** output. RoboScript fine-tunes a small model to do exactly that, then aligns it for safety and distills it for edge deployment.

---

## Safety invariants (non-negotiable)

These hold throughout the pipeline — enforced by Pydantic validation and a deterministic rule-check:

- A payload is **exactly one** of `CMD_VEL_DIRECT` / `NAV2_ACTION` / `FAULT_REPORT_ONLY` — `cmd_vel` and `NavigateToPose` never co-occur.
- `EMERGENCY_HALT` **always** commands zero velocity (`linear.x = 0.0`, `angular.z = 0.0`).
- Any output that fails schema validation is **discarded / replaced by an emergency-halt fallback** — never forwarded to ROS 2.
- The two life-safety categories — **person/forklift in path** and **e-stop** — must always `EMERGENCY_HALT`.

---

## Fault taxonomy (12 categories)

| # | Category | Key signal | Payload |
|---|----------|-----------|---------|
| 1 | Path obstruction (static) | LiDAR < 0.6 m, pallet/box | A or B |
| 2 | Path obstruction (person/forklift) | YOLO person/forklift > 0.8 | **A only (mandatory halt)** |
| 3 | LiDAR sensor fault | readings 0.0 / NaN | A + C |
| 4 | Battery critical (< 15%) | battery_pct < 15 | B |
| 5 | Battery low (15–25%) | battery_pct 15–25 | C or B |
| 6 | Motor overheating | temp_c > 85 | A + C |
| 7 | Wheel slip (low) | no obstruction | B slow / C |
| 8 | Wheel slip (high) | wet floor | A |
| 9 | Odometry / encoder fault | localization < 0.4 | A + C |
| 10 | Emergency stop | `ERR_ESTOP_TRIGGERED` | **A only (mandatory halt)** |
| 11 | Collision detected | IMU spike | A + C |
| 12 | YOLO / camera failure | confidence < 0.3 | B slow + C |

Payloads: **A** = `CMD_VEL_DIRECT` (direct halt), **B** = `NAV2_ACTION` (reroute / dock), **C** = `FAULT_REPORT_ONLY` (log).

---

## Pipeline overview

```
 telemetry scenarios
        │
        ▼
 [ Session A ] synthetic generation ──► validate (Pydantic) ──► LLM-as-judge ──► DPO pairs
        │
        ▼
 [ Session B ] chat format ──► loss masking ──► MinHash dedup ──► 80/10/10 split
        │
        ▼
   QLoRA SFT ──► DPO safety alignment ──► eval ──► distill (Phi-3) ──► vLLM/FastAPI ──► ROS 2 bridge
```

---

## Components

### Synthetic data generation
- **`roboscript/data/synthetic.py`** — multi-provider generation (**Gemini / Groq / OpenAI**, swappable via one env var; all driven through the OpenAI-compatible API). Every output is validated against the `RoboResponse` Pydantic schema (**invalid examples discarded, never patched**), quality-filtered with an **LLM-as-judge**, and paired into **DPO** (safe `chosen` vs. unsafe `rejected`) examples. Includes per-day/per-minute rate-limit handling and `max_tokens` budgeting.
- **`roboscript/data/offline.py`** — zero-API dataset builder from hand-authored seeds, using a **deterministic rule-check** in place of the LLM judge. Produces a balanced 12-category set with no external dependencies.

### Dataset preparation
- **`roboscript/data/pipeline.py`** — converts examples to **LLaMA-3 chat format**, applies **loss masking** (`labels = -100` on system/user tokens so only the assistant response is scored), **MinHash deduplication** (threshold 0.9), re-validates with Pydantic, and produces a reproducible **80/10/10** train/val/test split.

### Schemas & constants
- **`roboscript/schemas/`** — `RoboResponse`, ROS 2 payload models, telemetry models, and the `EMERGENCY_HALT_FALLBACK` constant.
- **`roboscript/constants.py`** — `SYSTEM_PROMPT`, the 12-category fault taxonomy, and the LLM-as-judge rubric.

## Quickstart

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. configure (copy and fill in your keys)
cp .env.example .env          # set PROVIDER + the matching API key

# 3a. generate data with no API key (hand-authored seeds + rule-check)
python -m roboscript.data.offline

# 3b. OR generate via an LLM provider (Gemini / Groq / OpenAI)
python -m roboscript.data.synthetic --per-category 5 --delay 7

# 4. prepare the training dataset (chat format + loss masking + split)
python -m roboscript.data.pipeline
```

Outputs land in `data/processed/` as `sft_train.json`, `dpo_pairs.json`, and `train/val/test.json`.

### Configuration (`.env`)

| Variable | Purpose |
|----------|---------|
| `PROVIDER` | `gemini` \| `groq` \| `openai` — selects the generation backend |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY` | key for the chosen provider |
| `HF_TOKEN` | HuggingFace token for the Llama-3.1 tokenizer (loss masking step) |

---

## Project structure

```
roboscript/
├── constants.py            # SYSTEM_PROMPT, 12-category taxonomy, judge rubric
├── schemas/
│   ├── payload.py          # RoboResponse + ROS 2 payloads + EMERGENCY_HALT_FALLBACK
│   └── telemetry.py        # YOLO / LiDAR / fleet telemetry models
└── data/
    ├── synthetic.py        # multi-provider generation + validate + judge + DPO
    ├── offline.py          # no-API builder (deterministic rule-check)
    └── pipeline.py         # chat format + loss masking + dedup + split
data/seed/examples.json     # hand-authored seed scenarios
RoboScript_PRD.md           # full product/architecture spec
```

---

## Tech stack

**Python · Pydantic · HuggingFace Transformers · datasketch (MinHash) · OpenAI-compatible LLM APIs (Gemini / Groq / OpenAI)**
Training & serving: Unsloth · TRL (SFT/DPO) · vLLM · FastAPI · ROS 2 (rclpy) · Docker · AWS SageMaker

---

## License

Educational / portfolio project. See `RoboScript_PRD.md` for the full specification.
