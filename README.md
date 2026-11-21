# RoboScript AI — Edge-AI LLM for AMR Fleet Control

A domain-specific LLM post-training + MLOps pipeline that turns Autonomous Mobile Robot
(AMR) sensor telemetry (LiDAR distance, YOLO detections, fleet telemetry, fault codes)
into a single, schema-validated **ROS 2 action payload** — halt, reroute, return-to-dock,
or log-only — with safety enforced by deterministic layers around the model, not only by
the model.

An LLM post-training and MLOps project.

```
 telemetry ─▶ synthetic / procedural data ─▶ validate ─▶ judge ─▶ SFT pairs + hard-negative DPO pairs
                                                                  │
                    chat format + loss mask + MinHash dedup + stratified split (leak-free DPO)
                                                                  │
               SFT (QLoRA on GPU / full FT on CPU) ─▶ on-policy negative mining ─▶ DPO (LoRA)
                                                                  │
                    logit / sequence-level distillation ─▶ edge model     evaluation report
                                                                  │
   FastAPI ─ deadline ─ Pydantic fallback ─ safety interlock ─▶ ROS 2 bridge ─▶ /cmd_vel · Nav2 · /fault_events
```

## Status

All seven PRD sessions (A–G) are implemented. The full pipeline runs end-to-end on a CPU
with **no API keys and no model downloads** (`local` profile). The `gpu` profile holds
the PRD configuration (Llama-3.1-8B QLoRA → Phi-3-mini) and uses the same code paths,
but it has not been run here because no GPU or Hub access was available.

| Stage | Module | Runs locally | Notes |
|---|---|---|---|
| A. Data generation | `data/synthetic.py`, `data/offline.py`, **`data/procedural.py`** | yes (procedural/offline) | LLM generation needs an API key |
| Hard negatives | **`data/negatives.py`**, **`data/onpolicy.py`** | yes | replaces the single fixed "rejected" answer |
| B. Dataset prep | `data/pipeline.py` | yes | model-agnostic; stratified; DPO split by prompt |
| B. SFT | **`training/sft_trainer.py`** | yes (tiny model) | `--profile gpu` = QLoRA r16 / NF4 |
| C. DPO | **`training/dpo_trainer.py`** | yes | fresh LoRA on the SFT model; AC-04 margin checks |
| D. Evaluation | **`eval/evaluator.py`**, **`eval/golden.py`** | yes | test split + 50-case safety set + compound stress set |
| E. Distillation | **`distillation/distill.py`** | yes | masked token-KL; falls back to sequence-level KD across tokenizers |
| E. Serving | **`serve/api.py`**, **`serve/backends.py`**, **`serve/vllm_server.py`** | yes (rules/local backend) | vLLM backend needs a GPU |
| F. ROS 2 bridge | **`ros2_bridge/`** | yes (in-process bus) | real `rclpy` transport in `docker/Dockerfile.ros2` |
| G. Deploy | **`deploy/`**, `docker/`, `docker-compose.yml` | dry-run only | needs AWS credentials |

## Safety architecture (defence in depth)

1. **Schema** (`schemas/payload.py`): a closed decision vocabulary; `payload_type` must
   match the `ros2` shape (cmd_vel and Nav2 can never co-occur); every halt decision on
   `/cmd_vel` must be exactly zero velocity.
2. **Deadline + fallback** (`serve/api.py`): timeout, transport error, unparsable or
   schema-invalid output → `EMERGENCY_HALT_FALLBACK`.
3. **Interlock** (`safety.py: guard`): a deterministic classifier with a fault priority
   order. It overrides any answer that does not stop the robot when a stop is required,
   and any direct `/cmd_vel` motion. It only ever makes an answer *more* conservative.
4. **Life-safety fast path**: on `/v1/emergency`, person/forklift-in-path and e-stop are
   answered by the interlock in about 1 ms. The LLM is not on the critical path for those.
5. **Bridge** (`ros2_bridge/bridge_node.py`): re-validates every payload. An invalid
   payload, an unreachable Nav2 action server or an API failure → zero Twist.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                                   # unit + integration tests
python -m roboscript.ros2_bridge.bridge_node --demo   # telemetry → API → bridge → mock ROS 2

bash scripts/run_local_pipeline.sh                    # full CPU pipeline (~1.5–2 h, 4 cores)
FAST=1 bash scripts/run_local_pipeline.sh             # smoke run (~10 min)

ROBOSCRIPT_BACKEND=rules uvicorn roboscript.serve.api:app --port 8000
curl -s localhost:8000/v1/emergency -H 'content-type: application/json' -d '{"yolo":null,
  "lidar":{"min_distance_meters":2.0,"status":"CLEAR"},
  "telemetry":{"current_speed_mps":1.0,"battery_level_pct":65,"active_fault_codes":["ERR_ESTOP_TRIGGERED"]}}'
```

### GPU profile (PRD configuration)

```bash
pip install bitsandbytes                       # + optionally unsloth, wandb
python -m roboscript.data.synthetic --per-category 250 --delay 7   # or data.procedural
python -m roboscript.data.pipeline
python -m roboscript.eval.golden
python -m roboscript.training.sft_trainer --profile gpu            # writes roboscript-sft-merged/
python -m roboscript.data.onpolicy --model outputs/roboscript-sft-merged
python -m roboscript.training.dpo_trainer --profile gpu --train-file data/processed/dpo_train_mixed.json
python -m roboscript.distillation.distill --profile gpu            # sequence-level KD (vocab mismatch)
python -m roboscript.eval.evaluator --profile gpu --models base=meta-llama/Llama-3.1-8B-Instruct \
       sft=outputs/roboscript-sft-merged sft+dpo=outputs/roboscript-dpo-merged edge-kd=outputs/roboscript-edge
python -m roboscript.serve.vllm_server                             # SFT-merged base + DPO adapter
ROBOSCRIPT_BACKEND=vllm uvicorn roboscript.serve.api:app
docker compose --profile gpu up
python -m roboscript.deploy.sagemaker_deploy --dry-run
```

## Results (local CPU profile)

Full tables: [`docs/eval_report.md`](docs/eval_report.md) /
[`docs/eval_report.json`](docs/eval_report.json). How to read them:
[`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md). The local models are 5.3M-parameter
(teacher) and 1.3M-parameter (edge student) Llama-architecture models trained from
scratch on 2,246 examples. They validate the pipeline and the relative effect of each
stage. They are **not** a claim about what Llama-3.1-8B would score.

Test split (n=281) / 50-case safety set / 120-case compound stress set, raw model output:

| model | JSON valid | ROS2 correct | safety pass (test / safety / stress) | guarded safety | p50 latency, batch 1, CPU |
|---|---|---|---|---|---|
| base (untrained) | 0% | 0% | 0 / 0 / 0% | 90–100% | 814 ms |
| SFT | 100% | 99.6% | 99.6 / 100 / 98.3% | 100% | 566 ms |
| SFT + DPO (RPO loss) | 97.9% | 98.6% | 98.2 / 100 / 91.7% | 100% | 510 ms |
| SFT + DPO (sigmoid only, lr 1e-3), ablation | 18.1% | 41.3% | 43.8 / 30 / 56.7% | 97.2% | 528 ms |
| edge student, logit KD (1.3M) | 99.6% | 100% | 99.6 / 100 / 99.2% | 100% | 310 ms |
| edge student, no KD (1.3M) | 100% | 100% | 100 / 100 / 100% | 100% | 304 ms |

What the numbers say:

* Every PRD numeric gate passes for SFT+DPO on the local profile (JSON ≥ 92%, ROS2
  correctness ≥ 80%, safety gate ≥ 95%, judge ≥ 4.3, edge within 5 pt of the teacher).
  AC-05's "SFT+DPO best on every metric" does **not** hold: SFT is already at the ceiling
  of this rule-derived task, so DPO has little to fix. On compound faults DPO is slightly
  *worse* (more invalid JSON, which the fallback turns into halts).
* Plain sigmoid DPO at a high learning rate collapsed the model (likelihood
  displacement: chosen log-prob −6.7 → −25). Adding the NLL anchor (RPO) fixed it
  (chosen log-prob stays near −5). RPO is now the default.
* With the guard and fallback in place, every configuration that produces output
  forwards **100% safe** commands on the safety and stress sets. The lowest guarded
  number (90% on the untrained base) comes from fallback halts during battery-critical
  cases, which the checker scores as "stranding". This is the conservative failure
  direction.
* On a task this narrow, KD did not beat plain SFT of the small student. Both match
  the teacher at about 1/4 the parameters and roughly half the latency.

## Fixes made to the original design

| Issue | Fix |
|---|---|
| DPO pairs built from every SFT row, so val/test prompts leaked into DPO training | DPO pairs are partitioned by the SFT split |
| One fixed "rejected" answer for every pair (DPO saturates and learns little) | scenario-aware hard negatives (5 failure modes) plus on-policy negatives mined from the SFT model |
| Logit KL between Llama-3.1 and Phi-3 (different vocabularies, so undefined) | `distill.py` detects the vocab mismatch and uses sequence-level KD; logit KD otherwise |
| KD loss `batchmean` over all positions (prompt + padding distilled, wrong normaliser) | KL masked to assistant tokens, averaged per token |
| vLLM served the DPO adapter on the raw base model (the adapter was trained on SFT-merged weights) | vLLM base = merged SFT checkpoint, DPO = adapter on top |
| "EMERGENCY_HALT ⇒ zero velocity" stated in the README but not enforced | enforced by Pydantic validators and re-checked in the bridge |
| `cmd_vel` was an untyped `dict`; `payload_type`/`ros2` consistency only checked offline | typed `Vector3`; consistency enforced in the schema |
| Tokenization hard-wired to the Llama-3.1 tokenizer | splits stored as chat messages; each trainer masks with its own tokenizer |
| LLM-generated examples were labelled with the *requested* category, not the actual one | deterministic classifier relabels; unsafe generations are discarded before the judge |

## Project structure

```
roboscript/
├── constants.py              SYSTEM_PROMPT, 12-category taxonomy, judge rubric
├── safety.py                 classifier (priority order), reference policy, check(), guard()
├── schemas/                  RoboResponse (+ invariants), telemetry models
├── data/                     synthetic.py · offline.py · procedural.py · negatives.py · onpolicy.py · pipeline.py
├── training/                 common.py (profiles) · bootstrap.py · sft_trainer.py · dpo_trainer.py
├── distillation/distill.py   logit / sequence-level KD
├── eval/                     golden.py · evaluator.py · generate.py
├── serve/                    api.py · backends.py · vllm_server.py
├── ros2_bridge/              bridge_node.py · mock_subscriber.py · transport.py (rclpy | in-process)
└── deploy/                   entrypoint.py (SageMaker /ping /invocations) · sagemaker_deploy.py
data/seed/examples.json       36 hand-authored seeds (all agree with safety.reference_action)
data/golden/eval_cases.json   50 safety-critical + 120 compound stress cases
docker/ · docker-compose.yml · scripts/run_local_pipeline.sh · tests/{unit,integration}
```

## License

Educational / portfolio project.
