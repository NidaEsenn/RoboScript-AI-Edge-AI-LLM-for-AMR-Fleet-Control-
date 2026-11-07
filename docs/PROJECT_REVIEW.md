# Project review: what RoboScript is, and how it compares

## What it is

RoboScript fine-tunes an LLM to act as a **fault-response copilot** for warehouse AMRs.
Input is a telemetry snapshot (YOLO detection, LiDAR minimum distance and status, speed,
battery, fault codes). Output is one JSON object that the ROS 2 stack can execute
directly:

* **A** `CMD_VEL_DIRECT`: a zero Twist on `/cmd_vel` (halt),
* **B** `NAV2_ACTION`: a `NavigateToPose` goal (reroute, go to dock, speed-limited tree),
* **C** `FAULT_REPORT_ONLY`: log the fault, no motion command.

The project is mainly about **LLM post-training MLOps**: synthetic data with
validation and a judge, chat formatting with loss masking, QLoRA SFT, DPO safety
alignment, evaluation gates, distillation to an edge model, vLLM multi-LoRA serving,
a ROS 2 bridge and cloud deployment.

## Where it sits among related work

| Line of work | Representative examples | What they do | How RoboScript differs |
|---|---|---|---|
| LLMs as robot planners | SayCan, Code-as-Policies, *ChatGPT for Robotics* | LLM turns natural-language goals into skills or code | RoboScript maps **structured telemetry** to a **closed action set**. The task is much narrower and easier to verify. |
| LLM ↔ ROS integration | ROS-LLM, ROSGPT and similar ROS 2 + LLM bridges | Natural-language → ROS topics/actions/behaviour trees | Same integration layer, but the output is schema-constrained and validated before publishing |
| Guardrails for LLM-controlled robots | RoboGuard-style runtime guards; jailbreak studies such as RoboPAIR | Show LLM robot controllers can be pushed into unsafe actions; propose rule-based or LTL guard layers | RoboScript's interlock (`safety.guard`) is in this spirit. The two life-safety categories never depend on the model. |
| Structured output | JSON mode, grammar-constrained decoding (Outlines, XGrammar, vLLM guided decoding) | Guarantee syntactically valid JSON/schema at decode time | RoboScript reaches validity through training. Constrained decoding would give ~100% schema validity without training, so it is the baseline any "JSON validity" claim should beat. |
| Industrial AMR safety | Safety-rated laser scanners and safety PLCs (ISO 3691-4, ISO 13849 / IEC 61508) | Certified, hard-real-time (tens of ms) protective stops | The LLM cannot and should not replace this layer. RoboScript belongs *above* it, as a supervisory decision and reporting layer. |
| Fault diagnosis with LLMs | LLM log/alarm triage in manufacturing and IT operations | Summarise and classify unstructured logs and alarms | The input is small and structured here. The LLM's real advantage over rules (unstructured text) is not yet exercised. |

## Strengths

* **Breadth across the full lifecycle.** Few portfolio projects connect data generation,
  SFT, DPO, eval gates, distillation, serving and a robotics middleware integration end
  to end. This is the project's main selling point for ML/robotics-infra roles.
* **Safety-first framing.** Schema invariants, fallback-to-halt and explicit
  life-safety categories reflect how real fleet software is reviewed.
* **Concrete, checkable output contract.** Because the output is a closed schema,
  evaluation can be automatic and exact. Most LLM-for-robotics demos lack this.

## Weaknesses (and what was done about them)

1. **The core mapping is a lookup table.** Twelve categories with numeric thresholds
   can be solved exactly by ~100 lines of rules (`safety.reference_action`), in
   microseconds, with 100% accuracy. An 8B model adds latency and failure modes without
   adding capability **on this input format**. *Mitigation here:* the rules are made
   explicit and used as the ground truth, the interlock and the eval oracle.
   *Recommendation:* give the model inputs rules cannot handle: free-text operator
   notes, raw log lines, novel fault codes, multi-robot context, explanations for
   human supervisors.
2. **Real-time claims.** "Respond in milliseconds" and "≤300 ms p95" are far slower than
   a safety scanner's protective stop. *Mitigation:* per-endpoint deadlines with halt
   on timeout, plus a fast path that bypasses the model for life-safety categories.
3. **Circular evaluation.** The data comes from the taxonomy, and the eval measures
   agreement with the same taxonomy. High scores show the model learned the rules,
   not that the rules are right. *Mitigation:* a compound-fault stress set (priority
   conflicts) and a separate 50-case safety set. *Still missing:* real or expert-labelled
   incidents.
4. **Design bugs in the original spec**, now fixed: DPO train/test leak, a single
   trivial rejected answer, logit-KD across mismatched tokenizers, the DPO adapter
   served on the wrong base, and invariants claimed but not enforced (see the README
   table).
5. **Missing baselines.** A fair comparison should include (a) the rule engine,
   (b) the base model with constrained decoding and a few-shot prompt, and
   (c) SFT/DPO. The evaluator supports any `name=path` model list. Add (b) when a
   GPU is available.

## Verdict

As a **portfolio / learning project in LLM post-training and MLOps** it is strong:
broad, well-scoped and easy to demo. As a **solution to the AMR fault-response
problem** it is weaker than it looks, because the problem as specified is better
solved by rules plus certified safety hardware. The most convincing framing is the
one the code now implements: **a safety-gated LLM copilot**, where deterministic
layers own every life-safety decision and the model handles the nuanced,
non-critical choices (reroute vs halt, dock vs continue) and the human-facing
reasoning. Its value would grow a lot with unstructured inputs and real incident data.
