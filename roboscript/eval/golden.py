"""Build the fixed golden evaluation sets (data/golden/eval_cases.json).

    python -m roboscript.eval.golden

* ``safety_critical`` — 50 scenarios whose highest-priority fault is a life-safety
  category (2: person/forklift in path, 10: e-stop), ~30% of them compound
  (e.g. e-stop *and* critical battery), drawn with a seed never used for training.
  This is the PRD's "50-case safety test set".
* ``stress`` — 120 compound scenarios (two simultaneous faults). The label follows
  the safety priority order, so this probes whether the model resolves conflicts
  the way the interlock does, not just pattern-matches the single-fault seeds.

Labels come from ``roboscript.safety.reference_action``. The test split from
pipeline.py is the third, in-distribution set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from roboscript.data.procedural import render_response, sample_scenario
from roboscript.safety import classify

GOLDEN = Path("data/golden/eval_cases.json")


def build(seed: int = 20260607) -> dict:
    rng = random.Random(seed)
    safety, stress = [], []
    while len(safety) < 50:
        cid = 2 if len(safety) % 2 == 0 else 10
        s = sample_scenario(cid, rng, compound_p=0.3)
        if classify(s) in (2, 10):
            safety.append({"scenario": s, "category_id": classify(s),
                           "reference": render_response(s, rng)})
    cats = list(range(1, 13))
    while len(stress) < 120:
        s = sample_scenario(rng.choice(cats), rng, compound_p=1.0)
        codes = [c for c in s["telemetry"]["active_fault_codes"] if c.startswith("ERR_")]
        if len(codes) + (s.get("yolo") is not None) >= 2:
            stress.append({"scenario": s, "category_id": classify(s),
                           "reference": render_response(s, rng)})
    return {"safety_critical": safety, "stress": stress}


def main() -> None:
    sets = build()
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(sets, indent=1))
    print({k: len(v) for k, v in sets.items()}, "->", GOLDEN)


if __name__ == "__main__":
    main()
