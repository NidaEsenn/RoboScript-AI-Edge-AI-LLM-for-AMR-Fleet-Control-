"""On-policy hard-negative mining for DPO.

    python -m roboscript.data.onpolicy --model outputs/roboscript-sft --n-prompts 800

Off-policy negatives (negatives.py) are written by hand, so the SFT model already
assigns them very low likelihood (log p ~ -190 vs -20 for the chosen answer in our
runs) and DPO saturates almost immediately — it learns little the SFT model did
not already know. Here we sample the SFT model itself (temperature > 0) on the
*training* prompts and keep every sample that ``safety.check`` scores as worse than
the reference: wrong decision, wrong payload shape, unsafe, or invalid JSON. Those
are exactly the mistakes the policy is likely to make, which is where preference
optimisation has signal (cf. on-policy / self-play preference data in RLHF work).

Writes data/processed/dpo_train_onpolicy.json (TRL conversational format) and a
mixed file dpo_train_mixed.json = on-policy pairs + the off-policy pairs.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from roboscript.eval.generate import build_messages, parse_json
from roboscript.safety import check

DATA = Path("data/processed")


def sample(model, tok, scenarios: list[dict], k: int, temperature: float, batch_size: int):
    import torch
    tok.padding_side = "left"
    eos = [tok.eos_token_id]
    out = []
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i + batch_size]
        prompts = [tok.apply_chat_template(build_messages(s), tokenize=False,
                                           add_generation_prompt=True) for s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                               num_return_sequences=k, max_new_tokens=256,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        g = g[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(g, skip_special_tokens=True)
        for j, s in enumerate(batch):
            out.append((s, texts[j * k:(j + 1) * k]))
    return out


def mine(model_dir: str, n_prompts: int, k: int, temperature: float, seed: int = 0) -> tuple[list[dict], Counter]:
    import torch
    from roboscript.eval.generate import load_for_inference
    torch.manual_seed(seed)
    model, tok = load_for_inference(model_dir)
    rows = json.loads((DATA / "train.json").read_text())
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n_prompts, len(rows)))
    scen = [json.loads(r["messages"][1]["content"]) for r in rows]
    gold = {json.dumps(json.loads(r["messages"][1]["content"])): r for r in rows}
    stats: Counter = Counter()
    pairs = []
    for s, samples in sample(model, tok, scen, k, temperature, batch_size=16):
        row = gold[json.dumps(s)]
        chosen = row["messages"][2]["content"]
        cv = check(s, json.loads(chosen))
        seen = set()
        for text in samples:
            stats["samples"] += 1
            obj = parse_json(text)
            v = check(s, obj)
            if obj is not None and v.valid_schema and v.exact_match and v.safe:
                stats["correct"] += 1
                continue
            kind = ("invalid_json" if obj is None else "schema" if not v.valid_schema
                    else "unsafe" if not v.safe else "wrong_action")
            stats[kind] += 1
            key = text.strip()
            if key in seen or not (len(v.violations) > len(cv.violations) or not v.exact_match):
                continue
            seen.add(key)
            pairs.append({"prompt": row["messages"][:2],
                          "chosen": [row["messages"][2]],
                          "rejected": [{"role": "assistant", "content": text.strip()}],
                          "failure_mode": f"onpolicy_{kind}",
                          "category_id": row["category_id"]})
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/roboscript-sft")
    ap.add_argument("--n-prompts", type=int, default=800)
    ap.add_argument("--k", type=int, default=4, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    pairs, stats = mine(a.model, a.n_prompts, a.k, a.temperature)
    (DATA / "dpo_train_onpolicy.json").write_text(json.dumps(pairs, indent=1))
    offp = json.loads((DATA / "dpo_train.json").read_text())
    mixed = pairs + offp
    random.Random(0).shuffle(mixed)
    (DATA / "dpo_train_mixed.json").write_text(json.dumps(mixed, indent=1))
    print("sampling stats:", dict(stats))
    print("on-policy pairs by category:", dict(sorted(Counter(p["category_id"] for p in pairs).items())))
    print(f"{len(pairs)} on-policy pairs; mixed set {len(mixed)} -> {DATA / 'dpo_train_mixed.json'}")


if __name__ == "__main__":
    main()
