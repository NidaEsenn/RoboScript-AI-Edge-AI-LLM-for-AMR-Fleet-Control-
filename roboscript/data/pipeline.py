"""Session B, step 1 — dataset preparation (PRD Section 6.2).

Turns validated {scenario, response} examples into a training-ready dataset:
  1. format into chat messages (system / user / assistant)
  2. re-validate the assistant JSON with RoboResponse (discard invalid)
  3. MinHash dedup at similarity >= 0.9
  4. 80/10/10 train/val/test split, *stratified by fault category*
  5. split the DPO pairs with the SAME assignment (no test prompt ever reaches DPO)
  6. optional: tokenize + build labels (system/user tokens masked with -100) for a
     given tokenizer, and optional push to the HuggingFace Hub

    python -m roboscript.data.pipeline                       # model-agnostic splits
    python -m roboscript.data.pipeline --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct

Changes vs. the first version:
  * tokenization is no longer hard-wired to the Llama-3.1 tokenizer. Splits are saved
    as chat messages; each trainer tokenizes with *its own* tokenizer, and the
    loss mask is computed generically (prompt = chat template with generation prompt,
    completion = the remainder), so the same data trains Llama, Phi-3 or a tiny model.
  * DPO pairs used to be built from every SFT row, including the ones that end up in
    val/test — a train/test leak. They are now partitioned by the split.
  * category_id is kept on every row so evaluation can score per category.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from pydantic import ValidationError

from roboscript.constants import SYSTEM_PROMPT
from roboscript.schemas.payload import RoboResponse

IN_PATH = Path("data/processed/sft_train.json")
DPO_PATH = Path("data/processed/dpo_pairs.json")
OUT_DIR = Path("data/processed")
DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # ungated Llama-3.1 mirror
SIM_THRESHOLD = 0.9   # MinHash: drop examples >= 90% similar to one already kept
NUM_PERM = 128        # MinHash permutations — more = more accurate, slightly slower
SEED = 42             # fixed so the split is reproducible
LABEL_IGNORE = -100   # PyTorch's cross-entropy ignores tokens with this label


def to_chat(example: dict) -> dict:
    """One {scenario, response} -> chat messages (+ metadata kept for evaluation)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(example["scenario"])},
            {"role": "assistant", "content": json.dumps(example["response"])},
        ],
        "category_id": example.get("category_id"),
        "source": example.get("source", "unknown"),
    }


def is_valid(chat: dict) -> bool:
    """Re-validate the assistant JSON with RoboResponse (PRD step 4)."""
    try:
        RoboResponse.model_validate_json(chat["messages"][2]["content"])
        return True
    except ValidationError:
        return False


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for token in set(text.split()):
        m.update(token.encode("utf-8"))
    return m


def dedup(chats: list[dict]) -> list[dict]:
    """Drop near-duplicates by scenario text using MinHash LSH (PRD step 3).

    DECISION: dedup on the USER (scenario) text, not the whole conversation.
    WHY: the system prompt is identical in every row, so including it would make
         everything look ~similar; the scenario is what actually varies.
    """
    lsh = MinHashLSH(threshold=SIM_THRESHOLD, num_perm=NUM_PERM)
    kept: list[dict] = []
    for i, chat in enumerate(chats):
        sig = _minhash(chat["messages"][1]["content"])
        if lsh.query(sig):
            continue
        lsh.insert(f"row-{i}", sig)
        kept.append(chat)
    return kept


def split(rows: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """Stratified 80/10/10 by category_id (fixed seed)."""
    by_cat: dict = defaultdict(list)
    for r in rows:
        by_cat[r.get("category_id")].append(r)
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for cid in sorted(by_cat, key=lambda c: (c is None, c)):
        part = by_cat[cid][:]
        rng.shuffle(part)
        n = len(part)
        n_val = max(1, round(n * 0.1)) if n >= 10 else 0
        n_test = max(1, round(n * 0.1)) if n >= 10 else 0
        out["val"] += part[:n_val]
        out["test"] += part[n_val:n_val + n_test]
        out["train"] += part[n_val + n_test:]
    for k in out:
        rng.shuffle(out[k])
    return out


def split_dpo(pairs: list[dict], splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Assign each DPO pair to the split of its prompt; pairs whose prompt was
    deduplicated away are dropped. Output is TRL's conversational preference format."""
    where = {}
    for name, rows in splits.items():
        for r in rows:
            where[r["messages"][1]["content"]] = name
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        name = where.get(p["prompt"])
        if name is None:
            continue
        out[name].append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": p["prompt"]}],
            "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}],
            "failure_mode": p.get("failure_mode", "forward_motion"),
            "category_id": p.get("category_id"),
        })
    return out


# ---------------------------------------------------------------- tokenization
def _ids(enc) -> list[int]:
    if hasattr(enc, "keys") and "input_ids" in enc:
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def tokenize_and_mask(tokenizer, chat: dict) -> dict:
    """Tokenize a conversation and mask everything except the assistant response.

    Generic for any chat template: prompt_ids = template(system+user, generation
    prompt); full_ids = template(all messages). If prompt_ids is a prefix of
    full_ids the completion is the remainder (and trailing template junk after the
    end-of-turn is kept, matching what the model sees at inference). Otherwise we
    fall back to prompt_ids + tokenize(response) + eos.
    """
    messages = chat["messages"]
    prompt_ids = _ids(tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True))
    full_ids = _ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True))
    if full_ids[:len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
        response_ids = full_ids[len(prompt_ids):]
    else:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        end = eot if isinstance(eot, int) and eot != tokenizer.unk_token_id else tokenizer.eos_token_id
        response_ids = _ids(tokenizer(messages[-1]["content"], add_special_tokens=False)) + [end]
    input_ids = prompt_ids + response_ids
    labels = [LABEL_IGNORE] * len(prompt_ids) + response_ids
    return {**chat, "input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def build(in_path: Path = IN_PATH, dpo_path: Path = DPO_PATH, out_dir: Path = OUT_DIR,
          tokenizer_id: str | None = None, push_to_hub: str | None = None) -> dict:
    raw = json.loads(Path(in_path).read_text())
    print(f"loaded {len(raw)} examples")
    chats = [c for c in (to_chat(e) for e in raw) if is_valid(c)]
    print(f"after Pydantic re-validation: {len(chats)}")
    chats = dedup(chats)
    print(f"after MinHash dedup @ {SIM_THRESHOLD}: {len(chats)}")
    splits = split(chats)

    if tokenizer_id:
        from dotenv import load_dotenv
        from transformers import AutoTokenizer
        load_dotenv()
        tok = AutoTokenizer.from_pretrained(tokenizer_id, token=os.getenv("HF_TOKEN"))
        splits = {k: [tokenize_and_mask(tok, c) for c in v] for k, v in splits.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(part, indent=1))
        print(f"  {name}: {len(part)} -> {out_dir / f'{name}.json'}")

    dpo_counts = {}
    if Path(dpo_path).exists():
        dpo = split_dpo(json.loads(Path(dpo_path).read_text()), splits)
        for name in ("train", "val"):
            (out_dir / f"dpo_{name}.json").write_text(json.dumps(dpo[name], indent=1))
        dpo_counts = {k: len(v) for k, v in dpo.items()}
        print(f"  DPO pairs by split (test pairs discarded): {dpo_counts}")

    if push_to_hub:
        from datasets import Dataset, DatasetDict
        DatasetDict({k: Dataset.from_list([{"messages": r["messages"],
                                            "category_id": r["category_id"]} for r in v])
                     for k, v in splits.items()}).push_to_hub(push_to_hub,
                                                             token=os.getenv("HF_TOKEN"))
        print(f"pushed to https://huggingface.co/datasets/{push_to_hub}")
    return {"splits": {k: len(v) for k, v in splits.items()}, "dpo": dpo_counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(IN_PATH))
    ap.add_argument("--dpo", default=str(DPO_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tokenizer", default=None,
                    help=f"also store input_ids/labels for this tokenizer (e.g. {DEFAULT_TOKENIZER})")
    ap.add_argument("--push-to-hub", default=None, help="e.g. {HF_USERNAME}/roboscript-dataset")
    a = ap.parse_args()
    build(Path(a.in_path), Path(a.dpo), Path(a.out_dir), a.tokenizer, a.push_to_hub)


if __name__ == "__main__":
    main()
