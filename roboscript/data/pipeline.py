"""Session B, step 1 — dataset preparation (PRD Section 6.2).

Turns validated {scenario, response} examples into a training-ready dataset:
  1. format into LLaMA-3 chat messages (system / user / assistant)
  2. re-validate the assistant JSON with RoboResponse (discard invalid)
  3. MinHash dedup at similarity >= 0.9
  4. tokenize + build labels, masking system/user tokens with -100
  5. 80/10/10 train/val/test split
  6. save splits locally (Hub push intentionally out of scope for now)

    python -m roboscript.data.pipeline

Loss masking needs the (gated) Llama-3.1 tokenizer, so set HF_TOKEN in .env and
accept the license at huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv
from pydantic import ValidationError
from transformers import AutoTokenizer

load_dotenv()  # read HF_TOKEN

from roboscript.constants import SYSTEM_PROMPT
from roboscript.schemas.payload import RoboResponse

IN_PATH = Path("data/processed/sft_train.json")
OUT_DIR = Path("data/processed")
# Ungated mirror of meta-llama/Llama-3.1-8B-Instruct: identical tokenizer + chat
# template, no license gate. Swap to the official repo once Meta approves access.
MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SIM_THRESHOLD = 0.9   # MinHash: drop examples >= 90% similar to one already kept
NUM_PERM = 128        # MinHash permutations — more = more accurate, slightly slower
SEED = 42             # fixed so the split is reproducible
LABEL_IGNORE = -100   # PyTorch's cross-entropy ignores tokens with this label


def to_chat(example: dict) -> dict:
    """One {scenario, response} -> the LLaMA-3 messages format (PRD 6.2).

    The user turn is the sensor telemetry; the assistant turn is the exact JSON
    the model must learn to produce. We json.dumps() both so the strings are
    unambiguous and round-trippable.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(example["scenario"])},
            {"role": "assistant", "content": json.dumps(example["response"])},
        ]
    }


def is_valid(chat: dict) -> bool:
    """Re-validate the assistant JSON with RoboResponse (PRD step 4)."""
    try:
        RoboResponse.model_validate_json(chat["messages"][2]["content"])
        return True
    except ValidationError:
        return False


def _minhash(text: str) -> MinHash:
    """Build a MinHash signature from a text's unique whitespace tokens."""
    m = MinHash(num_perm=NUM_PERM)
    for token in set(text.split()):
        m.update(token.encode("utf-8"))
    return m


def dedup(chats: list[dict]) -> list[dict]:
    """Drop near-duplicates by scenario text using MinHash LSH (PRD step 3).

    DECISION: dedup on the USER (scenario) text, not the whole conversation.
    WHY: the system prompt is identical in every row, so including it would make
         everything look ~similar; the scenario is what actually varies.
    TRADEOFF: two different scenarios with the same safe response are both kept —
         which is fine, we want decision diversity, not just text diversity.
    """
    lsh = MinHashLSH(threshold=SIM_THRESHOLD, num_perm=NUM_PERM)
    kept: list[dict] = []
    for i, chat in enumerate(chats):
        sig = _minhash(chat["messages"][1]["content"])
        if lsh.query(sig):          # a near-duplicate is already in the set
            continue
        lsh.insert(f"row-{i}", sig)
        kept.append(chat)
    return kept


def _template_ids(tokenizer, messages: list[dict], add_gen: bool) -> list[int]:
    """apply_chat_template -> plain list[int].

    return_dict=True forces a BatchEncoding so we can pull "input_ids" by name
    instead of guessing the return shape (the bug that saved dict KEYS as tokens).
    """
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_gen, return_dict=True)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):   # batched [[...]] form
        ids = ids[0]
    return list(ids)


def tokenize_and_mask(tokenizer, chat: dict) -> dict:
    """Tokenize the conversation and build masked labels (PRD step 2).

    DECISION: assemble input_ids = prompt + response ourselves; mask the prompt.
    WHY: labels must be -100 for every system/user/header token so the loss only
         scores the assistant's response. Building the sequence by hand makes the
         prompt-vs-response boundary exact (prompt_ids is the masked span) and
         avoids this template's stray trailing assistant header.
    TRADEOFF: we trust that tokenizing the response text standalone matches its
         in-context tokenization — safe here because the prompt ends on the
         assistant-header special tokens, which force a clean boundary.
    """
    messages = chat["messages"]
    # Prompt: system + user + the "assistant:" cue. This is the model's INPUT.
    prompt_ids = _template_ids(tokenizer, messages[:-1], add_gen=True)

    # Response: the assistant's JSON content + end-of-turn. This is what the model
    # must LEARN to produce. We assemble it ourselves rather than re-render the full
    # conversation, because this template appends a stray trailing assistant header
    # that would otherwise leak into the learned tokens.
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    response_ids = tokenizer(
        messages[-1]["content"], add_special_tokens=False)["input_ids"] + [eot_id]

    input_ids = prompt_ids + response_ids
    labels = [LABEL_IGNORE] * len(prompt_ids) + response_ids
    return {
        "messages": messages,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def split(rows: list[dict]) -> dict:
    """Shuffle once (fixed seed) and slice 80/10/10 (PRD step 5)."""
    rows = rows[:]
    random.Random(SEED).shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }


def main() -> None:
    raw = json.loads(IN_PATH.read_text())
    print(f"loaded {len(raw)} examples")

    chats = [to_chat(e) for e in raw]
    chats = [c for c in chats if is_valid(c)]
    print(f"after Pydantic re-validation: {len(chats)}")

    chats = dedup(chats)
    print(f"after MinHash dedup @ {SIM_THRESHOLD}: {len(chats)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv("HF_TOKEN"))
    rows = [tokenize_and_mask(tok, c) for c in chats]

    splits = split(rows)
    for name, part in splits.items():
        (OUT_DIR / f"{name}.json").write_text(json.dumps(part, indent=2))
        print(f"  {name}: {len(part)} -> {OUT_DIR / f'{name}.json'}")
    print(f"total {sum(len(p) for p in splits.values())} across splits")


if __name__ == "__main__":
    main()
