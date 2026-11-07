import json
from collections import Counter

import pytest

from roboscript.data.pipeline import build, tokenize_and_mask, to_chat
from roboscript.data.procedural import generate
from roboscript.safety import check
from roboscript.schemas.payload import RoboResponse


@pytest.fixture(scope="module")
def generated():
    return generate(per_category=40, seed=1, include_seeds=True)


def test_procedural_rows_valid_and_safe(generated):
    rows, dpo, discarded = generated
    assert not discarded
    for r in rows:
        RoboResponse.model_validate(r["response"])
        v = check(r["scenario"], r["response"])
        assert v.safe and v.exact_match and v.category == r["category_id"]


def test_procedural_is_roughly_balanced(generated):
    rows, _, _ = generated
    counts = Counter(r["category_id"] for r in rows)
    assert set(counts) == set(range(1, 13))
    mean = sum(counts.values()) / 12
    assert all(abs(c - mean) / mean < 0.35 for c in counts.values()), counts


def test_dpo_modes_are_diverse(generated):
    _, dpo, _ = generated
    assert len({d["failure_mode"] for d in dpo}) >= 4


def test_split_has_no_leak_and_dpo_follows_split(generated, tmp_path):
    rows, dpo, _ = generated
    (tmp_path / "sft.json").write_text(json.dumps(rows))
    (tmp_path / "dpo.json").write_text(json.dumps(dpo))
    out = tmp_path / "out"
    build(tmp_path / "sft.json", tmp_path / "dpo.json", out)
    splits = {k: json.loads((out / f"{k}.json").read_text()) for k in ("train", "val", "test")}
    prompts = {k: {r["messages"][1]["content"] for r in v} for k, v in splits.items()}
    assert not prompts["train"] & prompts["test"]
    assert not prompts["train"] & prompts["val"]
    for name in ("train", "val"):
        for p in json.loads((out / f"dpo_{name}.json").read_text()):
            assert p["prompt"][1]["content"] in prompts[name]
            assert p["prompt"][1]["content"] not in prompts["test"]
    # stratified: every category present in test
    assert {r["category_id"] for r in splits["test"]} == set(range(1, 13))


def test_loss_mask_covers_only_assistant(seeds):
    from roboscript.training.bootstrap import train_tokenizer
    chats = [to_chat(e) for e in seeds]
    tok = train_tokenizer([m["content"] for c in chats for m in c["messages"]], 600)
    x = tokenize_and_mask(tok, chats[0])
    assert len(x["input_ids"]) == len(x["labels"])
    learned = [t for t, l in zip(x["input_ids"], x["labels"]) if l != -100]
    text = tok.decode(learned, skip_special_tokens=True)
    assert json.loads(text) == seeds[0]["response"]
    prompt_part = tok.decode([t for t, l in zip(x["input_ids"], x["labels"]) if l == -100])
    assert "Edge-AI Guidance Copilot" in prompt_part and "decision" not in prompt_part
