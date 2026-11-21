"""Batched greedy generation shared by evaluation, distillation and serving."""
from __future__ import annotations

import json
import time

from roboscript.constants import SYSTEM_PROMPT


def build_messages(scenario: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(scenario)}]


def load_for_inference(path: str):
    """Load a model dir (full model or PEFT adapter) + tokenizer for CPU/GPU inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pathlib import Path
    p = Path(path)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if (p / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=dtype).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tok


def generate(model, tok, scenarios: list[dict], batch_size: int = 16,
             max_new_tokens: int = 256) -> list[tuple[str, float]]:
    """Return [(completion_text, latency_ms_per_example)] in input order."""
    import torch
    tok.padding_side = "left"
    eos_ids = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot not in eos_ids and eot != tok.unk_token_id:
        eos_ids.append(eot)
    out: list[tuple[str, float]] = []
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i + batch_size]
        prompts = [tok.apply_chat_template(build_messages(s), tokenize=False,
                                           add_generation_prompt=True) for s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        dt = (time.perf_counter() - t0) * 1000 / len(batch)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            out.append((tok.decode(row, skip_special_tokens=True).strip(), dt))
    return out


def parse_json(text: str) -> dict | None:
    """Parse the model output; tolerate leading/trailing junk around one JSON object."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
