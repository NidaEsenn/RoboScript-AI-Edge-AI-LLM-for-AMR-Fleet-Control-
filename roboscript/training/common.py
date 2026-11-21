"""Shared training utilities: hardware profiles, model/tokenizer loading, datasets.

Two profiles share every line of training code:

* ``gpu``   — the PRD configuration: Llama-3.1-8B-Instruct, 4-bit NF4 QLoRA,
              LoRA r=16/alpha=32 on all attention+MLP projections, bf16.
* ``local`` — a CPU-only profile used to run the whole pipeline end-to-end on a
              laptop / CI box with *no access to pretrained weights*. It trains a
              small Llama-architecture model from scratch with a domain BPE
              tokenizer (see ``bootstrap.py``). Numbers from this profile validate
              the plumbing and the relative effect of each stage; they are not a
              substitute for the 8B run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path("data/processed")
OUT_DIR = Path("outputs")

# Llama-3-style chat template (same special tokens), used for the local tokenizer.
LOCAL_CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)
SPECIAL_TOKENS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                  "<|eot_id|>", "<|pad|>", "<|unk|>"]

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Profile:
    name: str
    base_model: str
    max_length: int = 1024
    # SFT
    sft_epochs: float = 3
    sft_lr: float = 2e-4
    sft_batch: int = 2
    sft_grad_accum: int = 8
    sft_use_lora: bool = True
    load_in_4bit: bool = True
    bf16: bool = True
    # DPO
    dpo_epochs: float = 1
    dpo_lr: float = 5e-5
    dpo_batch: int = 1
    dpo_grad_accum: int = 16
    dpo_beta: float = 0.1
    # RPO = DPO + NLL on the chosen answer (Pang et al., 2024). Pure sigmoid DPO pushed
    # the chosen log-prob down (likelihood displacement) and broke JSON validity in our
    # local run; the NLL anchor keeps the policy on the valid-output manifold.
    dpo_loss_types: list[str] = field(default_factory=lambda: ["sigmoid", "sft"])
    dpo_loss_weights: list[float] = field(default_factory=lambda: [1.0, 1.0])
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: list(LORA_TARGETS))
    # distillation
    student_model: str = "microsoft/Phi-3-mini-4k-instruct"
    distill_epochs: float = 2
    distill_lr: float = 1e-4
    distill_alpha: float = 0.5
    distill_temperature: float = 4.0


PROFILES = {
    "gpu": Profile(name="gpu", base_model=os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                   student_model=os.getenv("EDGE_MODEL", "microsoft/Phi-3-mini-4k-instruct")),
    "local": Profile(
        name="local", base_model=str(OUT_DIR / "local-base"), max_length=512,
        sft_epochs=10, sft_lr=1.5e-3, sft_batch=16, sft_grad_accum=1, sft_use_lora=False,
        load_in_4bit=False, bf16=False,
        dpo_epochs=1, dpo_lr=2e-4, dpo_batch=16, dpo_grad_accum=1,
        student_model=str(OUT_DIR / "local-student-init"),
        distill_epochs=10, distill_lr=1.5e-3, distill_temperature=2.0,
    ),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]


def load_json(path: Path | str) -> list[dict]:
    return json.loads(Path(path).read_text())


def to_prompt_completion(rows: list[dict]) -> list[dict]:
    """Chat rows -> TRL conversational prompt/completion (loss on completion only)."""
    return [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows]


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, token=os.getenv("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(path: str, profile: Profile, for_training: bool = True):
    """Load a causal LM; 4-bit NF4 on the GPU profile, fp32 on CPU."""
    import torch
    from transformers import AutoModelForCausalLM
    kwargs: dict = {"token": os.getenv("HF_TOKEN")}
    if profile.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    elif profile.bf16:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if profile.load_in_4bit and for_training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def lora_config(profile: Profile):
    from peft import LoraConfig
    return LoraConfig(r=profile.lora_r, lora_alpha=profile.lora_alpha,
                      lora_dropout=profile.lora_dropout, bias="none",
                      target_modules=profile.lora_targets, task_type="CAUSAL_LM")


def report_to() -> list[str]:
    return ["wandb"] if os.getenv("WANDB_API_KEY") else ["none"]


def write_log_history(trainer, out_dir: Path) -> None:
    (Path(out_dir) / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=1))
