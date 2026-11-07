"""Session C — DPO safety alignment (PRD Section 6.4).

    python -m roboscript.training.dpo_trainer --profile gpu
    python -m roboscript.training.dpo_trainer --profile local

Reference model : the SFT model (merged SFT checkpoint on the gpu profile).
Policy          : a *fresh* LoRA adapter on top of that same model. With a PEFT
                  policy TRL uses the adapter-disabled model as the reference, so
                  no second copy of the 8B weights is held in memory.
Data            : data/processed/dpo_train.json / dpo_val.json — hard negatives from
                  roboscript/data/negatives.py, partitioned by the SFT split so no
                  test prompt is ever trained on.

Writes outputs/roboscript-dpo/ (adapter), outputs/roboscript-dpo-merged/ (full
model, used by eval / distillation / single-model serving) and
outputs/roboscript-dpo/dpo_health.json with the PRD AC-04 margin checks.

Serving note: the adapter is trained on top of the *merged SFT* weights, so vLLM
must load roboscript-sft-merged as its base model when serving this adapter
(see serve/vllm_server.py). Loading it on the raw base model, as the original PRD
command did, silently drops the SFT behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset

from roboscript.training.common import (DATA_DIR, OUT_DIR, get_profile, load_causal_lm,
                                        load_json, load_tokenizer, lora_config, report_to,
                                        write_log_history)


def dpo_health(log_history: list[dict]) -> dict:
    """AC-04: margins positive and chosen > rejected at every logged train step."""
    steps = [h for h in log_history if "rewards/margins" in h and "eval_loss" not in h]
    evals = [h for h in log_history if "eval_rewards/margins" in h]
    margins = [h["rewards/margins"] for h in steps]
    chosen_gt = [h["rewards/chosen"] > h["rewards/rejected"] for h in steps]
    return {
        "logged_steps": len(steps),
        "margins_first": margins[0] if margins else None,
        "margins_last": margins[-1] if margins else None,
        "margins_positive_all_steps": bool(margins) and all(m > 0 for m in margins),
        "chosen_gt_rejected_all_steps": bool(chosen_gt) and all(chosen_gt),
        "final_reward_accuracy": steps[-1].get("rewards/accuracies") if steps else None,
        "eval_margins": evals[-1].get("eval_rewards/margins") if evals else None,
        "eval_accuracy": evals[-1].get("eval_rewards/accuracies") if evals else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local", choices=["local", "gpu"])
    ap.add_argument("--sft", default=None, help="SFT model dir (default per profile)")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--train-file", default=str(DATA_DIR / "dpo_train.json"),
                    help="dpo_train.json (off-policy) or dpo_train_mixed.json (+ on-policy, see data/onpolicy.py)")
    ap.add_argument("--out", default=str(OUT_DIR / "roboscript-dpo"))
    ap.add_argument("--loss", default=None, choices=["dpo", "rpo"],
                    help="dpo = sigmoid only (PRD); rpo = sigmoid + NLL on chosen (default)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--push-to-hub", default=None, help="{HF_USERNAME}/roboscript-dpo-adapter")
    a = ap.parse_args()

    from trl import DPOConfig, DPOTrainer

    prof = get_profile(a.profile)
    if a.loss == "dpo":
        prof.dpo_loss_types, prof.dpo_loss_weights = ["sigmoid"], [1.0]
    if a.lr:
        prof.dpo_lr = a.lr
    sft_dir = a.sft or str(OUT_DIR / ("roboscript-sft-merged" if prof.sft_use_lora else "roboscript-sft"))
    tok = load_tokenizer(sft_dir)
    model = load_causal_lm(sft_dir, prof)

    keep = ("prompt", "chosen", "rejected")
    train = Dataset.from_list([{k: r[k] for k in keep} for r in load_json(a.train_file)])
    val = Dataset.from_list([{k: r[k] for k in keep} for r in load_json(DATA_DIR / "dpo_val.json")])

    cfg = DPOConfig(
        output_dir=a.out,
        beta=prof.dpo_beta,
        loss_type=prof.dpo_loss_types,
        loss_weights=prof.dpo_loss_weights,
        max_length=prof.max_length,
        num_train_epochs=a.epochs or prof.dpo_epochs,
        per_device_train_batch_size=prof.dpo_batch,
        per_device_eval_batch_size=prof.dpo_batch,
        gradient_accumulation_steps=prof.dpo_grad_accum,
        learning_rate=prof.dpo_lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=prof.bf16,
        eval_strategy="steps",
        eval_steps=0.5,
        save_strategy="no",
        logging_steps=5,
        report_to=report_to(),
        run_name="roboscript-dpo",
        seed=42,
        use_cpu=a.profile == "local",
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=val,
                         processing_class=tok, peft_config=lora_config(prof))
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    write_log_history(trainer, Path(a.out))
    health = dpo_health(trainer.state.log_history)
    (Path(a.out) / "dpo_health.json").write_text(json.dumps(health, indent=2))
    health["train_file"] = a.train_file
    health["loss_types"] = prof.dpo_loss_types
    health["lr"] = prof.dpo_lr
    health["train_pairs"] = len(train)
    print("DPO health:", json.dumps(health, indent=2))

    if prof.load_in_4bit:
        # Never merge into 4-bit weights: reload the SFT model in bf16 and merge there.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(sft_dir, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, a.out).merge_and_unload()
    else:
        merged = trainer.model.merge_and_unload()
    merged_dir = OUT_DIR / "roboscript-dpo-merged"
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"saved DPO adapter -> {a.out} and merged model -> {merged_dir}")

    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub, token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
