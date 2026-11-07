"""Session B, step 2 — supervised fine-tuning (PRD Section 6.3).

    python -m roboscript.training.sft_trainer --profile gpu     # QLoRA on Llama-3.1-8B
    python -m roboscript.training.sft_trainer --profile local   # CPU, tiny model from scratch

Data: data/processed/{train,val}.json (chat messages). They are fed to TRL as
conversational prompt/completion pairs, so the loss is computed on the assistant
JSON only (the same masking as pipeline.tokenize_and_mask, done by TRL with the
model's own tokenizer).

Output: outputs/roboscript-sft/ (LoRA adapter on gpu, full model on local) and,
for the gpu profile, a merged copy at outputs/roboscript-sft-merged/ that DPO uses
as its reference model.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset

from roboscript.training.common import (DATA_DIR, OUT_DIR, get_profile, load_causal_lm,
                                        load_json, load_tokenizer, lora_config, report_to,
                                        to_prompt_completion, write_log_history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local", choices=["local", "gpu"])
    ap.add_argument("--model", default=None, help="override base model path/id")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--out", default=str(OUT_DIR / "roboscript-sft"))
    ap.add_argument("--push-to-hub", default=None, help="{HF_USERNAME}/roboscript-sft-adapter")
    a = ap.parse_args()

    from trl import SFTConfig, SFTTrainer

    prof = get_profile(a.profile)
    base = a.model or prof.base_model
    tok = load_tokenizer(base)
    model = load_causal_lm(base, prof)

    train = Dataset.from_list(to_prompt_completion(load_json(DATA_DIR / "train.json")))
    val = Dataset.from_list(to_prompt_completion(load_json(DATA_DIR / "val.json")))

    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs or prof.sft_epochs,
        per_device_train_batch_size=prof.sft_batch,
        per_device_eval_batch_size=prof.sft_batch,
        gradient_accumulation_steps=prof.sft_grad_accum,
        learning_rate=prof.sft_lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        weight_decay=0.0 if prof.sft_use_lora else 0.01,
        bf16=prof.bf16,
        max_length=prof.max_length,
        completion_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to=report_to(),
        run_name="roboscript-sft",
        seed=42,
        dataloader_num_workers=0,
        use_cpu=a.profile == "local",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=val,
                         processing_class=tok,
                         peft_config=lora_config(prof) if prof.sft_use_lora else None)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    write_log_history(trainer, Path(a.out))
    print(f"saved SFT model -> {a.out}")

    if prof.sft_use_lora:
        # Merge for DPO's reference model and for serving without adapters.
        merged_dir = OUT_DIR / "roboscript-sft-merged"
        from peft import AutoPeftModelForCausalLM
        import torch
        merged = AutoPeftModelForCausalLM.from_pretrained(
            a.out, dtype=torch.bfloat16 if prof.bf16 else torch.float32).merge_and_unload()
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"saved merged SFT model -> {merged_dir}")

    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub, token=os.getenv("HF_TOKEN"))
        tok.push_to_hub(a.push_to_hub, token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
