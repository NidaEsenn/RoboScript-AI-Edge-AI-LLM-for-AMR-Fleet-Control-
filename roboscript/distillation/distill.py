"""Session E, part 1 — knowledge distillation for the edge model (PRD Section 6.6).

    python -m roboscript.distillation.distill --profile local
    python -m roboscript.distillation.distill --profile gpu     # Llama-8B -> Phi-3-mini

Two modes, chosen automatically:

* ``logit`` (token-level KD) when teacher and student share a vocabulary:
      loss = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(teacher_T || student_T)
  computed **only on assistant tokens** and averaged per token. (The PRD sketch
  used ``F.kl_div(..., reduction="batchmean")`` over every position, which divides
  by batch size rather than token count and also distils the prompt and padding.)

* ``sequence`` (sequence-level KD, Kim & Rush 2016) when the vocabularies differ.
  This is the case for the PRD's own pairing: Llama-3.1 (128k-token vocab) and
  Phi-3-mini (32k-token vocab) do not share token ids, so a token-wise KL between
  their logits is undefined. The teacher instead generates answers for the training
  prompts, only schema-valid *and* safety-checked answers are kept, and the student
  is fine-tuned on them (optionally mixed with the gold labels).

Output: outputs/roboscript-edge/ + distill_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from roboscript.data.pipeline import tokenize_and_mask
from roboscript.training.common import (DATA_DIR, OUT_DIR, get_profile, load_json,
                                        load_tokenizer, report_to, write_log_history)


def distillation_loss(student_logits, teacher_logits, labels, alpha: float = 0.5,
                      temperature: float = 4.0):
    """Masked token-level KD loss. Shapes: logits [B, T, V], labels [B, T]."""
    s = student_logits[:, :-1].float()
    t = teacher_logits[:, :-1].float()
    y = labels[:, 1:]
    mask = y != -100
    ce = F.cross_entropy(s.reshape(-1, s.size(-1)), y.reshape(-1), ignore_index=-100)
    if alpha >= 1.0:
        return ce, ce.detach(), torch.zeros_like(ce)
    log_p_s = F.log_softmax(s / temperature, dim=-1)
    log_p_t = F.log_softmax(t / temperature, dim=-1)
    kl_tok = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1)          # [B, T-1]
    kl = (kl_tok * mask).sum() / mask.sum().clamp(min=1) * temperature ** 2
    return alpha * ce + (1 - alpha) * kl, ce.detach(), kl.detach()


def same_vocab(tok_a, tok_b) -> bool:
    return tok_a.get_vocab() == tok_b.get_vocab()


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {"input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in feats]),
                "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in feats]),
                "labels": torch.tensor([pad(f["labels"], -100) for f in feats])}


def make_trainer_cls():
    from transformers import Trainer

    class KDTrainer(Trainer):
        def __init__(self, *args, teacher=None, alpha=0.5, temperature=4.0, **kw):
            super().__init__(*args, **kw)
            self.teacher, self.alpha, self.temperature = teacher, alpha, temperature
            self._kd_log = {"ce": [], "kl": []}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            if self.teacher is None:
                loss, ce, kl = distillation_loss(out.logits, out.logits, labels, alpha=1.0)
            else:
                with torch.no_grad():
                    t_logits = self.teacher(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"]).logits
                loss, ce, kl = distillation_loss(out.logits, t_logits, labels,
                                                 self.alpha, self.temperature)
            if model.training:
                self._kd_log["ce"].append(ce.item())
                self._kd_log["kl"].append(kl.item())
            return (loss, out) if return_outputs else loss

        def log(self, logs, *args, **kw):
            if self._kd_log["ce"]:
                logs["ce"] = sum(self._kd_log["ce"]) / len(self._kd_log["ce"])
                logs["kl"] = sum(self._kd_log["kl"]) / len(self._kd_log["kl"])
                self._kd_log = {"ce": [], "kl": []}
            super().log(logs, *args, **kw)

    return KDTrainer


def teacher_relabel(teacher_dir: str, rows: list[dict], batch_size: int) -> tuple[list[dict], dict]:
    """Sequence-level KD data: replace gold answers with safe, valid teacher answers."""
    from roboscript.eval.generate import generate, load_for_inference, parse_json
    from roboscript.safety import check
    model, tok = load_for_inference(teacher_dir)
    scenarios = [json.loads(r["messages"][1]["content"]) for r in rows]
    outs = generate(model, tok, scenarios, batch_size=batch_size)
    kept, stats = [], {"teacher_valid_and_safe": 0, "fallback_to_gold": 0}
    for r, s, (text, _) in zip(rows, scenarios, outs):
        obj = parse_json(text)
        v = check(s, obj)
        if obj is not None and v.valid_schema and v.safe:
            msgs = r["messages"][:-1] + [{"role": "assistant", "content": json.dumps(obj)}]
            kept.append({**r, "messages": msgs})
            stats["teacher_valid_and_safe"] += 1
        else:
            kept.append(r)
            stats["fallback_to_gold"] += 1
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local", choices=["local", "gpu"])
    ap.add_argument("--teacher", default=str(OUT_DIR / "roboscript-dpo-merged"))
    ap.add_argument("--student", default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "logit", "sequence"])
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None,
                    help="CE weight; 1.0 = plain SFT of the student (no-KD ablation)")
    ap.add_argument("--out", default=str(OUT_DIR / "roboscript-edge"))
    ap.add_argument("--push-to-hub", default=None, help="{HF_USERNAME}/roboscript-phi3-edge")
    a = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, TrainingArguments

    prof = get_profile(a.profile)
    student_id = a.student or prof.student_model
    s_tok = load_tokenizer(student_id)
    t_tok = load_tokenizer(a.teacher)
    mode = a.mode if a.mode != "auto" else ("logit" if same_vocab(s_tok, t_tok) else "sequence")
    print(f"distillation mode: {mode}")

    train_rows = load_json(DATA_DIR / "train.json")
    val_rows = load_json(DATA_DIR / "val.json")
    alpha = prof.distill_alpha if a.alpha is None else a.alpha
    report: dict = {"mode": mode, "teacher": a.teacher, "student": student_id}
    teacher = None
    if alpha >= 1.0:
        report["mode"] = mode = "none (student SFT on gold labels)"
    elif mode == "logit":
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        train_rows, stats = teacher_relabel(a.teacher, train_rows, batch_size=16)
        report.update(stats)

    def encode(rows):
        feats = []
        for r in rows:
            x = tokenize_and_mask(s_tok, r)
            if len(x["input_ids"]) <= prof.max_length:
                feats.append({k: x[k] for k in ("input_ids", "attention_mask", "labels")})
        return Dataset.from_list(feats)

    dtype = torch.bfloat16 if prof.bf16 else torch.float32
    student = AutoModelForCausalLM.from_pretrained(student_id, dtype=dtype)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs or prof.distill_epochs,
        per_device_train_batch_size=prof.sft_batch, per_device_eval_batch_size=prof.sft_batch,
        gradient_accumulation_steps=prof.sft_grad_accum, learning_rate=prof.distill_lr,
        lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.01, bf16=prof.bf16,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", logging_steps=10,
        report_to=report_to(), run_name="roboscript-distill", seed=42,
        remove_unused_columns=False, use_cpu=a.profile == "local")
    trainer = make_trainer_cls()(
        model=student, args=args, train_dataset=encode(train_rows), eval_dataset=encode(val_rows),
        data_collator=Collator(s_tok.pad_token_id), teacher=teacher,
        alpha=alpha, temperature=prof.distill_temperature)
    trainer.train()
    trainer.save_model(a.out)
    s_tok.save_pretrained(a.out)
    write_log_history(trainer, Path(a.out))
    n_s = sum(p.numel() for p in student.parameters())
    report.update({"student_params": n_s,
                   "teacher_params": sum(p.numel() for p in teacher.parameters()) if teacher else None,
                   "alpha": alpha, "temperature": prof.distill_temperature})
    (Path(a.out) / "distill_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub, token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
