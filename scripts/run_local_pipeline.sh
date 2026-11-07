#!/usr/bin/env bash
# End-to-end local (CPU, no API keys, no downloads) run of the full RoboScript pipeline.
#   bash scripts/run_local_pipeline.sh            # ~1.5-2 h on a 4-core CPU
#   FAST=1 bash scripts/run_local_pipeline.sh     # ~10 min smoke run (1 epoch each, small eval)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
EP=""; LIMIT=""
if [[ "${FAST:-0}" == "1" ]]; then EP="--epochs 1"; LIMIT="--limit 30"; fi

python -m roboscript.data.procedural --per-category 250          # Session A (no-API)
python -m roboscript.data.pipeline                                # Session B, dataset prep
python -m roboscript.eval.golden                                  # fixed eval sets
python -m roboscript.training.bootstrap                           # local tokenizer + tiny Llamas
python -m roboscript.training.sft_trainer --profile local $EP 2>&1 | tee logs/sft.log | grep -E "eval_loss|saved" || true
python -m roboscript.training.dpo_trainer --profile local          2>&1 | tee logs/dpo.log | grep -E "health|margins|saved" || true
python -m roboscript.distillation.distill --profile local $EP      --out outputs/roboscript-edge       2>&1 | tee logs/distill.log | grep -E "mode|eval_loss" || true
python -m roboscript.distillation.distill --profile local $EP --alpha 1.0 --out outputs/roboscript-edge-nokd 2>&1 | tee logs/distill_nokd.log | grep -E "mode|eval_loss" || true
python -m roboscript.eval.evaluator $LIMIT                         2>&1 | tee logs/eval.log
python -m pytest -q
