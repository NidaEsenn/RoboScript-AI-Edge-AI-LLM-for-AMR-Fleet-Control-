"""vLLM multi-adapter launcher (PRD Section 6.7) — corrected base model.

    python -m roboscript.serve.vllm_server            # prints + execs the command
    python -m roboscript.serve.vllm_server --dry-run  # print only

DECISION: serve the *merged SFT* checkpoint as the base and expose
  roboscript-sft -> the base itself (served under an alias via --served-model-name)
  roboscript-dpo -> the DPO LoRA adapter
WHY: the DPO adapter was trained on top of the merged SFT weights. The PRD command
  loaded both adapters on raw Llama-3.1-8B-Instruct, which would apply the DPO delta
  to a model that never saw SFT — the safety-aligned endpoint would lose the
  learned output format.
"""
from __future__ import annotations

import argparse
import os
import shlex


def build_command(base: str, dpo_adapter: str, port: int, max_lora_rank: int = 16) -> list[str]:
    return ["python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", base,
            "--served-model-name", "roboscript-sft",
            "--enable-lora",
            "--lora-modules", f"roboscript-dpo={dpo_adapter}",
            "--max-lora-rank", str(max_lora_rank),
            "--max-model-len", os.getenv("MAX_SEQ_LENGTH", "1024"),
            "--port", str(port), "--host", "0.0.0.0"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.getenv("SFT_MERGED_PATH", "outputs/roboscript-sft-merged"))
    ap.add_argument("--dpo-adapter", default=os.getenv("DPO_ADAPTER_PATH", "outputs/roboscript-dpo"))
    ap.add_argument("--port", type=int, default=int(os.getenv("VLLM_PORT", "8080")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cmd = build_command(a.base, a.dpo_adapter, a.port)
    print(shlex.join(cmd))
    if not a.dry_run:
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
