"""Create the local-profile base model and student from scratch (no downloads).

    python -m roboscript.training.bootstrap

Writes:
  outputs/local-base/           tokenizer + randomly initialised Llama (~6M params)
  outputs/local-student-init/   same tokenizer + smaller Llama (~2M params)

The tokenizer is a byte-level BPE trained on the *training split only* (so the
test set never influences it), wrapped as a PreTrainedTokenizerFast with a
Llama-3-style chat template. Teacher and student share it, which is what makes
token-level (logit) distillation well-defined — see distill.py for why the PRD's
Llama-8B -> Phi-3 pairing needs a different approach.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from roboscript.data.pipeline import _ids
from roboscript.training.common import (DATA_DIR, LOCAL_CHAT_TEMPLATE, OUT_DIR,
                                        SPECIAL_TOKENS, load_json)


def train_tokenizer(texts: list[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
        pad_token="<|pad|>", unk_token="<|unk|>")
    tok.chat_template = LOCAL_CHAT_TEMPLATE
    return tok


def make_llama(tok, hidden: int, layers: int, heads: int, kv_heads: int, inter: int, seed: int):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads,
                      intermediate_size=inter, max_position_embeddings=1024,
                      tie_word_embeddings=True, bos_token_id=tok.bos_token_id,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_json(DATA_DIR / "train.json")
    texts = [m["content"] for r in rows for m in r["messages"]]
    tok = train_tokenizer(texts, a.vocab_size)
    lens = [len(_ids(tok.apply_chat_template(r["messages"], tokenize=True))) for r in rows[:200]]
    print(f"tokenizer vocab={len(tok)}  example lengths: mean={sum(lens)/len(lens):.0f} max={max(lens)}")

    for name, dims in {"local-base": (256, 6, 8, 4, 768),
                       "local-student-init": (160, 3, 4, 2, 480)}.items():
        model = make_llama(tok, *dims, seed=a.seed)
        out = OUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        n = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n/1e6:.2f}M params -> {out}")


if __name__ == "__main__":
    main()
