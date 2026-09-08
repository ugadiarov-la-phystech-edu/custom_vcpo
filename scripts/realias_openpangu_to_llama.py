# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialise openPangu-Embedded-7B as a plain ``LlamaForCausalLM`` checkpoint.

openPangu ships as a ``trust_remote_code`` architecture, which costs us twice:

* ``modeling_openpangu_dense.py`` imports ``LossKwargs``, removed in transformers >= 4.54 (we run
  4.57.6), so ``AutoModelForCausalLM`` cannot even build the model;
* vLLM 0.11.0 has no ``PanguEmbeddedForCausalLM`` in its registry, so the rollout engine would have to
  fall back to ``model_impl=transformers``.

Both come from the custom modeling code, and the model does not need it: the generated modeling file
is a ``modular`` derivative of Llama whose math-carrying functions -- RMSNorm, rotary embedding,
``apply_rotary_pos_emb``, MLP, attention, decoder layer, and both forwards -- are byte-for-byte copies
of transformers' Llama. The only substantive difference is that Pangu collapses Llama's
``attention_bias``/``mlp_bias`` into a single ``bias`` flag that drives q/k/v/o only.

So we rewrite ``config.json`` to say Llama and drop its ``auto_map``. transformers then uses its
native Llama, and vLLM its native (fast, well-exercised) Llama -- which is also the weight-sync path
every other arm already uses. ``tokenizer_config.json`` keeps its own ``auto_map``, so the tokenizer
still comes from ``tokenization_openpangu.py`` and ``trust_remote_code`` is still required for it.

Weight names need no remapping: they are already ``model.layers.N.self_attn.{q,k,v,o}_proj.*`` etc.

Idempotent -- safe to re-run. Verify the result with
``OPENPANGU_MODEL_PATH=<out> pytest tests/models/test_openpangu_tokenizer_contract_on_cpu.py``.

Usage:
    source /home/jovyan/ugadiarov/custom_vcpo2/activate.sh
    python scripts/realias_openpangu_to_llama.py --out /home/jovyan/ugadiarov/models/openPangu-Embedded-7B-llama
"""

import argparse
import json
import os
import shutil
import sys

SRC_DEFAULT = "FreedomIntelligence/openPangu-Embedded-7B"

# config.json keys that select the custom modeling code. Removing auto_map entirely is what stops
# transformers importing modeling_openpangu_dense.py; the tokenizer's auto_map lives in
# tokenizer_config.json and is deliberately left alone.
DROP_KEYS = ("auto_map",)


def realias(cfg: dict) -> dict:
    """Rewrite a PanguEmbedded config in place into an equivalent Llama config."""
    out = dict(cfg)
    for k in DROP_KEYS:
        out.pop(k, None)
    out["architectures"] = ["LlamaForCausalLM"]
    out["model_type"] = "llama"

    # Pangu's single `bias` flag drives q/k/v/o only (modeling_openpangu_dense.py:223-226); its MLP
    # hard-codes bias=False. Llama splits these, and applies attention_bias to all four projections
    # including o_proj (modeling_llama.py:210-221), which is exactly the shape we need.
    bias = bool(cfg.get("bias", False))
    out["attention_bias"] = bias
    out["mlp_bias"] = False
    # `bias` is left in place on purpose: inert for transformers, and vLLM reads it directly as a
    # fallback for attention_bias (vllm/model_executor/models/llama.py:263-266).

    # These three differ from LlamaConfig's defaults, so they must survive explicitly. The checkpoint
    # already states all of them; assert rather than assume.
    for key, llama_default in (("rms_norm_eps", 1e-6), ("pad_token_id", None), ("attention_dropout", 0.0)):
        if key not in cfg:
            raise SystemExit(
                f"config.json lacks {key!r}; LlamaConfig would default it to {llama_default!r}, "
                f"which is not what PanguEmbeddedConfig uses. Refusing to guess."
            )
    # head_dim is left unset: Pangu computes hidden_size // num_attention_heads and so does
    # LlamaConfig, giving 128 either way.
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT, help="hub id or a local directory")
    ap.add_argument("--out", required=True, help="destination directory for the re-aliased checkpoint")
    ap.add_argument("--no-download", action="store_true", help="--src is already a local directory")
    args = ap.parse_args()

    if args.no_download or os.path.isdir(args.src):
        src_dir = args.src
        if os.path.abspath(src_dir) != os.path.abspath(args.out):
            os.makedirs(args.out, exist_ok=True)
            for name in os.listdir(src_dir):
                dst = os.path.join(args.out, name)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(src_dir, name), dst)
    else:
        from huggingface_hub import snapshot_download

        print(f"downloading {args.src} -> {args.out} (~16 GB, resumable)")
        snapshot_download(repo_id=args.src, local_dir=args.out)

    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))

    if cfg.get("model_type") == "llama":
        print(f"{cfg_path} is already re-aliased; nothing to do")
    else:
        backup = cfg_path + ".pangu.bak"
        if not os.path.exists(backup):
            shutil.copy2(cfg_path, backup)
            print(f"original config preserved at {backup}")
        new_cfg = realias(cfg)
        with open(cfg_path, "w") as f:
            json.dump(new_cfg, f, indent=2)
            f.write("\n")
        print(f"rewrote {cfg_path}")
        for key in ("architectures", "model_type", "attention_bias", "mlp_bias", "bias", "rms_norm_eps"):
            print(f"  {key:18s} {cfg.get(key, '<absent>')!r} -> {new_cfg.get(key, '<absent>')!r}")

    # The tokenizer must still resolve through remote code -- only the modeling entries were dropped.
    tok_cfg = os.path.join(args.out, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        has_map = "auto_map" in json.load(open(tok_cfg))
        print(f"tokenizer_config.json auto_map preserved: {has_map} (trust_remote_code still required)")

    print(f"\nNext: OPENPANGU_MODEL_PATH={args.out} pytest tests/models/test_openpangu_tokenizer_contract_on_cpu.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
