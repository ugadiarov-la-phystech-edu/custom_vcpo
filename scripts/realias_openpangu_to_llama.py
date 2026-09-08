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

import argparse
import json
import os
import shutil
import sys

SRC_DEFAULT = "FreedomIntelligence/openPangu-Embedded-7B"

DROP_KEYS = ("auto_map",)


def realias(cfg: dict) -> dict:
    out = dict(cfg)
    for k in DROP_KEYS:
        out.pop(k, None)
    out["architectures"] = ["LlamaForCausalLM"]
    out["model_type"] = "llama"

    bias = bool(cfg.get("bias", False))
    out["attention_bias"] = bias
    out["mlp_bias"] = False

    for key, llama_default in (("rms_norm_eps", 1e-6), ("pad_token_id", None), ("attention_dropout", 0.0)):
        if key not in cfg:
            raise SystemExit(
                f"config.json lacks {key!r}; LlamaConfig would default it to {llama_default!r}, "
                f"which is not what PanguEmbeddedConfig uses. Refusing to guess."
            )
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

    tok_cfg = os.path.join(args.out, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        has_map = "auto_map" in json.load(open(tok_cfg))
        print(f"tokenizer_config.json auto_map preserved: {has_map} (trust_remote_code still required)")

    print(f"\nNext: OPENPANGU_MODEL_PATH={args.out} pytest tests/models/test_openpangu_tokenizer_contract_on_cpu.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
