# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Megatron support for Llama checkpoints with attention_bias=True (the re-aliased openPangu-7B).

HF ``attention_bias=True`` puts a bias on q/k/v AND o_proj. Megatron-Core has no o_proj-only switch:
``add_qkv_bias`` covers the fused qkv and ``add_bias_linear`` is one flag for linear_proj (o_proj)
plus both MLP projections. The legacy (non-mbridge) path therefore needs four cooperating pieces,
each pinned here on CPU:

* config converter: ``add_bias_linear`` derived from the HF flags (and unchanged for Qwen);
* weight converter: o_proj.bias synced to vLLM, the extra MLP biases NOT handed over;
* ``freeze_absent_mlp_biases``: those MLP biases frozen at their zero init, o_proj.bias trainable;
* loader + saver: o_proj.bias loaded from and exported to HF checkpoints (source tripwire; the
  distributed broadcast itself is exercised by the 3+3 Megatron smoke on GPU).

Run: pytest tests/models/test_mcore_openpangu_bias_on_cpu.py -q
"""

import os
import unittest

import torch
from torch import nn


def _llama_config(**overrides):
    from transformers import LlamaConfig

    cfg = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )
    cfg.update(overrides)
    config = LlamaConfig(**cfg)
    config.architectures = ["LlamaForCausalLM"]
    return config


def _pangu_config():
    """The re-aliased openPangu shape: Llama with attention bias, no MLP bias."""
    return _llama_config(attention_bias=True, mlp_bias=False)


def _qwen3_config():
    from transformers import Qwen3Config

    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    return config


def _qwen2_config():
    from transformers import Qwen2Config

    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
    )
    config.architectures = ["Qwen2ForCausalLM"]
    return config


# ---------------------------------------------------------------------------- config converter


class _SingleRankMegatron:
    """A 1-rank gloo process group + Megatron parallel state, enough for the config converter."""

    initialized = False

    @classmethod
    def ensure(cls):
        if cls.initialized:
            return
        try:
            import megatron.core.parallel_state as mpu
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"megatron not importable: {e}") from e
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29517")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        cls.initialized = True


class TestConfigConverterBias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _SingleRankMegatron.ensure()
        from verl.models.mcore.config_converter import hf_to_mcore_config_dense

        cls.convert = staticmethod(hf_to_mcore_config_dense)

    def test_attention_bias_turns_on_add_bias_linear_and_qkv_bias(self):
        """The only way Megatron gives linear_proj (o_proj) a bias."""
        tf = self.convert(_pangu_config(), torch.bfloat16)
        self.assertTrue(tf.add_bias_linear)
        self.assertTrue(tf.add_qkv_bias)

    def test_mlp_bias_alone_also_turns_on_add_bias_linear(self):
        tf = self.convert(_llama_config(attention_bias=False, mlp_bias=True), torch.bfloat16)
        self.assertTrue(tf.add_bias_linear)
        self.assertFalse(tf.add_qkv_bias)

    def test_plain_llama_keeps_both_off(self):
        tf = self.convert(_llama_config(), torch.bfloat16)
        self.assertFalse(tf.add_bias_linear)
        self.assertFalse(tf.add_qkv_bias)

    def test_qwen3_is_unchanged(self):
        """The running Qwen3-8B arms must build the same Megatron config as before this change."""
        tf = self.convert(_qwen3_config(), torch.bfloat16)
        self.assertFalse(tf.add_bias_linear)
        self.assertFalse(tf.add_qkv_bias)
        self.assertTrue(tf.qk_layernorm)

    def test_qwen2_keeps_qkv_bias_only(self):
        tf = self.convert(_qwen2_config(), torch.bfloat16)
        self.assertFalse(tf.add_bias_linear)
        self.assertTrue(tf.add_qkv_bias)
        self.assertFalse(tf.qk_layernorm)

    def test_override_still_wins(self):
        """override_transformer_config is applied after the derivation, as for every other key."""
        tf = self.convert(_pangu_config(), torch.bfloat16, add_bias_linear=False)
        self.assertFalse(tf.add_bias_linear)


# ---------------------------------------------------------------------------- weight converter


class TestDenseWeightConverterBias(unittest.TestCase):
    def setUp(self):
        from verl.models.mcore.weight_converter import McoreToHFWeightConverterDense

        self.conv = McoreToHFWeightConverterDense(_pangu_config(), mcore_config=None)

    def test_o_proj_bias_is_synced_under_its_hf_name(self):
        t = torch.zeros(64)
        names, params = self.conv.convert_param("decoder.layers.3.self_attention.linear_proj.bias", [t])
        self.assertEqual(names, ["model.layers.3.self_attn.o_proj.bias"])
        self.assertIs(params[0], t)

    def test_o_proj_weight_still_maps(self):
        names, _ = self.conv.convert_param("decoder.layers.3.self_attention.linear_proj.weight", [torch.zeros(64, 64)])
        self.assertEqual(names, ["model.layers.3.self_attn.o_proj.weight"])

    def test_mlp_biases_are_not_handed_to_vllm(self):
        """HF mlp_bias=False: vLLM has no such tensors; the frozen zeros must be skipped, not sent."""
        for name in ("decoder.layers.0.mlp.linear_fc1.bias", "decoder.layers.0.mlp.linear_fc2.bias"):
            with self.subTest(name=name):
                names, params = self.conv.convert_param(name, [torch.zeros(128), torch.zeros(128)])
                self.assertEqual((names, params), ([], []))

    def test_skipped_params_yield_nothing_in_the_generator_contract(self):
        """per_tensor_generator does ``zip(names, params, strict=True)``; empty lists are a no-op."""
        names, params = self.conv.convert_param("decoder.layers.0.mlp.linear_fc2.bias", [torch.zeros(64)])
        self.assertEqual(list(zip(names, params, strict=True)), [])

    def test_existing_mappings_are_untouched(self):
        m = "decoder.layers.1."
        h = "model.layers.1."
        cases = {
            m + "self_attention.linear_qkv.weight": (
                3,
                [h + "self_attn.q_proj.weight", h + "self_attn.k_proj.weight", h + "self_attn.v_proj.weight"],
            ),
            m + "self_attention.linear_qkv.bias": (
                3,
                [h + "self_attn.q_proj.bias", h + "self_attn.k_proj.bias", h + "self_attn.v_proj.bias"],
            ),
            m + "self_attention.linear_qkv.layer_norm_weight": (1, [h + "input_layernorm.weight"]),
            m + "mlp.linear_fc1.weight": (2, [h + "mlp.gate_proj.weight", h + "mlp.up_proj.weight"]),
            m + "mlp.linear_fc1.layer_norm_weight": (1, [h + "post_attention_layernorm.weight"]),
            m + "mlp.linear_fc2.weight": (1, [h + "mlp.down_proj.weight"]),
            "embedding.word_embeddings.weight": (1, ["model.embed_tokens.weight"]),
            "decoder.final_layernorm.weight": (1, ["model.norm.weight"]),
            "output_layer.weight": (1, ["lm_head.weight"]),
        }
        for name, (n_params, expected) in cases.items():
            with self.subTest(name=name):
                names, params = self.conv.convert_param(name, [torch.zeros(2)] * n_params)
                self.assertEqual(names, expected)
                self.assertEqual(len(params), n_params)

    def test_unknown_names_still_raise(self):
        with self.assertRaises(NotImplementedError):
            self.conv.convert_param("decoder.layers.0.mlp.something_new.weight", [torch.zeros(2)])
        with self.assertRaises(NotImplementedError):
            self.conv.convert_param("decoder.layers.0.self_attention.something_new.bias", [torch.zeros(2)])


# ---------------------------------------------------------------------------- freeze helper


class _Linear(nn.Module):
    def __init__(self, out_features, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, 8))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None


class _Layer(nn.Module):
    def __init__(self, mlp_bias=True, proj_bias=True):
        super().__init__()
        self.self_attention = nn.Module()
        self.self_attention.linear_qkv = _Linear(24, bias=True)
        self.self_attention.linear_proj = _Linear(8, bias=proj_bias)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = _Linear(32, bias=mlp_bias)
        self.mlp.linear_fc2 = _Linear(8, bias=mlp_bias)


class _FakeGPT(nn.Module):
    """The attribute shape freeze_absent_mlp_biases walks: model.decoder.layers[i].mlp.linear_fc{1,2}.bias."""

    def __init__(self, n_layers=3, **layer_kwargs):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(_Layer(**layer_kwargs) for _ in range(n_layers))


class _TF:
    def __init__(self, add_bias_linear):
        self.add_bias_linear = add_bias_linear


class TestFreezeAbsentMlpBiases(unittest.TestCase):
    def setUp(self):
        from verl.models.mcore.model_initializer import freeze_absent_mlp_biases

        self.freeze = staticmethod(freeze_absent_mlp_biases)

    def test_freezes_exactly_the_mlp_biases(self):
        model = _FakeGPT(n_layers=3)
        frozen = self.freeze(model, _TF(True), _pangu_config())
        self.assertEqual(
            frozen,
            [f"decoder.layers.{i}.mlp.{a}.bias" for i in range(3) for a in ("linear_fc1", "linear_fc2")],
        )
        for name, p in model.named_parameters():
            with self.subTest(param=name):
                self.assertEqual(p.requires_grad, not name.endswith(("linear_fc1.bias", "linear_fc2.bias")))

    def test_o_proj_bias_and_all_weights_stay_trainable(self):
        model = _FakeGPT()
        self.freeze(model, _TF(True), _pangu_config())
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertIn("decoder.layers.0.self_attention.linear_proj.bias", trainable)
        self.assertIn("decoder.layers.0.self_attention.linear_qkv.bias", trainable)
        self.assertIn("decoder.layers.0.mlp.linear_fc1.weight", trainable)
        self.assertIn("decoder.layers.0.mlp.linear_fc2.weight", trainable)

    def test_frozen_biases_get_no_grad_and_stay_zero(self):
        """What DDP/optimizer rely on: no grad is produced, so the zeros can never move."""
        model = _FakeGPT(n_layers=1)
        self.freeze(model, _TF(True), _pangu_config())
        layer = model.decoder.layers[0]
        x = torch.randn(4, 8)
        y = x @ layer.mlp.linear_fc1.weight.t() + layer.mlp.linear_fc1.bias
        y.sum().backward()
        self.assertIsNone(layer.mlp.linear_fc1.bias.grad)
        self.assertIsNotNone(layer.mlp.linear_fc1.weight.grad)
        self.assertTrue(torch.equal(layer.mlp.linear_fc1.bias, torch.zeros(32)))

    def test_noop_when_hf_model_really_has_mlp_bias(self):
        model = _FakeGPT()
        self.assertEqual(self.freeze(model, _TF(True), _llama_config(attention_bias=True, mlp_bias=True)), [])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_noop_when_megatron_has_no_linear_bias(self):
        """Qwen3/ORZ builds: add_bias_linear=False, the helper must not touch anything."""
        model = _FakeGPT(mlp_bias=False)
        self.assertEqual(self.freeze(model, _TF(False), _qwen3_config()), [])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_tolerates_layers_without_bias_tensors(self):
        model = _FakeGPT(mlp_bias=False)
        self.assertEqual(self.freeze(model, _TF(True), _pangu_config()), [])

    def test_tolerates_models_without_a_decoder(self):
        self.assertEqual(self.freeze(nn.Module(), _TF(True), _pangu_config()), [])

    def test_dense_initializer_calls_the_helper(self):
        """DenseModel.initialize (used for Llama/Qwen2/Qwen3) is where the freeze happens - inside
        the model provider, before DDP wrap and optimizer construction."""
        import inspect

        from verl.models.mcore.model_initializer import DenseModel

        self.assertIn("freeze_absent_mlp_biases", inspect.getsource(DenseModel.initialize))


# ---------------------------------------------------------------------------- loader / saver tripwire


class TestLoaderAndSaverHandleOProjBias(unittest.TestCase):
    """The distributed broadcasts need a process group and a GPU model; the smoke test covers them.
    Here: the two functions must at least name the tensor, or checkpoints silently lose it."""

    def _source(self, module_name):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(module_name))

    def test_loader_loads_o_proj_bias(self):
        src = self._source("verl.models.mcore.loader")
        self.assertIn("self_attn.o_proj.bias", src, "loader would leave o_proj.bias at its init value")
        self.assertIn("linear_proj.bias", src)

    def test_saver_exports_o_proj_bias(self):
        src = self._source("verl.models.mcore.saver")
        self.assertIn(
            "self_attn.o_proj.bias",
            src,
            "hf_model checkpoints would lose o_proj.bias: vLLM refuses such a checkpoint and HF zero-inits it",
        )
        self.assertIn("add_bias_linear", src)

    def test_saver_does_not_export_the_frozen_mlp_biases(self):
        src = self._source("verl.models.mcore.saver")
        self.assertNotIn("mlp.gate_proj.bias", src)
        self.assertNotIn("mlp.down_proj.bias", src)


if __name__ == "__main__":
    unittest.main()
