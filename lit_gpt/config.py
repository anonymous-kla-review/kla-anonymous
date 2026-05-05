# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    local_window: int = -1
    mlp: bool = True
    full_per_layer: int = 1000000
    mb_per_layer: int = -1
    ret_per_layer: int = -1
    gla_per_layer: int = -1
    nope: bool = False
    mamba: bool = False
    sc_attn: bool = False
    rms_norm: bool= True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    mamba_init: bool = False
    attn_layer_pos: str = None
    gated_delta_per_layer: int = -1
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1
    qk_norm: str = 'l2'
    use_mamba_gate: bool = True
    seq_factor_mode: str = 'none'
    gate_mode: str = 'dual'
    expand_k: float = 0.75
    expand_v: float = 1.5
    learned_norm_init: float = 1.0

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)


configs=[]

GatedDeltaNet = [
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
    ),
    dict(
        org="NVIDIA",
        name="DeltaNet_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'softmax',
        use_mamba_gate = False,
    ),
    dict(
        org="NVIDIA",
        name="Mamba2_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mamba=True,
        n_layer=48,
        n_head=8,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="GptNeoxMLP",
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="Longhorn_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'longhorn',
    ),
    dict(
        org="NVIDIA",
        name="GLA_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gla_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="Kaczmarz_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'kaczmarz',
    ),
    dict(
        org="NVIDIA",
        name="RelaxedKaczmarz_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'relaxed_kaczmarz',
    ),
    dict(
        org="NVIDIA",
        name="RelaxedKaczmarzQNorm_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=11,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'relaxed_kaczmarz_q_norm',
    ),
        dict(
        org="NVIDIA",
        name="RelaxedKaczmarzQNorm_H1_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'relaxed_kaczmarz_q_norm',
    ),
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_H1_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
    ),
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_1.3B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=16,
        n_head=16,
        n_embd=2400,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5888,
        local_window = 2048,
        mamba_init = True,
    ),
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_H1_1.3B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=2,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
    ),
    dict(
        org="NVIDIA",
        name="Longhorn_H1_0.4B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gated_delta_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'longhorn',
    ),
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window = 2048,
        mamba_init = True,
    ),
    dict(
        org="NVIDIA",
        name="GLA_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        gla_per_layer=1,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="Mamba2_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        mamba=True,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="GptNeoxMLP",
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="Longhorn_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'longhorn',
    ),
    dict(
        org="NVIDIA",
        name="Kaczmarz_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'kaczmarz',
    ),
    dict(
        org="NVIDIA",
        name="RelaxedKaczmarz_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'relaxed_kaczmarz',
    ),
    dict(
        org="NVIDIA",
        name="RelaxedKaczmarzQNorm_MQAR",
        block_size=1024,
        vocab_size=8192,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window = 2048,
        mamba_init = True,
        qk_norm = 'relaxed_kaczmarz_q_norm',
    ),
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_Palindrome",
        block_size=8192,
        vocab_size=128,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=2,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window=8192,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="Mamba2_Palindrome",
        block_size=8192,
        vocab_size=128,
        padding_multiple=64,
        mamba=True,
        n_layer=2,
        n_head=2,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="GptNeoxMLP",
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="RelaxedKaczmarzQNorm_Palindrome",
        block_size=8192,
        vocab_size=128,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=2,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window=8192,
        mamba_init=True,
        qk_norm='relaxed_kaczmarz_q_norm',
    ),
    dict(
        org="NVIDIA",
        name="GatedDeltaNet_Stack",
        block_size=2048,
        vocab_size=128,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=2,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="Mamba2_Stack",
        block_size=2048,
        vocab_size=128,
        padding_multiple=64,
        mamba=True,
        n_layer=2,
        n_head=2,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="GptNeoxMLP",
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="RelaxedKaczmarzQNorm_Stack",
        block_size=2048,
        vocab_size=128,
        padding_multiple=64,
        gated_delta_per_layer=1,
        n_layer=2,
        n_head=2,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        local_window=2048,
        mamba_init=True,
        qk_norm='relaxed_kaczmarz_q_norm',
    ),
]
configs.extend(GatedDeltaNet)

name_to_config = {config["name"]: config for config in configs}
