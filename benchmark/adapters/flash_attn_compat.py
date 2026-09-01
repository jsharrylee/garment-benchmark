from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec


def install_flash_attn_compat() -> None:
    """Expose the FlashAttention API through PyTorch SDPA for unsupported Windows builds."""
    if "flash_attn" in sys.modules:
        return
    import torch
    import torch.nn.functional as functional

    def _sdpa(q, k, v, dropout_p=0.0, causal=False):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)
        return output.transpose(1, 2)

    def flash_attn_func(q, k, v, dropout_p=0.0, causal=False, **_kwargs):
        return _sdpa(q, k, v, dropout_p, causal)

    def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False, **_kwargs):
        return _sdpa(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], dropout_p, causal)

    def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, causal=False, **_kwargs):
        return _sdpa(q, kv[:, :, 0], kv[:, :, 1], dropout_p, causal)

    def _segments(cumulative):
        values = cumulative.detach().cpu().tolist()
        return zip(values[:-1], values[1:], strict=True)

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens=None, dropout_p=0.0, causal=False, **_kwargs):
        outputs = []
        for start, end in _segments(cu_seqlens):
            item = qkv[start:end].unsqueeze(0)
            outputs.append(flash_attn_qkvpacked_func(item, dropout_p, causal).squeeze(0))
        return torch.cat(outputs, dim=0)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, dropout_p=0.0, causal=False, **_kwargs):
        outputs = []
        q_segments = list(_segments(cu_seqlens_q))
        k_segments = list(_segments(cu_seqlens_k))
        for (q_start, q_end), (k_start, k_end) in zip(q_segments, k_segments, strict=True):
            outputs.append(_sdpa(q[q_start:q_end].unsqueeze(0), k[k_start:k_end].unsqueeze(0), v[k_start:k_end].unsqueeze(0), dropout_p, causal).squeeze(0))
        return torch.cat(outputs, dim=0)

    def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q=None, cu_seqlens_k=None, dropout_p=0.0, causal=False, **kwargs):
        return flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu_seqlens_q, cu_seqlens_k, dropout_p, causal, **kwargs)

    def index_first_axis(values, indices):
        return values[indices]

    def pad_input(values, indices, batch_size, sequence_length):
        flat = values.new_zeros((batch_size * sequence_length, *values.shape[1:]))
        flat[indices] = values
        return flat.reshape(batch_size, sequence_length, *values.shape[1:])

    def unpad_input(values, attention_mask):
        lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cumulative = functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
        maximum = int(lengths.max().item())
        return index_first_axis(values.reshape(-1, *values.shape[2:]), indices), indices, cumulative, maximum

    module = types.ModuleType("flash_attn")
    module.__spec__ = ModuleSpec("flash_attn", loader=None, is_package=True)
    module.flash_attn_func = flash_attn_func
    module.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    module.flash_attn_kvpacked_func = flash_attn_kvpacked_func
    module.flash_attn_varlen_func = flash_attn_varlen_func
    module.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    module.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = ModuleSpec("flash_attn.bert_padding", loader=None)
    padding.index_first_axis = index_first_axis
    padding.pad_input = pad_input
    padding.unpad_input = unpad_input
    sys.modules["flash_attn"] = module
    sys.modules["flash_attn.bert_padding"] = padding


def install_missing_image_block_v2() -> None:
    """Restore the image model block referenced by the official module but absent at its release commit."""
    import torch
    import torch.nn as nn

    import models.sparse_lightningdit_v3_cross_attn_varlen as official
    from models.rmsnorm import RMSNorm
    from models.swiglu_ffn import SwiGLUFFN

    if hasattr(official, "LightningDiTCrossAttnVarlenBlockV2"):
        return

    class LightningDiTCrossAttnVarlenBlockV2(nn.Module):
        def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            use_qknorm=False,
            use_swiglu=False,
            use_rmsnorm=False,
            wo_shift=False,
            backend="flash-attn",
            **_kwargs,
        ):
            super().__init__()
            norm = RMSNorm if use_rmsnorm else lambda size: nn.LayerNorm(size, elementwise_affine=False, eps=1e-6)
            self.norm1 = norm(hidden_size)
            self.norm2 = norm(hidden_size)
            self.norm3 = norm(hidden_size)
            self.additional_norm = norm(hidden_size)
            self.attn = official.AttentionVarlen(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, backend=backend)
            self.cross_attn = official.CrossAttentionVarlen(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, backend=backend)
            self.additional_cross_attn = official.CrossAttentionVarlen(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, backend=backend)
            mlp_hidden = int(hidden_size * mlp_ratio)
            if use_swiglu:
                self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden))
            else:
                from timm.models.vision_transformer import Mlp

                self.mlp = Mlp(hidden_size, mlp_hidden, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, (4 if wo_shift else 6) * hidden_size, bias=True))
            self.wo_shift = wo_shift

        def forward(
            self,
            x,
            c,
            text_context,
            image_context,
            cu_input_lens,
            max_input_len,
            cu_text_lens,
            max_text_len,
            cu_image_lens,
            max_image_len,
        ):
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                shift_msa, shift_mlp = None, None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            attention = self.attn(official.modulate(self.norm1(x), shift_msa, scale_msa), cu_lens=cu_input_lens, max_len=max_input_len)
            x = x + gate_msa * attention
            x = x + self.cross_attn(self.norm2(x), text_context, cu_input_lens, max_input_len, cu_text_lens, max_text_len)
            x = x + self.additional_cross_attn(self.additional_norm(x), image_context, cu_input_lens, max_input_len, cu_image_lens, max_image_len)
            x = x + gate_mlp * self.mlp(official.modulate(self.norm3(x), shift_mlp, scale_mlp))
            return x

    official.LightningDiTCrossAttnVarlenBlockV2 = LightningDiTCrossAttnVarlenBlockV2
