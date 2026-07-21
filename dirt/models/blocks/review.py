import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from dirt.models.config import ModelConfig
from typing import Optional

from dirt.models.common import RMSNorm, apply_rope, swiglu
from dirt.models.attention import cross_attention

def default_init():
    return nn.initializers.normal(stddev=0.02)

def out_init(n_blocks: int):
    return nn.initializers.normal(stddev=0.02 / math.sqrt(2.0 * n_blocks))

class ReviewBlock(nn.Module):
    cfg: ModelConfig
    dtype: jnp.dtype

    def setup(self) -> None:
        head_dim = self.cfg.head_dim
        self.norm_attn = RMSNorm(self.cfg.d_model, eps=self.cfg.rms_norm_eps, dtype=self.dtype)
        self.norm_ffn = RMSNorm(self.cfg.d_model, eps=self.cfg.rms_norm_eps, dtype=self.dtype)

        self.q_proj = nn.Dense(self.cfg.n_heads * head_dim, use_bias=False, kernel_init= default_init(), dtype=self.dtype, name= "q_proj")
        self.k_proj = nn.Dense(self.cfg.n_heads * head_dim, use_bias=False, kernel_init= default_init(), dtype=self.dtype, name= "k_proj")
        self.v_proj = nn.Dense(self.cfg.n_heads * head_dim, use_bias=False, kernel_init= default_init(), dtype=self.dtype, name= "v_proj")
        self.o_proj = nn.Dense(self.cfg.n_heads * head_dim, use_bias=False, kernel_init= out_init(self.cfg.n_blocks), dtype=self.dtype, name= "o_proj")

        self.gate_proj = nn.Dense(self.cfg.d_ffn, use_bias=False, kernel_init= default_init(), dtype=self.dtype, name= "gate_proj")
        self.up_proj = nn.Dense(self.cfg.d_ffn, use_bias=False, kernel_init= default_init(), dtype=self.dtype, name= "up_proj")
        self.down_proj = nn.Dense(self.cfg.d_model, use_bias=False, kernel_init= out_init(self.cfg.n_blocks), dtype=self.dtype, name= "down_proj") 

        self.prob_linear = nn.Dense(self.cfg.d_model, use_bias=False, kernel_init= out_init(self.cfg.n_blocks), dtype=self.dtype, name= "prob_linear")


    def __call__(
        self,
        z_L: jnp.ndarray,
        new: jnp.ndarray,
        positions: jnp.ndarray,
        sincos: tuple[jnp.ndarray, jnp.ndarray],
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch, seq_len, _ = z_L.shape
        head_dim = self.cfg.head_dim

        delta_v =  new - z_L

        z_L_norm = self.norm_attn(z_L)
        delta_v_norm = self.norm_attn(delta_v)
        q = self.q_proj(delta_v_norm).reshape(batch, seq_len, self.cfg.n_heads, head_dim)
        k = self.k_proj(z_L_norm).reshape(batch, seq_len, self.cfg.n_heads, head_dim)
        v = self.v_proj(z_L_norm).reshape(batch, seq_len, self.cfg.n_heads, head_dim)

        q_t = jnp.transpose(q, (0, 2, 1, 3))
        k_t = jnp.transpose(k, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))

        q_t, k_t = apply_rope(q_t, k_t, sincos, positions)
        attn_out = cross_attention(q_t, k_t, v_t, padding_mask)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq_len, self.cfg.d_model)
        attn_out = self.o_proj(attn_out)

        _delta_v = delta_v + attn_out
        ffn_norm = self.norm_ffn(_delta_v)
        _review = swiglu(ffn_norm, self.gate_proj, self.up_proj, self.down_proj)

        gate = nn.sigmoid(self.prob_linear(_review))
        review = gate * _review

        out = z_L + review

        delta_v_l2 = jnp.linalg.norm(delta_v, axis=-1)

        imp_review_l2 = jnp.linalg.norm(_review, axis=-1)
        gate_mean = jnp.mean(gate, axis=-1)
        review_l2 = jnp.linalg.norm(review, axis=-1)

        out_l2 = jnp.linalg.norm(out, axis=-1)

        return out, delta_v_l2, imp_review_l2, gate_mean, review_l2, out_l2