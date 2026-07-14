import flax.linen as nn
import jax
import jax.numpy as jnp

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def rope_tables(max_seq_len: int, dim: int, base: float, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    half = dim // 2
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    scales = jnp.arange(half, dtype=jnp.float32) / half
    inv_freq = 1.0 / (base ** scales)
    angles = positions[:, None] * inv_freq[None, :]
    sin = jnp.sin(angles).astype(dtype)
    cos = jnp.cos(angles).astype(dtype)
    return sin, cos


def apply_rope(q: jnp.ndarray, k: jnp.ndarray, sincos: tuple[jnp.ndarray, jnp.ndarray], positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    sin, cos = sincos
    sin_pos = sin[positions]
    cos_pos = cos[positions]
    for _ in range(q.ndim - 2):
        sin_pos = sin_pos[None, ...]
        cos_pos = cos_pos[None, ...]
    sin_pos = jnp.concatenate([sin_pos, sin_pos], axis=-1)
    cos_pos = jnp.concatenate([cos_pos, cos_pos], axis=-1)
    q_rot = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_rot = (k * cos_pos) + (rotate_half(k) * sin_pos)
    return q_rot, k_rot


def causal_mask(seq_len: int, dtype: jnp.dtype) -> jnp.ndarray:
    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=dtype), k=1)
    return jnp.where(mask > 0, jnp.finfo(dtype).min, 0.0)

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        x_f = x.astype(jnp.float32)
        norm = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
        x_n = x_f * jax.lax.rsqrt(norm + self.eps)
        return (x_n * scale).astype(self.dtype)

def swiglu(x: jnp.ndarray, w_gate: nn.Dense, w_up: nn.Dense, w_down: nn.Dense) -> jnp.ndarray:
    gate = jax.nn.silu(w_gate(x))
    up = w_up(x)
    return w_down(gate * up)