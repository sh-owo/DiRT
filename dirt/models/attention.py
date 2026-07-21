import jax
import jax.numpy as jnp
from typing import Optional

from dirt.models.common import causal_mask

@jax.remat
def scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    padding_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if padding_mask is not None:
        scores = scores + padding_mask
    if mask is not None:
        scores = scores + mask
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    padding_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    seq_len = q.shape[-2]
    mask = causal_mask(seq_len, q.dtype)[None, None, :, :]
    return scaled_dot_product_attention(q, k, v, mask, padding_mask)

def cross_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    padding_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    return scaled_dot_product_attention(q, k, v, None, padding_mask)