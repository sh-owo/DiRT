from __future__ import annotations

import jax.numpy as jnp
import optax


def autoregressive_nll(logits: jnp.ndarray, input_ids: jnp.ndarray) -> jnp.ndarray:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    xent = optax.softmax_cross_entropy_with_integer_labels(shifted_logits, shifted_labels)
    return xent.mean()

def perplexity_from_nll(nll: float) -> float:
    return float(jnp.exp(jnp.asarray(nll, dtype=jnp.float32)))
