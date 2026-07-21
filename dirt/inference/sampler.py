from __future__ import annotations

import jax
import jax.numpy as jnp


def _top_k_filter(logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = jax.lax.top_k(logits, top_k)
    threshold = values[..., -1, None]
    return jnp.where(logits < threshold, jnp.full_like(logits, -jnp.inf), logits)


def _top_p_filter(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_idx = jnp.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_idx, axis=-1)
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)
    remove = cumulative > top_p
    remove = remove.at[..., 0].set(False)
    sorted_filtered = jnp.where(remove, jnp.full_like(sorted_logits, -jnp.inf), sorted_logits)
    inverse_idx = jnp.argsort(sorted_idx, axis=-1)
    return jnp.take_along_axis(sorted_filtered, inverse_idx, axis=-1)


def _repetition_penalty_filter(
    logits: jnp.ndarray,
    prev_tokens: list[int],
    penalty: float,
) -> jnp.ndarray:
    if penalty <= 1.0 or not prev_tokens:
        return logits
    unique = jnp.array(list(set(prev_tokens)), dtype=jnp.int32)
    prev = logits[unique]
    penalized = jnp.where(prev > 0, prev / penalty, prev * penalty)
    return logits.at[unique].set(penalized)


def sample_next_token(
    rng: jnp.ndarray,
    logits: jnp.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float = 1.0,
    prev_tokens: list[int] | None = None,
) -> jnp.ndarray:
    logits = _repetition_penalty_filter(logits, prev_tokens or [], repetition_penalty)
    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1)
    temp = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), 1e-5)
    scaled = logits / temp
    filtered = _top_k_filter(scaled, int(top_k))
    filtered = _top_p_filter(filtered, float(top_p))
    return jax.random.categorical(rng, filtered, axis=-1)
