from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_blocks: int
    n_heads: int
    head_dim: int
    d_ffn: int
    max_seq_len: int
    rope_base: float
    rms_norm_eps: float
    attn_dropout: float
    dtype: str

def dtype_from_name(name: str) -> jnp.dtype:
    if name == "float32":
        return jnp.float32
    elif name == "float16":
        return jnp.float16
    elif name == "bfloat16":
        return jnp.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {name}")
    
    
