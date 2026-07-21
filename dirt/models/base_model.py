import flax.linen as nn
import jax.numpy as jnp

from dirt.models.config import ModelConfig, dtype_from_name
from dirt.models.common import RMSNorm, rope_tables
from dirt.models.blocks.propose import ProposeBlock


class BaseLayer(nn.Module):
    cfg: ModelConfig
    dtype: jnp.dtype

    def setup(self) -> None:
        self.propose_block = ProposeBlock(cfg=self.cfg, dtype=self.dtype)

    def __call__(
        self,
        z_L: jnp.ndarray,
        positions: jnp.ndarray,
        sincos: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        out = self.propose_block(z_L, positions, sincos)
        return out, {}


class BaseModel(nn.Module):
    cfg: ModelConfig

    def setup(self) -> None:
        self.dtype = dtype_from_name(self.cfg.dtype)
        self.token_embedding = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )
        self.blocks = [
            nn.remat(BaseLayer)(cfg=self.cfg, dtype=self.dtype, name=f"block_{i}")
            for i in range(self.cfg.n_blocks)
        ]
        self.final_norm = RMSNorm(self.cfg.d_model, eps=self.cfg.rms_norm_eps, dtype=self.dtype)

    def __call__(self, input_ids: jnp.ndarray, train: bool) -> tuple[jnp.ndarray, list[dict[str, jnp.ndarray]]]:
        batch, seq_len = input_ids.shape
        x = self.token_embedding(input_ids).astype(self.dtype)
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        sincos = rope_tables(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_base, self.dtype)

        all_metrics = []
        for block in self.blocks:
            x, metrics = block(x, positions, sincos)
            all_metrics.append(metrics)

        x = self.final_norm(x)
        embedding = self.token_embedding.embedding
        logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embedding.astype(jnp.float32))

        all_metrics.append({})

        return logits, all_metrics
