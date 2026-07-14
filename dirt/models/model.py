import flax.linen as nn
import jax.numpy as jnp

from dirt.models.config import ModelConfig, dtype_from_name
from dirt.models.layers import DirtLayer
from dirt.models.common import RMSNorm, rope_tables

class DiRTModel(nn.Module):
    cfg: ModelConfig

    def setup(self) -> None:
        self.dtype = dtype_from_name(self.cfg.dtype)
        self.token_embedding = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )
        self.blocks = [DirtLayer(cfg=self.cfg, dtype=self.dtype, name=f"block_{i}") for i in range(self.cfg.n_blocks)]
        self.final_norm = RMSNorm(self.cfg.d_model, eps=self.cfg.rms_norm_eps, dtype=self.dtype)
        self.sincos = rope_tables(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_base, self.dtype)

    def __call__(self, input_ids: jnp.ndarray, train: bool) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        batch, seq_len = input_ids.shape
        x = self.token_embedding(input_ids).astype(self.dtype)
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        all_metrics = []

        for block in self.blocks:
            x, metrics = block(x, positions, self.sincos)
            all_metrics.append(metrics)

        x = self.final_norm(x)
        embedding = self.token_embedding.embedding
        logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embedding.astype(jnp.float32))

        aggregate = self._aggregate_metrics(all_metrics)
        all_metrics.append(aggregate)

        return logits, all_metrics
    
    def _aggregate_metrics(self, all_metrics: list[dict[str, jnp.ndarray]]) -> dict[str, jnp.ndarray]:
        stacked = {k: jnp.stack([m[k] for m in all_metrics]) for k in ["delta_v", "gate", "review"]}
        return {
            "avg_delta_v": jnp.mean(stacked["delta_v"]),
            "avg_gate": jnp.mean(stacked["gate"]),
            "avg_review": jnp.mean(stacked["review"]),
        }