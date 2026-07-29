from typing import Optional

import jax
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
        self.blocks = [
            nn.remat(DirtLayer, static_argnums=(3,))(cfg=self.cfg, dtype=self.dtype, name=f"block_{i}")
            for i in range(self.cfg.n_blocks)
        ]
        self.final_norm = RMSNorm(self.cfg.d_model, eps=self.cfg.rms_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        train: bool = False,
        attention_mask: Optional[jnp.ndarray] = None,
        analysis_mode: bool = False,
    ) -> tuple[jnp.ndarray, list[dict[str, jnp.ndarray]]]:
        batch, seq_len = input_ids.shape
        x = self.token_embedding(input_ids).astype(self.dtype)
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        sincos = rope_tables(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_base, self.dtype)

        all_metrics = []
        hidden_states = [x]
        for block in self.blocks:
            x, metrics = block(x, positions, sincos, analysis_mode=analysis_mode)
            all_metrics.append(metrics)
            hidden_states.append(x)

        x = self.final_norm(x)
        embedding = self.token_embedding.embedding
        logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embedding.astype(jnp.float32))

        if analysis_mode:
            for i in range(len(hidden_states) - 1):
                all_metrics[i]["z_L_hidden"] = hidden_states[i]
            all_metrics[-1]["x_before_norm_hidden"] = hidden_states[-1]

        aggregate = self._aggregate_metrics(all_metrics)
        all_metrics.append(aggregate)

        return logits, all_metrics

    def _aggregate_metrics(self, all_metrics: list[dict[str, jnp.ndarray]]) -> dict[str, jnp.ndarray]:
        metric_keys = [k for k in all_metrics[0].keys() if not k.endswith("_raw") and not k.endswith("_hidden")]
        if not metric_keys:
            return {}
        stacked = {k: jnp.stack([m[k] for m in all_metrics]) for k in metric_keys}
        return {f"avg_{k}": jnp.mean(v) for k, v in stacked.items()}
