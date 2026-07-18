import flax.linen as nn
import jax
import jax.numpy as jnp

from dirt.models.config import ModelConfig
from dirt.models.blocks.propose import ProposeBlock
from dirt.models.blocks.review import ReviewBlock

class DirtLayer(nn.Module):
    cfg: ModelConfig
    dtype: jnp.dtype

    def setup(self) -> None:
        self.propose_block = ProposeBlock(cfg=self.cfg, dtype=self.dtype)
        self.review_block = ReviewBlock(cfg=self.cfg, dtype=self.dtype)

    def __call__(
        self,
        z_L: jnp.ndarray,
        positions: jnp.ndarray,
        sincos: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        new = self.propose_block(z_L, positions, sincos)
        z_L, delta_v_l2, gate_mean, review_l2 = self.review_block(z_L, new, positions, sincos)
        metrics = {
            "delta_v": delta_v_l2,
            "gate": gate_mean,
            "review": review_l2,
        }
        return z_L, metrics