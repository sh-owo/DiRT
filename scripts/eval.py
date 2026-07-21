import argparse
from pathlib import Path

from dirt.eval import mmlu_pro
from dirt.inference.common import load_model
from dirt.models.config import ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="dirt", choices=["dirt", "hf"])
    parser.add_argument("--config", default="dirt_700m", help="Model config name (for dirt)")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model.safetensors")
    parser.add_argument("--tokenizer-model", type=Path, required=True, help="Path to SentencePiece tokenizer model")
    parser.add_argument("--output-dir", type=Path, default=None, help="Results directory (default: model_path parent)")
    args = parser.parse_args()

    if args.model_type == "dirt":
        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).resolve().parent.parent / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose("config", overrides=[f"model={args.config}"])

        m = cfg.model
        model_cfg = ModelConfig(
            name=m.name, vocab_size=m.vocab_size, d_model=m.d_model,
            n_blocks=m.n_blocks, n_heads=m.n_heads, head_dim=m.head_dim,
            d_ffn=m.d_ffn, max_seq_len=m.max_seq_len, rope_base=m.rope_base,
            rms_norm_eps=m.rms_norm_eps, attn_dropout=m.get("attn_dropout", 0.0),
            dtype=m.dtype,
        )

        model, params = load_model(args.model_path, model_cfg)
        predict_fn = lambda x: model.apply({"params": params}, x, train=False)[0]

    elif args.model_type == "hf":
        raise NotImplementedError("HF model not supported yet")

    output_dir = args.output_dir or args.model_path.parent
    mmlu_pro(
        predict_fn=predict_fn,
        tokenizer_model=args.tokenizer_model,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
