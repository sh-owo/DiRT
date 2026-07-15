from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import sentencepiece as spm


DATA_DIR = Path("data")
TOKENIZER_MODEL = DATA_DIR / "tokenizer"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(repo_id="t5-small", filename="spiece.model")
    shutil.copy2(model_path, str(TOKENIZER_MODEL))

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    print(f"Tokenizer saved to {TOKENIZER_MODEL}")
    print(f"  vocab_size = {sp.vocab_size()}")
    print(f"  eos_id     = {sp.eos_id()}")
    print(f"  pad_id     = {sp.pad_id()}")


if __name__ == "__main__":
    main()
