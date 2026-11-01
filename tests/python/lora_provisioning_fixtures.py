"""Locally generated safetensors/tokenizer assets; not downloaded or represented as pretrained weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import lora_finetune as lf
import transformers_manifest as manifest
from tokenizers import Tokenizer, models


def build(root, name="approved-llama"):
    directory = Path(root) / name
    directory.mkdir()
    lf.build_base().save_pretrained(directory, safe_serialization=True)
    (directory / "generation_config.json").unlink(missing_ok=True)
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tokenizer.save(str(directory / "tokenizer.json"))
    manifest.write_manifest(str(directory), "causal-lm", "llama", "meticule-fixtures/generated-not-pretrained", "0" * 40)
    return directory


if __name__ == "__main__":
    build(sys.argv[1])
