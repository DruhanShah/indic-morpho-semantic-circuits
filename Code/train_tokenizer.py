from pathlib import Path
from itertools import islice

import hydra
from datasets import load_dataset
from omegaconf import OmegaConf
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

from utils import init_environ, set_seed


@hydra.main(version_base=None, config_path=".", config_name="config")
def train_tokenizer(cfg):
    init_environ(cfg.paths.assets_dir)
    set_seed(cfg.seed)

    output_dir = Path(cfg.paths.output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_dataset(**OmegaConf.to_object(cfg.data.current))

    text_iterator = (
        ex["text"]
        for ex in islice(
            raw_dataset.filter(lambda ex: ex["text"] and not ex["text"].isspace()),
            int(cfg.tokenizer.train_samples),
        )
    )

    bpe_tokenizer = ByteLevelBPETokenizer()
    bpe_tokenizer.train_from_iterator(
        text_iterator,
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    bpe_tokenizer.save_model(str(tokenizer_dir))

    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        }
    )
    tokenizer.save_pretrained(str(tokenizer_dir))


if __name__ == "__main__":
    train_tokenizer()