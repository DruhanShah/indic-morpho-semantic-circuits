from pathlib import Path

import hydra
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

from utils import init_environ, set_seed, load_data, filter_empty_texts


@hydra.main(version_base=None, config_path="conf", config_name="training")
def train_tokenizer(cfg):
    init_environ(cfg.paths.assets_dir)
    set_seed(cfg.seed)

    output_dir = Path(cfg.paths.output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {cfg.data.dataset_name} (language: {cfg.language})")
    dataset = load_data(cfg.data.dataset_name, cfg.data.val_split)
    
    dataset = filter_empty_texts(dataset)
    print(f"Filtered dataset size: {len(dataset)}")
    
    # Take subset for tokenizer training
    train_samples = min(cfg.tokenizer.train_samples, len(dataset))
    dataset = dataset.select(range(train_samples))
    print(f"Training tokenizer on {train_samples} samples")

    # Train BPE tokenizer
    text_iterator = (ex["text"] for ex in dataset)
    
    bpe_tokenizer = ByteLevelBPETokenizer()
    bpe_tokenizer.train_from_iterator(
        text_iterator,
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    bpe_tokenizer.save_model(str(tokenizer_dir))
    print(f"Saved BPE tokenizer to {tokenizer_dir}")

    # Convert to HuggingFace format
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
    })
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"Saved HuggingFace tokenizer to {tokenizer_dir}")
    print(f"Vocabulary size: {len(tokenizer)}")


if __name__ == "__main__":
    train_tokenizer()
