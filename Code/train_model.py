from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling as Collator,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import init_environ, set_seed


def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])

    if total_length < block_size:
        return {k: [] for k in concatenated.keys()}

    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


@hydra.main(version_base=None, config_path=".", config_name="config")
def train_model(cfg):
    init_environ(cfg.paths.assets_dir)
    set_seed(cfg.seed)

    output_dir = Path(cfg.paths.output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    block_size = cfg.model.n_ctx

    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer dir not found at {tokenizer_dir}. "
                              f"Please run train_tokenizer.py first.")
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))

    raw_dataset = load_dataset(**OmegaConf.to_object(cfg.data.current))

    cleaned_dataset = raw_dataset.filter(lambda ex: ex["text"] and not ex["text"].isspace())
    if cfg.data.max_train_samples > 0:
        cleaned_dataset = cleaned_dataset.take(cfg.data.max_train_samples)

    tokenized_dataset = cleaned_dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=False),
        remove_columns=["text"],
    )

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=cfg.processing.map_batch_size,
        fn_kwargs={"block_size": block_size},
    )
    lm_dataset = lm_dataset.filter(lambda ex: len(ex["input_ids"]) > 0)

    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_positions=cfg.model.n_ctx,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
    )
    model = AutoModelForCausalLM.from_config(model_config)

    data_collator = Collator(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **OmegaConf.to_container(cfg.training, resolve=True)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    train_model()