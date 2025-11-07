from pathlib import Path

import hydra
from omegaconf import OmegaConf
from datasets import load_from_disk
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast,
    Trainer, TrainingArguments,
)

from utils import (
    init_environ, set_seed, load_data,
    filter_empty_texts, tokenize_and_group, split_data,
    plot_metrics,
)


def preprocess_data(cfg, tokenizer):
    data_path = Path(cfg.paths.output_dir) / "data" / "processed" / cfg.language
    train_path = data_path / "train"
    eval_path = data_path / "eval"

    if train_path.exists() and eval_path.exists():
        lm_train = load_from_disk(str(train_path))
        lm_eval = load_from_disk(str(eval_path))
        print(f"Loaded train blocks: {len(lm_train)}")
        print(f"Loaded eval blocks: {len(lm_eval)}")
    else:
        dataset = load_data(cfg.data.dataset_name, cfg.language)
        dataset = filter_empty_texts(dataset)
        print(f"Filtered dataset size: {len(dataset)}")

        train_dataset, eval_dataset = split_data(dataset, cfg.data.val_split, cfg.seed)

        print("Tokenizing dataset...")
        lm_train, lm_eval = tokenize_and_group(train_dataset, eval_dataset, tokenizer,
                                               cfg.processing, cfg.model.n_positions)
        lm_train.save_to_disk(str(train_path))
        lm_eval.save_to_disk(str(eval_path))

    return lm_train, lm_eval


@hydra.main(version_base=None, config_path="conf", config_name="training")
def train_model(cfg):
    init_environ(cfg.paths.assets_dir)
    set_seed(cfg.seed)

    output_dir = Path(cfg.paths.output_dir)
    tokenizer_dir = output_dir / "tokenizer"

    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_dir}. "
            "Run train_tokenizer.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    print(f"Loaded tokenizer with vocab size: {len(tokenizer)}")

    print("Loading dataset...")
    lm_train, lm_eval = preprocess_data(cfg, tokenizer)

    print("Initializing model...")
    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **OmegaConf.to_container(cfg.model)
    )
    model = AutoModelForCausalLM.from_config(model_config)
    print(f"Model initialized with {model.num_parameters():,} parameters.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **OmegaConf.to_container(cfg.training, resolve=True),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=data_collator,
    )

    print("Training...")
    trainer.train()
    print(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    plot_metrics(output_dir, cfg.language)


if __name__ == "__main__":
    train_model()
