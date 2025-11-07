import os
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset


def init_environ(scratch: str):
    """Initialize environment variables for caching."""
    os.environ["HF_HOME"] = scratch + "/cache/hf"
    os.environ["UV_CACHE_DIR"] = scratch + "/cache/uv"


def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(dataset_name: str, language: str):
    """Load data with correct column chosen and renamed."""
    dataset = load_dataset(dataset_name, split="train")
    column_name = "original" if language == "en" else "translated"
    columns_to_remove = ["original"] if language != "en" else ["translated"]

    # Unholy column manipulation because dataset structure stinks (wadr)
    dataset = dataset.rename_column(column_name, "text")
    dataset = dataset.remove_columns(columns_to_remove)
    print(dataset)
    return dataset


def filter_empty_texts(dataset):
    """Filter out empty or whitespace-only texts."""
    return dataset.filter(lambda ex: ex["text"] and not ex["text"].isspace())


def group_texts(examples, block_size):
    """Concatenate and chunk texts into fixed-size blocks."""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])

    if total_length < block_size:
        return {k: [] for k in concatenated.keys()}

    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def split_data(dataset, val_split: float, seed: int):
    split_dataset = dataset.train_test_split(test_size=val_split, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def tokenize_and_group(train, eval, tokenizer, cfg, block_size):
    tokenized_train = train.map(
        lambda ex: tokenizer(ex["text"], truncation=False),
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
        remove_columns=["text"],
        desc="Tokenizing train",
    )

    tokenized_eval = eval.map(
        lambda ex: tokenizer(ex["text"], truncation=False),
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
        remove_columns=["text"],
        desc="Tokenizing eval",
    )

    lm_train = tokenized_train.map(
        group_texts,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
        fn_kwargs={"block_size": block_size},
        desc="Grouping train",
    ).filter(lambda ex: len(ex["input_ids"]) > 0)

    lm_eval = tokenized_eval.map(
        group_texts,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
        fn_kwargs={"block_size": block_size},
        desc="Grouping eval",
    ).filter(lambda ex: len(ex["input_ids"]) > 0)

    print(f"Train blocks: {len(lm_train)}")
    print(f"Eval blocks: {len(lm_eval)}")
    return lm_train, lm_eval


def plot_metrics(output_dir: Path, language: str):
    state_file = output_dir / "trainer_state.json"

    if not state_file.exists():
        print(f"Warning: {state_file} not found, skipping plots")
        return

    with open(state_file, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        print("Warning: No log history found in trainer_state.json")
        return

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])

    train_perplexities = [np.exp(loss) for loss in train_losses]
    eval_perplexities = [np.exp(loss) for loss in eval_losses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(train_steps, train_losses, label="Train Loss")
    if eval_steps:
        ax1.plot(eval_steps, eval_losses, label="Eval Loss")
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Training Loss ({language.upper()})",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Perplexity
    ax2.plot(train_steps, train_perplexities, label="Train Perplexity")
    if eval_steps:
        ax2.plot(eval_steps, eval_perplexities, label="Eval Perplexity")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Perplexity", fontsize=12)
    ax2.set_title(f"Training Perplexity ({language.upper()})",
                  fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / "training_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved training plots to {plot_path}")
    plt.close()
