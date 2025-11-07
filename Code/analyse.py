import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from mech.model import get_model, list_modules
from mech.hooks import get_logits
from mech.ablation import run_ablation, run_ablation_scan
from mech.patching import run_patching, run_patching_scan
from mech.utils import show_diff


@hydra.main(version_base=None, config_path="conf", config_name="analysis")
def main(cfg: DictConfig):
    # Setup
    torch.set_grad_enabled(False)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = get_model(
        cfg.model_path,
        cfg.device,
        model_config=OmegaConf.to_container(cfg.model),
    )

    exp = cfg.experiment

    if exp.type == 'list_modules':
        print(f"Listing modules for model at: {cfg.model_path}")
        list_modules(model)

    elif exp.type == 'ablation':
        L, H = exp.target_layer, exp.target_head
        print(f"Running Ablation: L{L}H{H}")
        print(f"Prompt: '{exp.prompt}'")

        inputs = tokenizer(exp.prompt, return_tensors="pt").to(cfg.device)
        baseline_logits = get_logits(model, inputs)
        ablated_logits = run_ablation(model, inputs, L, H)

        show_diff(tokenizer, baseline_logits, "Baseline",
                  exp.target_token_idx, exp.compare_tokens)
        show_diff(tokenizer, ablated_logits, f"Ablated L{L}H{H}",
                  exp.target_token_idx, exp.compare_tokens)

    elif exp.type == 'patching':
        print(f"Running Patching: {exp.target_module}")
        print(f"Clean Prompt:     '{exp.clean_prompt}'")
        print(f"Corrupted Prompt: '{exp.corrupted_prompt}'")

        clean_inputs = tokenizer(exp.clean_prompt,
                                 return_tensors="pt").to(cfg.device)
        corrupted_inputs = tokenizer(exp.corrupted_prompt,
                                     return_tensors="pt").to(cfg.device)

        baseline_logits_clean = get_logits(model, clean_inputs)
        baseline_logits_corrupted = get_logits(model, corrupted_inputs)
        patched_logits = run_patching(model, corrupted_inputs,
                                      clean_inputs, exp.target_module)

        show_diff(tokenizer, baseline_logits_clean, "Clean",
                  exp.target_token_idx, exp.compare_tokens)
        show_diff(tokenizer, baseline_logits_corrupted, "Corrupted",
                  exp.target_token_idx, exp.compare_tokens)
        show_diff(tokenizer, patched_logits, f"Patched '{exp.target_module}'",
                  exp.target_token_idx, exp.compare_tokens)

    elif exp.type == 'scan_patching':
        print("Running Patching")
        print(f"Clean Prompt:     '{exp.clean_prompt}'")
        print(f"Corrupted Prompt: '{exp.corrupted_prompt}'")

        run_patching_scan(model, tokenizer, cfg)

    elif exp.type == 'scan_ablation':
        print("Running Ablation")
        print(f"Prompt: {exp.prompt}")
        print(f"Comparing: {exp.compare_tokens}")

        run_ablation_scan(model, tokenizer, cfg)


if __name__ == "__main__":
    main()
