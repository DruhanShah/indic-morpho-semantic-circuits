from .hooks import run_hooked, get_logits
from .utils import get_logit_diff


def get_head_ablation_hook(model, layer: int, head: int):
    try:
        n_head = model.config.n_head
        head_dim = model.config.n_embd // n_head
    except AttributeError:
        print("Error: Could not get model config. n_head or n_embd not found.")
        return None

    start_idx = head * head_dim
    end_idx = (head + 1) * head_dim

    module_name = f"transformer.h.{layer}.attn"

    def ablate_hook(module, inp, outp):
        if isinstance(outp, tuple):
            outp[0][:, :, start_idx:end_idx] = 0
            return outp
        else:
            outp[:, :, start_idx:end_idx] = 0
            return outp

    return module_name, ablate_hook


def run_ablation(model, inputs, layer: int, head: int):
    hook = get_head_ablation_hook(model, layer, head)
    if hook is None:
        return

    module_name, hook_fn = hook
    return run_hooked(model, inputs, [(module_name, hook_fn)])


def run_multi_ablation(model, inputs, heads_to_ablate: list):
    hooks = []
    for (layer, head) in heads_to_ablate:
        hook = get_head_ablation_hook(model, layer, head)
        if hook:
            hooks.append(hook)

    return run_hooked(model, inputs, hooks)


def run_ablation_scan(model, tokenizer, cfg):
    """
    Runs head ablation on all heads and prints a grid of results.
    """
    exp = cfg.experiment
    print("--- Running Ablation Scan ---")
    print(f"Prompt: '{exp.prompt}'")

    inputs = tokenizer(exp.prompt, return_tensors="pt").to(cfg.device)
    baseline_logits = get_logits(model, inputs)
    baseline_diff = get_logit_diff(tokenizer, baseline_logits,
                                   exp.target_token_idx, exp.compare_tokens)

    print("\n--- Baseline ---")
    print(f"Baseline Logit Diff: {baseline_diff:.4f}")

    results_grid = []
    print("\nRunning scan...")
    for layer in range(cfg.model.n_layer):
        layer_results = []
        for head in range(cfg.model.n_head):
            ablated_logits = run_ablation(model, inputs, layer, head)
            ablated_diff = get_logit_diff(tokenizer, ablated_logits,
                                          exp.target_token_idx, exp.compare_tokens)

            diff_drop = baseline_diff - ablated_diff
            layer_results.append(diff_drop)
        results_grid.append(layer_results)

    print("\n--- Ablation Scan Results ---")
    print("(Metric: Drop in Logit Diff. Higher = More Important)\n")

    header = "L/H   " + "".join([f"H{h:<9}" for h in range(cfg.model.n_head)])
    print(header)
    print("-" * len(header))

    for layer_idx, layer_results in enumerate(results_grid):
        row_str = f"L{layer_idx:<5}"
        for val in layer_results:
            row_str += f"{val:<9.4f}"
        print(row_str)
