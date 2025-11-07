from .hooks import cache, run_hooked, get_logits
from .utils import get_logit_diff

def run_patching(model, clean_inputs, corrupted_inputs, module_name: str):
    """
    Runs the 'corrupted' model while patching in an activation
    from the 'clean' run at a specific module.

    Returns:
        Logits from the patched run.
    """

    # 1. Run clean model and cache the activation
    _, clean_activation = cache(model, clean_inputs, module_name)

    # 2. Define the patch hook
    def patch_hook(module, inp, outp):
        """
        This hook replaces the module's output with the
        cached 'clean' activation.
        """
        # Return the cached activation.
        # We assume the hook replaces the *entire* output of the module.
        # If the output is a tuple, we might need to be more careful,
        # but for MLP blocks and attn blocks, this is often fine.
        return clean_activation

    # 3. Run corrupted model with the patching hook
    patched_logits = run_hooked(
        model, 
        corrupted_inputs, 
        [(module_name, patch_hook)]
    )

    return patched_logits


def run_patching_scan(model, tokenizer, cfg):
    exp = cfg.experiment
    print("--- Running Patching Scan ---")
    print(f"Clean Prompt:     '{exp.clean_prompt}'")
    print(f"Corrupted Prompt: '{exp.corrupted_prompt}'")

    clean_inputs = tokenizer(exp.clean_prompt, return_tensors="pt").to(cfg.device)
    corrupted_inputs = tokenizer(exp.corrupted_prompt, return_tensors="pt").to(cfg.device)

    clean_logits = get_logits(model, clean_inputs)
    corrupted_logits = get_logits(model, corrupted_inputs)

    clean_diff = get_logit_diff(tokenizer, clean_logits,
                                exp.target_token_idx, exp.compare_tokens)
    corrupted_diff = get_logit_diff(tokenizer, corrupted_logits,
                                    exp.target_token_idx, exp.compare_tokens)

    print("\n--- Baselines ---")
    print(f"Clean Logit Diff:     {clean_diff:.4f}")
    print(f"Corrupted Logit Diff: {corrupted_diff:.4f}")

    if clean_diff - corrupted_diff == 0:
        print("Warning: Identical logits!")
        return

    modules_to_scan = []
    for layer in range(cfg.model.n_layer):
        modules_to_scan.append(f"transformer.h.{layer}.attn")
        modules_to_scan.append(f"transformer.h.{layer}.mlp")

    results = []
    print("\nRunning scan...")
    for module_name in modules_to_scan:
        # Use the single run_patching helper
        patched_logits = run_patching(model, clean_inputs, corrupted_inputs, module_name)
        patched_diff = get_logit_diff(tokenizer, patched_logits, exp.target_token_idx, exp.compare_tokens)
        restoration = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
        results.append({"module": module_name,
                        "logit_diff": patched_diff,
                        "restoration": restoration * 100})

    # 5. Print sorted results
    print("\n--- Patching Scan Results ---")
    print("(Sorted by % Restoration)\n")

    sorted_results = sorted(results, key=lambda x: x["restoration"], reverse=True)

    print(f"{'Module':<20} | {'Logit Diff':<12} | {'Restoration (%)':<15}")
    print("-" * 52)
    for res in sorted_results:
        print(f"{res['module']:<20} | {res['logit_diff']:<12.4f} | {res['restoration']:<15.2f}")


def run_head_patching(model, clean_inputs, corrupted_inputs, layer: int, head: int):
    module_name = f"transformer.h.{layer}.attn"
    _, clean_activation = cache(model, clean_inputs, module_name)
    clean_activation = clean_activation[0]
    
    def head_patch_hook(module, inp, outp):
        current_output = outp[0]
        # Assuming output shape: [batch, seq_len, n_embd]
        batch_size, seq_len, n_embd = current_output.shape
        n_heads = model.config.n_head
        d_head = n_embd // n_heads
        
        # Reshape to separate heads: [batch, seq_len, n_heads, d_head]
        current_reshaped = current_output.view(batch_size, seq_len, n_heads, d_head)
        clean_reshaped = clean_activation.view(batch_size, seq_len, n_heads, d_head)
        
        # Patch only the target head
        current_reshaped[:, :, head, :] = clean_reshaped[:, :, head, :]
        
        # Reshape back to original format
        patched_output = current_reshaped.view(batch_size, seq_len, n_embd)
        
        return (patched_output, None)
    
    patched_logits = run_hooked(
        model, 
        corrupted_inputs, 
        [(module_name, head_patch_hook)]
    )
    
    return patched_logits


def run_head_patching_scan(model, tokenizer, cfg):
    exp = cfg.experiment
    print("--- Running Head-Level Patching Scan ---")
    
    clean_inputs = tokenizer(exp.clean_prompt, return_tensors="pt").to(cfg.device)
    corrupted_inputs = tokenizer(exp.corrupted_prompt, return_tensors="pt").to(cfg.device)
    
    clean_logits = get_logits(model, clean_inputs)
    corrupted_logits = get_logits(model, corrupted_inputs)
    
    clean_diff = get_logit_diff(tokenizer, clean_logits,
                                exp.target_token_idx, exp.compare_tokens)
    corrupted_diff = get_logit_diff(tokenizer, corrupted_logits,
                                    exp.target_token_idx, exp.compare_tokens)
    
    print(f"\nClean Logit Diff:     {clean_diff:.4f}")
    print(f"Corrupted Logit Diff: {corrupted_diff:.4f}")
    
    if clean_diff - corrupted_diff == 0:
        print("Warning: Identical logits!")
        return
    
    results = []
    print("\nScanning all heads...")
    
    for layer in range(cfg.model.n_layer):
        for head in range(cfg.model.n_head):
            # Patch this specific head
            patched_logits = run_head_patching(
                model, clean_inputs, corrupted_inputs, layer, head
            )
            patched_diff = get_logit_diff(
                tokenizer, patched_logits, 
                exp.target_token_idx, exp.compare_tokens
            )
            restoration = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
            
            results.append({
                "layer": layer,
                "head": head,
                "module": f"L{layer}H{head}",
                "logit_diff": patched_diff,
                "restoration": restoration * 100
            })
    
    print("\n--- Head Patching Results ---")
    sorted_results = sorted(results, key=lambda x: x["restoration"], reverse=True)
    
    print(f"{'Head':<10} | {'Logit Diff':<12} | {'Restoration (%)':<15}")
    print("-" * 42)
    for res in sorted_results:
        print(f"{res['module']:<10} | {res['logit_diff']:<12.4f} | {res['restoration']:<15.2f}")
    
    return sorted_results
