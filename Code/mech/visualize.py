import torch
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Noto Sans Devanagari"

import seaborn as sns
from .hooks import find_module


def get_attention_patterns(model, inputs, layer: int, head: int):
    """
    Extract attention patterns for a specific head.
    
    Args:
        model: The transformer model
        inputs: Tokenized inputs
        layer: Layer index
        head: Head index
    
    Returns:
        attention_weights: [seq_len, seq_len] tensor
    """
    attention_weights = None
    
    def attention_hook(module, inp, outp):
        nonlocal attention_weights
        # outp is typically (output, attention_weights) for GPT-2
        if isinstance(outp, tuple) and len(outp) > 1 and outp[1] is not None:
            # Shape: [batch, n_heads, seq_len, seq_len]
            attn = outp[1]
            # Extract specific head
            attention_weights = attn[0, head, :, :].detach().cpu()
        else:
            # Need to request attention output
            print("Warning: Attention weights not available. Set output_attentions=True")
    
    module_name = f"transformer.h.{layer}.attn"
    module = find_module(model, module_name)
    handle = module.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        # Run with output_attentions if model supports it
        try:
            _ = model(**inputs, output_attentions=True)
        except:
            _ = model(**inputs)
    
    handle.remove()
    return attention_weights


def visualize_attention_pattern(attention_weights, tokens, layer, head, 
                                 save_path=None, title=None):
    """
    Visualize attention pattern as a heatmap.
    
    Args:
        attention_weights: [seq_len, seq_len] tensor
        tokens: List of token strings
        layer: Layer index (for title)
        head: Head index (for title)
        save_path: Optional path to save figure
        title: Optional custom title
    """
    if attention_weights is None:
        print("No attention weights to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights.numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )
    
    # Set title
    if title is None:
        title = f'Attention Pattern: Layer {layer}, Head {head}'
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def visualize_multiple_heads(model, tokenizer, cfg, heads_list):
    """
    Visualize attention patterns for multiple heads.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        cfg: Config object with experiment.prompt
        heads_list: List of (layer, head) tuples
    """
    exp = cfg.experiment
    inputs = tokenizer(exp.prompt, return_tensors="pt").to(cfg.device)
    
    # Get tokens for labels
    token_ids = inputs['input_ids'][0].cpu().tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Create subplots
    n_heads = len(heads_list)
    cols = min(3, n_heads)
    rows = (n_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (layer, head) in enumerate(heads_list):
        attention_weights = get_attention_patterns(model, inputs, layer, head)
        
        if attention_weights is not None:
            ax = axes[idx]
            sns.heatmap(
                attention_weights.numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Attention'},
                ax=ax
            )
            ax.set_title(f'{layer}.{head}', fontsize=12)
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    print("Saved to attention_patterns.png")
    plt.show()
    
    return fig


def run_attention_visualization(model, tokenizer, cfg):
    """
    Main function to run attention visualization based on config.
    """
    exp = cfg.experiment
    
    if hasattr(exp, 'visualize_heads') and exp.visualize_heads:
        print("--- Visualizing Attention Patterns ---")
        print(f"Prompt: '{exp.prompt}'")
        print(f"Heads: {exp.visualize_heads}")
        
        visualize_multiple_heads(model, tokenizer, cfg, exp.visualize_heads)
    else:
        print("No heads specified for visualization. Add 'visualize_heads' to config.")
