import torch.nn.functional as F


def show_diff(tokenizer, logits, title, token_idx, compare_tokens):
    print(f"\n--- {title} ---")

    logits = logits[0, token_idx, :]
    probs = F.softmax(logits, dim=-1)

    token_ids = [
        tokenizer(word, add_special_tokens=False).input_ids
        for word in compare_tokens
    ]

    token_data = []
    for token, token_id in zip(compare_tokens, token_ids):
        if isinstance(token_id, list):
            token_id = token_id[0]

        token_logit = logits[token_id].item()
        token_prob = probs[token_id].item()
        token_data.append((token, token_logit, token_prob * 100))
    token_data.sort(key=lambda x: x[1], reverse=True)

    print(f"Comparing: {compare_tokens}")
    for token, logit, prob in token_data:
        print(f"  {token:<10} (Logit: {logit:8.4f}) (Prob: {prob:6.2f}%)")

    if len(token_data) >= 2:
        logit_a = token_data[0][1]
        logit_b = token_data[1][1]
        return logit_a - logit_b
    return 0.0


def get_logit_diff(tokenizer, logits, token_idx, compare_tokens):
    logits = logits[0, token_idx, :]
    token_ids = [
        tokenizer(word, add_special_tokens=False).input_ids
        for word in compare_tokens
    ]

    if len(token_ids) < 2:
        print("Warning: Need at least 2 compare_tokens to calculate diff.")
        return 0.0

    id_a = token_ids[0][0] if isinstance(token_ids[0], list) else token_ids[0]
    id_b = token_ids[1][0] if isinstance(token_ids[1], list) else token_ids[1]

    logit_a = logits[id_a].item()
    logit_b = logits[id_b].item()

    return logit_a - logit_b
