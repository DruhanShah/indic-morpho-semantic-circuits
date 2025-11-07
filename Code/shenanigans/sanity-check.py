import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "../../Assets/en_tinystories"

PROMPTS = [
    # "बिल्ली ने चूहे का पीछा किया। बिल्ली चूहे को",
    "John and Mary went to the garden. John gave a flower to"
]

MAX_LENGTH = 44
TEMPERATURE = 0.9
DO_SAMPLE = False
NUM_RETURN_SEQUENCES = 1


def run_generation_check():
    print(f"Loading model and tokenizer from: {MODEL_PATH}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        print(f"Model loaded successfully on {device}.")

    except Exception as e:
        print(e)
        return

    print("\n--- Generating Completions ---")

    for prompt in PROMPTS:
        print(f"\nPrompt: '{prompt}'")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_LENGTH,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    num_return_sequences=NUM_RETURN_SEQUENCES,
                    pad_token_id=tokenizer.pad_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Full Output: {generated_text}")

        except Exception as e:
            print(f"  Error during generation for this prompt: {e}")


if __name__ == "__main__":
    run_generation_check()
