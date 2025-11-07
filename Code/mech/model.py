from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_path: str, device: str, model_config: dict, attention: bool = False):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager").to(device)
        model.eval()
        if attention:
            model.config.output_attentions = True
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        assert model.config.n_layer == model_config['n_layer']
        assert model.config.n_head == model_config['n_head']
        assert model.config.n_embd == model_config['n_embd']

        return model, tokenizer
    except Exception as e:
        print(f"\nError: {e}")
        raise


def list_modules(model):
    for name, _ in model.named_modules():
        if '.' not in name:
            print("\n---")
        print(name)
