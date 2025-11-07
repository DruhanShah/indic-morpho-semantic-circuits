import torch


def find_module(model, name: str):
    return dict(model.named_modules())[name]


def cache(model, inputs, module_name: str):
    cached_val = None

    def hook_fn(module, inp, outp):
        nonlocal cached_val
        cached_val = outp

    module = find_module(model, module_name)
    handle = module.register_forward_hook(hook_fn)
    with torch.no_grad():
        logits = model(**inputs).logits
    handle.remove()

    return logits, cached_val


def run_hooked(model, inputs, hooks: list):
    handles = []
    try:
        for module_name, hook_fn in hooks:
            module = find_module(model, module_name)
            handles.append(module.register_forward_hook(hook_fn))
        with torch.no_grad():
            logits = model(**inputs).logits
    finally:
        for h in handles:
            h.remove()

    return logits


def get_logits(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits
