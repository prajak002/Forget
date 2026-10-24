"""Model construction: real Qwen2.5-Coder with LoRA, or a tiny mock Llama with the same code path."""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model

from .config import Config

MOCK_VOCAB = 256
MOCK_SEQ_LEN = 32


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mock_base(seed: int = 0, hidden: int = 64):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    mc = LlamaConfig(
        vocab_size=MOCK_VOCAB, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=MOCK_SEQ_LEN * 2, tie_word_embeddings=False,
    )
    return LlamaForCausalLM(mc)


def _mock_pretrain(model, cfg: Config, device: torch.device, steps: int = 3000, lr: float = 2e-3):
    """Teach the tiny model both domains so there is general knowledge to strip and code to keep.
    Cached at checkpoints/mock_base_seed{seed}.pt so every method starts from the same weights."""
    from .config import CHECKPOINT_DIR
    from .dataset import build_data, infinite
    from .losses import cross_entropy

    path = CHECKPOINT_DIR / f"mock_base_h{cfg.model.mock_hidden}_seed{cfg.data.seed}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    data = build_data(cfg)
    f_it = infinite(data.loader(data.forget_train, 16, seed=cfg.data.seed))
    r_it = infinite(data.loader(data.retain_train, 16, seed=cfg.data.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        loss = 0.0
        for it in (f_it, r_it):
            b = {k: v.to(device) for k, v in next(it).items()}
            loss = loss + cross_entropy(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits, b["labels"])
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"[mock-pretrain] step {step} loss {float(loss)/2:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval()


def load_base_model(cfg: Config, device: torch.device):
    if cfg.mock:
        return _mock_pretrain(_mock_base(cfg.data.seed, cfg.model.mock_hidden).to(device), cfg, device)
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, cfg.model.dtype)
    return AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, torch_dtype=dtype).to(device)


def load_tokenizer(cfg: Config):
    if cfg.mock:
        return None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def wrap_lora(model, cfg: Config):
    lc = LoraConfig(
        r=cfg.model.lora_r, lora_alpha=cfg.model.lora_alpha, lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.lora_target_modules), bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


def lora_pairs(model) -> dict[str, tuple[torch.nn.Module, torch.nn.Linear, torch.nn.Linear]]:
    """Map base-module name -> (wrapped LoraLayer, lora_A, lora_B) for adapter 'default'."""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            out[name] = (mod, mod.lora_A["default"], mod.lora_B["default"])
    return out
