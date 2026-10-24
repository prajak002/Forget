"""Hyperparameters, model paths and loss weights for the exclusive-unlearning pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
FIGURE_DIR = ROOT / "figures"


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B"
    dtype: str = "bfloat16"
    max_seq_len: int = 512
    # LoRA (peft) settings. Projection is applied in the adapter's input space.
    mock_hidden: int = 64  # only used when Config.mock is True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class DataConfig:
    # "code": forget UltraFeedback, retain MBPP/HumanEval.
    # "indian_football": forget general Wikipedia (incl. world football, cricket, India in general),
    #                    retain Indian-football Wikipedia; built by build_football_data.py.
    task: str = "code"
    football_dir: str = "data/indian_football"
    forget_dataset: str = "openbmb/UltraFeedback"
    forget_split: str = "train"
    retain_dataset: str = "google-research-datasets/mbpp"
    retain_subset: str = "full"
    retain_split: str = "train"
    eval_code_dataset: str = "openai/openai_humaneval"
    n_forget: int = 4000
    n_retain: int = 400
    n_eval_forget: int = 200
    n_eval_retain: int = 100
    seed: int = 0


@dataclass
class LossConfig:
    """Weights for L = w_forget * L_forget + w_retain * L_retain."""
    forget_objective: str = "max_entropy"  # one of: ga | npo | max_entropy
    w_forget: float = 1.0
    w_retain: float = 1.0
    npo_beta: float = 0.1
    # Fraction of logits to drop from the uniform target (0 = full-vocab uniform).
    max_entropy_temperature: float = 1.0


@dataclass
class FisherConfig:
    """Empirical Fisher on the retain set, used to build the protected subspace."""
    n_samples: int = 128
    # "true": targets sampled from the model's predictive distribution (proper Fisher).
    # "empirical": ground-truth targets (vanishes at a well-fit point; kept as an ablation).
    mode: str = "true"
    # "retain": protect top-energy directions of the retain Fisher factor.
    # "discriminative": protect directions where retain curvature dominates forget curvature,
    #   i.e. top generalized eigenvectors of (C_retain, C_forget + eps I) with eigenvalue > ratio_threshold.
    subspace: str = "discriminative"
    ratio_threshold: float = 1.0
    # Protected subspace rank per layer; if `energy_fraction` is set it overrides `rank`.
    rank: int = 64
    energy_fraction: float | None = 0.95
    # Soft projection: residual retained = (1 - strength). 1.0 == exact null-space.
    strength: float = 1.0
    damping: float = 1e-4
    # Recompute the protected subspace every N unlearning steps (0 = once, before training).
    refresh_every: int = 50
    # Project the post-Adam parameter *update* as well as the raw gradient. Adam's per-coordinate
    # rescaling does not commute with the projector, so gradient-only projection leaks.
    project_update: bool = True


@dataclass
class PruneConfig:
    """Weight-removal unlearning (Fisher-ratio pruning of base weights)."""
    enabled: bool = False
    mode: str = "prune"          # prune (zero) | dampen (SSD-style scaling)
    alpha: float = 10.0          # remove weight if F_forget / F_retain > alpha
    lam: float = 1.0             # dampen scale factor
    eps: float = 1e-2            # relative damping on F_retain (fraction of its mean)
    max_fraction: float | None = 0.05  # cap on fraction of weights removed per matrix
    n_samples: int = 64
    acc_dtype: str = "bfloat16"  # Fisher accumulator dtype (memory)
    group_layers: int = 2        # layers per Fisher pass (lower = less memory, more passes)
    # Substrings selecting which block weights may be pruned; None = all 2-D block weights.
    targets: tuple[str, ...] | None = ("gate_proj", "up_proj", "down_proj")


@dataclass
class TrainConfig:
    method: str = "cg_oss"  # base | ga | npo | max_entropy | cg_oss
    lr: float = 1e-4
    # Optimizer for the *forget* phase. Adam's per-coordinate normalisation re-amplifies the small
    # residual left after projection; SGD keeps the step proportional to the projected gradient.
    forget_optimizer: str = "adamw"  # adamw | sgd
    forget_lr: float | None = None   # defaults to lr
    batch_size: int = 4
    grad_accum: int = 4
    max_steps: int = 300
    warmup_steps: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    device: str = "auto"
    output_dir: Path = CHECKPOINT_DIR / "cg_oss"


@dataclass
class EvalConfig:
    n_code_problems: int = 164  # HumanEval size
    n_general_prompts: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.0
    refusal_entropy_threshold: float = 6.0  # nats; ln(|V|) for Qwen ~ 11.9
    # Re-entanglement probe: steps of fine-tuning on held-out general text.
    relearn_steps: int = 50
    relearn_lr: float = 5e-5
    code_exec_timeout: float = 5.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # `mock=True` swaps the HF model for a tiny random transformer and synthetic
    # data so the whole pipeline runs on a laptop CPU in minutes.
    mock: bool = False


METHOD_PRESETS: dict[str, dict] = {
    "base":        {"loss": {"w_forget": 0.0}, "train": {"max_steps": 0}, "fisher": {"strength": 0.0}},
    "ga":          {"loss": {"forget_objective": "ga"},          "fisher": {"strength": 0.0}},
    "npo":         {"loss": {"forget_objective": "npo"},         "fisher": {"strength": 0.0}},
    "max_entropy": {"loss": {"forget_objective": "max_entropy"}, "fisher": {"strength": 0.0}},
    "cg_oss":      {"loss": {"forget_objective": "max_entropy"}, "fisher": {"strength": 1.0}},
    # Weight removal only (no gradient unlearning), then weight removal followed by CG-OSS.
    "prune":       {"prune": {"enabled": True}, "loss": {"w_forget": 0.0}, "train": {"max_steps": 0}, "fisher": {"strength": 0.0}},
    "prune_retain": {"prune": {"enabled": True}, "loss": {"w_forget": 0.0}, "fisher": {"strength": 0.0}},
    "prune_cg_oss": {"prune": {"enabled": True}, "loss": {"forget_objective": "max_entropy"}, "fisher": {"strength": 1.0}},
}


def make_config(method: str = "cg_oss", mock: bool = False, **overrides) -> Config:
    """Build a Config with a method preset applied, then flat `section.key=value` overrides."""
    if method not in METHOD_PRESETS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHOD_PRESETS)}")
    cfg = Config(mock=mock)
    cfg.train.method = method
    cfg.train.output_dir = CHECKPOINT_DIR / method
    for section, kv in METHOD_PRESETS[method].items():
        for k, v in kv.items():
            setattr(getattr(cfg, section), k, v)
    if not mock and overrides.pop("laptop", False):
        # Fits a 16 GB Apple-silicon machine: short sequences, batch 1, small eval.
        cfg.model.max_seq_len = 256
        cfg.data.n_forget, cfg.data.n_retain = 600, 374  # MBPP-full train has 374
        cfg.data.n_eval_forget, cfg.data.n_eval_retain = 40, 40
        cfg.fisher.n_samples, cfg.prune.n_samples = 32, 48
        cfg.train.batch_size, cfg.train.grad_accum = 1, 4
        cfg.train.max_steps = min(cfg.train.max_steps, 100)
        cfg.train.eval_every, cfg.train.save_every, cfg.train.log_every = 0, 0, 10
        cfg.eval.n_code_problems, cfg.eval.n_general_prompts = 40, 40
        cfg.eval.max_new_tokens = 192
        cfg.eval.relearn_steps = 20
        cfg.fisher.refresh_every = 0
        cfg.model.lora_target_modules = ("q_proj", "v_proj", "up_proj", "down_proj")
    if mock:
        cfg.data.n_forget, cfg.data.n_retain = 1024, 512
        cfg.data.n_eval_forget, cfg.data.n_eval_retain = 32, 16
        cfg.fisher.n_samples, cfg.fisher.rank = 32, 8
        cfg.train.max_steps = min(cfg.train.max_steps, 200)
        cfg.train.lr = 5e-3
        cfg.train.batch_size, cfg.train.grad_accum = 8, 1
        cfg.train.eval_every = cfg.train.save_every = 50
        cfg.eval.relearn_lr = 1e-3
        cfg.eval.max_new_tokens = 32
        cfg.eval.relearn_steps = 10
    for key, v in overrides.items():
        section, _, k = key.partition(".")
        if not k or not hasattr(cfg, section) or not hasattr(getattr(cfg, section), k):
            raise KeyError(f"unknown override {key!r}")
        setattr(getattr(cfg, section), k, v)
    return cfg
