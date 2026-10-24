import math

import torch

from exclusive_unlearning.config import make_config
from exclusive_unlearning.dataset import IGNORE_INDEX, build_data
from exclusive_unlearning.fisher import compute_protected_subspaces, project_lora_grads
from exclusive_unlearning.losses import (cross_entropy, max_entropy_loss, npo_loss,
                                         gradient_ascent_loss, token_entropy)
from exclusive_unlearning.models import load_base_model, lora_pairs, wrap_lora


def _fake(B=2, T=6, V=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g)
    labels[:, :2] = IGNORE_INDEX
    return logits, labels


def test_max_entropy_zero_at_uniform_and_positive_otherwise():
    logits, labels = _fake()
    assert torch.allclose(max_entropy_loss(torch.zeros_like(logits), labels), torch.tensor(0.0), atol=1e-6)
    assert max_entropy_loss(logits * 5, labels) > 0.1


def test_max_entropy_bounded_and_ga_unbounded():
    logits, labels = _fake()
    peaked = logits * 50
    assert math.isfinite(float(max_entropy_loss(peaked, labels)))
    assert gradient_ascent_loss(peaked, labels) < -10  # GA keeps decreasing without bound


def test_ga_is_negative_ce():
    logits, labels = _fake()
    assert torch.allclose(gradient_ascent_loss(logits, labels), -cross_entropy(logits, labels))


def test_npo_equals_reference_at_init_and_bounded():
    logits, labels = _fake()
    l0 = npo_loss(logits, labels, logits.clone(), beta=0.1)
    # at pi == pi_ref, log-ratio = 0 -> loss = -(2/beta) log(0.5) = 2 ln2 / beta
    assert torch.allclose(l0, torch.tensor(2 * math.log(2) / 0.1), atol=1e-4)
    assert npo_loss(logits - 100, labels, logits, beta=0.1) >= 0


def test_token_entropy_max_at_uniform():
    logits, labels = _fake(V=11)
    assert torch.allclose(token_entropy(torch.zeros_like(logits), labels), torch.tensor(math.log(11)), atol=1e-5)


def test_projection_removes_protected_component_exactly():
    cfg = make_config("cg_oss", mock=True)
    cfg.fisher.energy_fraction = None
    cfg.fisher.rank = 4
    dev = torch.device("cpu")
    model = wrap_lora(load_base_model(cfg, dev), cfg)
    data = build_data(cfg)
    ps = compute_protected_subspaces(model, data, cfg, dev)
    assert len(ps.inputs) == len(lora_pairs(model)) > 0
    for s in ps.inputs.values():
        assert s.rank == 4
        assert torch.allclose(s.U.T @ s.U, torch.eye(4), atol=1e-4)
    # give every LoRA param a random grad, project, check residual is orthogonal to U
    for _, (_, A, B) in lora_pairs(model).items():
        A.weight.grad = torch.randn_like(A.weight)
        B.weight.grad = torch.randn_like(B.weight)
    stats = project_lora_grads(model, ps, strength=1.0)
    assert 0 < stats["removed_frac"] < 1
    for name, (_, A, B) in lora_pairs(model).items():
        assert torch.allclose(A.weight.grad @ ps.inputs[name].U, torch.zeros(A.weight.shape[0], 4), atol=1e-5)
        assert torch.allclose(ps.outputs[name].U.T @ B.weight.grad, torch.zeros(4, B.weight.shape[1]), atol=1e-5)


def test_projection_strength_zero_is_noop():
    cfg = make_config("max_entropy", mock=True)
    model = wrap_lora(load_base_model(cfg, torch.device("cpu")), cfg)
    for _, (_, A, _) in lora_pairs(model).items():
        A.weight.grad = torch.ones_like(A.weight)
    assert project_lora_grads(model, None, strength=0.0)["removed_frac"] == 0.0


def test_config_presets():
    assert make_config("ga").loss.forget_objective == "ga"
    assert make_config("ga").fisher.strength == 0.0
    assert make_config("cg_oss").fisher.strength == 1.0
    assert make_config("base").train.max_steps == 0
    assert make_config("npo", **{"loss.npo_beta": 0.5}).loss.npo_beta == 0.5


def test_discriminative_subspace_rank_deficient_large_d():
    """Must not crash when d >> samples (real Qwen MLP widths) and must prefer retain-only directions."""
    from exclusive_unlearning.fisher import _discriminative_subspace
    cfg = make_config("cg_oss", mock=True)
    g = torch.Generator().manual_seed(0)
    d = 512
    shared = torch.randn(1, d, generator=g)
    retain_only = torch.randn(1, d, generator=g)
    Rr = torch.randn(40, 1, generator=g) * shared + torch.randn(40, 1, generator=g) * 3 * retain_only
    Rf = torch.randn(40, 1, generator=g) * shared * 3
    sub = _discriminative_subspace(Rr, Rf, cfg)
    assert torch.allclose(sub.U.T @ sub.U, torch.eye(sub.rank), atol=1e-4)
    ro = retain_only / retain_only.norm()
    sh = shared / shared.norm()
    assert float((ro @ sub.U).pow(2).sum()) > 0.9   # retain-only direction is protected
    assert float((sh @ sub.U).pow(2).sum()) < 0.5   # shared, forget-dominated direction is not


def test_check_program_pass_fail_timeout():
    from exclusive_unlearning.code_eval import check_program
    assert check_program("assert 1 + 1 == 2\n", 5.0)
    assert not check_program("assert 1 + 1 == 3\n", 5.0)
    assert not check_program("while True:\n    pass\n", 1.0)
    assert not check_program("import exclusive_unlearning\n", 5.0)  # project not importable from sandbox
