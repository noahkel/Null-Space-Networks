"""
test_nsn.py — the project's test suite, in one module.

Covers, in order:

   1. attack objectives      mse / shift / hybrid / null / range / targeted
   2. attack primitives      norm projections, gradient normalisation, budgets
   3. the PGD attack         the one algorithm the suite runs
   4. metrics + aggregation  evaluate_batch, summarize_metrics, aggregate_*
   5. artifacts              the .npz/.json on-disk contract
   6. rendering              figure entry points and the render progress protocol
   7. models                 RESNET / NSN / DPNSN / DPNSN_RES forward semantics
   8. the UNet blocks        shape contracts and the odd-size skip padding
   9. pipeline setup         prepare_run, build_init_inputs, radon/model loading
  10. radon operators        the AstraRadonAdapter / MatrixRadonAdapter identities
                             and the FBP filter construction
  11. numeric helpers        the image-quality metrics in src/utils.py
  12. source hygiene         invariants the architecture depends on

Every test is deterministic and needs no data directory, trained checkpoint or
GPU. Gated with importorskip so it skips cleanly wherever torch / astra / scipy
/ odl / dival are unavailable instead of erroring.  Run:  pytest -v
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import src.attack as attack
from src.artifacts import (
    read_metric_rows,
    read_rows_bundle,
    read_transfer_bundle,
    write_rows_bundle,
    write_transfer_bundle,
)
from src import utils
from src.utils import mse_loss


# --------------------------------------------------------------------------- #
# Size-parametric fake limited-angle operator + adapter, mirroring the real
# MatrixRadonAdapter / ModelAttackAdapter interface used by src/attack.py.
# Images are (B,1,H,W); sinograms are (B,1,rows,1). Operator ratios from the
# original test (M_FULL = 1.5 N, LA_ROWS = 0.75 N).
# --------------------------------------------------------------------------- #
IMG = 6
N = IMG * IMG
M_FULL = 54
LA_ROWS = 27


def _build_operator(n, m_full, la_rows, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m_full, n))
    A_la = A[:la_rows]
    _, S, Vt = np.linalg.svd(A_la, full_matrices=False)
    Vk = Vt[S >= 1e-6 * S.max()]
    P_null = np.eye(n) - Vk.T @ Vk
    return A, A_la, P_null


class FakeRadon:
    def __init__(self, img=IMG, seed=0, dtype=torch.float64):
        self.IMG = img
        self.N = img * img
        self.M_FULL = (3 * self.N) // 2
        self.LA_ROWS = (3 * self.N) // 4
        A, A_la, P_null = _build_operator(self.N, self.M_FULL, self.LA_ROWS, seed)
        self.dtype = dtype
        self._A = torch.tensor(A, dtype=dtype)
        self._A_la = torch.tensor(A_la, dtype=dtype)
        self._P = torch.tensor(P_null, dtype=dtype)
        # Pseudoinverses back the fbp / fbp_la stubs below, so the wrapper models
        # (which compose forward -> project -> back-project) see an operator pair
        # that actually round-trips instead of two unrelated linear maps.
        self._A_pinv = torch.tensor(np.linalg.pinv(A), dtype=dtype)
        self._A_la_pinv = torch.tensor(np.linalg.pinv(A_la), dtype=dtype)
        self.norm_A2 = 1.0

    def proj_null_image(self, v):
        b = v.shape[0]
        out = v.reshape(b, self.N).to(self.dtype) @ self._P.T
        return out.reshape(v.shape).to(v.dtype)

    def proj_ran(self, y):
        # Fake forward_la already emits only measured rows, so this is identity.
        return y

    def forward_la(self, v):
        b = v.shape[0]
        y = v.reshape(b, self.N).to(self.dtype) @ self._A_la.T
        return y.reshape(b, 1, self.LA_ROWS, 1).to(v.dtype)

    def forward(self, v):
        b = v.shape[0]
        y = v.reshape(b, self.N).to(self.dtype) @ self._A.T
        return y.reshape(b, 1, self.M_FULL, 1).to(v.dtype)

    def proj_nsn(self, y):
        """Projector onto the *unmeasured* rows of a full sinogram: forward()
        stacks the measured LA_ROWS first, so this zeroes them."""
        out = y.clone()
        out.reshape(y.shape[0], -1)[:, : self.LA_ROWS] = 0
        return out

    def fbp_la(self, y):
        b = y.shape[0]
        x = y.reshape(b, self.LA_ROWS).to(self.dtype) @ self._A_la_pinv.T
        return x.reshape(b, 1, self.IMG, self.IMG).to(y.dtype)

    def fbp(self, y):
        b = y.shape[0]
        x = y.reshape(b, self.M_FULL).to(self.dtype) @ self._A_pinv.T
        return x.reshape(b, 1, self.IMG, self.IMG).to(y.dtype)

    def decompose_error(self, e, iters=50, tol=1e-6):
        e_nul = self.proj_null_image(e)
        e_ran = e - e_nul
        return e_ran, e_nul


class _FakeInit:
    def __init__(self, radon):
        self.radon = radon

    def surrogate(self, y):
        return torch.zeros(y.shape[0], 1, self.radon.IMG, self.radon.IMG, dtype=y.dtype)


class FakeAdapter:
    """Duck-typed ModelAttackAdapter: a fixed linear 'model' mapping the
    (flattened) sinogram to an image, identity measurement projector, radon
    handle for the null/range/zero objectives. Sizes read off the radon."""

    def __init__(self, radon, seed=1, dtype=torch.float64):
        rng = np.random.default_rng(seed)
        self.radon = radon
        self.W = torch.tensor(rng.standard_normal((radon.LA_ROWS, radon.N)), dtype=dtype)
        self.init_reconstructor = _FakeInit(radon)
        self.projector = lambda y: y
        self.dtype = dtype

    def forward(self, y_adv, mode=None, project=True):
        b = y_adv.shape[0]
        img = self.radon.IMG
        pred = (y_adv.reshape(b, self.radon.LA_ROWS).to(self.dtype) @ self.W).reshape(b, 1, img, img)
        return pred.to(y_adv.dtype), None, y_adv


@pytest.fixture
def radon():
    return FakeRadon()


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# thin re-export
# --------------------------------------------------------------------------- #
def _obj(pred, gt, clean, name, radon, w=0.25):
    return attack.attack_objective(pred, gt, clean, name, w, radon=radon)


def test_objective_mse_shift_hybrid(radon):
    pred, gt, clean = _rand(4, 1, IMG, IMG), _rand(4, 1, IMG, IMG), _rand(4, 1, IMG, IMG)
    assert torch.allclose(_obj(pred, gt, clean, "mse", radon), ((pred - gt) ** 2).mean())
    assert torch.allclose(_obj(pred, gt, clean, "shift", radon), ((pred - clean) ** 2).mean())
    hybrid = _obj(pred, gt, clean, "hybrid", radon)
    expect = ((pred - gt) ** 2).mean() + 0.25 * ((pred - clean) ** 2).mean()
    assert torch.allclose(hybrid, expect)


def test_objective_zero_is_negative_pred_energy(radon):
    pred, gt, clean = _rand(3, 1, IMG, IMG), _rand(3, 1, IMG, IMG), _rand(3, 1, IMG, IMG)
    val = _obj(pred, gt, clean, "zero", radon)
    assert torch.allclose(val, -(pred ** 2).mean())
    assert _obj(2 * pred, gt, clean, "zero", radon) < _obj(pred, gt, clean, "zero", radon)


def test_objective_target_is_negative_distance_to_target(radon):
    pred, gt, clean = _rand(3, 1, IMG, IMG), _rand(3, 1, IMG, IMG), _rand(3, 1, IMG, IMG)
    target = _rand(3, 1, IMG, IMG)
    val = attack.attack_objective(pred, gt, clean, "target", 0.25, radon=radon, target=target)
    assert torch.allclose(val, -((pred - target) ** 2).mean())
    zeros = torch.zeros_like(pred)
    val0 = attack.attack_objective(pred, gt, clean, "target", 0.25, radon=radon, target=zeros)
    assert torch.allclose(val0, _obj(pred, gt, clean, "zero", radon))
    closer = target + 0.1 * (pred - target)
    val_close = attack.attack_objective(closer, gt, clean, "target", 0.25, radon=radon, target=target)
    assert val_close > val


def test_objective_target_requires_target(radon):
    pred = _rand(2, 1, IMG, IMG)
    with pytest.raises(ValueError):
        attack.attack_objective(pred, pred, pred, "target", 0.25, radon=radon)


def test_objective_null_range_decomposition(radon):
    pred, gt, clean = _rand(4, 1, IMG, IMG), _rand(4, 1, IMG, IMG), _rand(4, 1, IMG, IMG)
    null = _obj(pred, gt, clean, "null", radon)
    rng = _obj(pred, gt, clean, "range", radon)
    e = pred - gt
    en = radon.proj_null_image(e)
    er = e - en
    assert torch.allclose(null, (en ** 2).mean())
    assert torch.allclose(rng, (er ** 2).mean())
    assert torch.allclose((e ** 2).mean(), null + rng, atol=1e-8)


def test_objective_null_requires_radon():
    pred = _rand(2, 1, IMG, IMG)
    with pytest.raises(ValueError):
        attack.attack_objective(pred, pred, pred, "null", 0.25, radon=None)


def test_objective_unknown_raises(radon):
    pred = _rand(2, 1, IMG, IMG)
    with pytest.raises(ValueError):
        attack.attack_objective(pred, pred, pred, "does_not_exist", 0.25, radon=radon)


# --------------------------------------------------------------------------- #
# norm / gradient / projection helpers
# --------------------------------------------------------------------------- #
def test_norm_batch_helpers():
    x = _rand(3, 1, IMG, IMG)
    l2 = attack.l2_norm_batch(x)
    linf = attack.linf_norm_batch(x)
    assert l2.shape == (3,) and linf.shape == (3,)
    for i in range(3):
        assert math.isclose(float(l2[i]), float(torch.linalg.norm(x[i].reshape(-1))), rel_tol=1e-6)
        assert math.isclose(float(linf[i]), float(x[i].abs().max()), rel_tol=1e-6)


def test_normalize_grad_is_unit_l2():
    g = _rand(4, 1, IMG, IMG)
    u = attack.normalize_grad(g)
    assert torch.allclose(attack.l2_norm_batch(u), torch.ones(4, dtype=torch.float64), atol=1e-6)


def test_proj_l2_ball_enforces_radius():
    delta = _rand(4, 1, IMG, IMG) * 10.0
    proj = attack.proj_l2_ball(delta, 2.0)
    assert float(attack.l2_norm_batch(proj).max()) <= 2.0 + 1e-6


def test_proj_l2_ball_leaves_small_delta_untouched():
    delta = _rand(4, 1, IMG, IMG)
    delta = delta / attack.l2_norm_batch(delta).view(-1, 1, 1, 1)
    proj = attack.proj_l2_ball(delta, 10.0)
    assert torch.allclose(proj, delta, atol=1e-6)


def test_project_delta_applies_projector_and_ball(radon):
    proj = radon.proj_null_image
    delta = _rand(4, 1, IMG, IMG) * 3.0
    out = attack.project_delta(delta, 1.0, proj)
    assert float(attack.l2_norm_batch(out).max()) <= 1.0 + 1e-6
    assert torch.allclose(out, proj(out), atol=1e-8)


# --------------------------------------------------------------------------- #
# Budget representation. There is one: a per-sample [B,1,1,1] tensor, produced by
# as_eps_batch. A float is sugar for "this budget for every sample" and must be
# indistinguishable from the vector spelling it out, or the suite (which always
# passes a vector) and the tests (which mostly pass floats) verify different code.
# --------------------------------------------------------------------------- #
def test_as_eps_batch_broadcasts_a_scalar_and_normalises_a_vector():
    x = _rand(3, 1, IMG, IMG)
    from_scalar = attack.as_eps_batch(0.5, x)
    from_vector = attack.as_eps_batch(torch.full((3,), 0.5, dtype=x.dtype), x)
    assert from_scalar.shape == (3, 1, 1, 1)
    assert torch.allclose(from_scalar, from_vector)


def test_as_eps_batch_clamps_a_negative_budget_to_zero():
    """The old scalar paths early-returned zeros for eps <= 0; the clamp is what
    reproduces that now that only the tensor path exists."""
    x = _rand(2, 1, IMG, IMG)
    assert torch.count_nonzero(attack.as_eps_batch(-1.0, x)) == 0
    assert torch.count_nonzero(
        attack.as_eps_batch(torch.tensor([-1.0, -2.0], dtype=x.dtype), x)) == 0


@pytest.mark.parametrize("eps", [0.0, 0.5, 3.0])
def test_scalar_and_per_sample_budgets_project_identically(eps):
    """The equivalence the unified representation rests on."""
    delta = _rand(4, 1, IMG, IMG) * 5.0
    vec = torch.full((4,), eps, dtype=delta.dtype)
    assert torch.allclose(attack.proj_l2_ball(delta, eps),
                          attack.proj_l2_ball(delta, vec), atol=1e-10)


def test_zero_budget_gives_no_perturbation(radon):
    """A zero budget has no special case left in the code, so check the three
    entry points still produce exactly nothing at eps = 0."""
    delta = _rand(4, 1, IMG, IMG) * 5.0
    proj = radon.proj_null_image
    assert torch.count_nonzero(attack.project_delta(delta, 0.0, proj)) == 0
    assert torch.count_nonzero(attack.random_start_like(delta, 0.0, proj)) == 0
    assert torch.count_nonzero(
        attack.random_start_like(delta, torch.zeros(4, dtype=delta.dtype), proj)) == 0


def test_per_sample_budgets_are_enforced_independently():
    """The reason the vector form exists: sample i must be held to eps_i, not to
    the batch's largest or smallest budget."""
    delta = _rand(3, 1, IMG, IMG) * 10.0
    vec = torch.tensor([0.1, 1.0, 5.0], dtype=delta.dtype)
    norms = attack.l2_norm_batch(attack.proj_l2_ball(delta, vec))
    assert torch.all(norms <= vec + 1e-6)
    # and each one is actually *at* its own budget, since the input overshoots
    assert torch.allclose(norms, vec, rtol=1e-5)


# --------------------------------------------------------------------------- #
# The PGD attack.
# --------------------------------------------------------------------------- #
def _clean_setup(radon):
    adapter = FakeAdapter(radon)
    y_clean = _rand(4, 1, radon.LA_ROWS, 1)
    with torch.no_grad():
        clean_pred, _, _ = adapter.forward(y_clean, project=False)
    return adapter, y_clean, clean_pred


def test_pgd_zero_objective_shrinks_prediction(radon):
    adapter, y_clean, clean_pred = _clean_setup(radon)
    eps = float(3.0 * attack.l2_norm_batch(y_clean).mean())
    res = attack.pgd_attack(adapter, x_gt=torch.zeros_like(clean_pred), y_clean=y_clean,
                            clean_pred=clean_pred, eps=eps, alpha=0.1 * eps, steps=150,
                            restarts=1, objective="zero", shift_weight=0.0)
    with torch.no_grad():
        adv_pred, _, _ = adapter.forward(res.y_adv, project=False)
    assert attack.l2_norm_batch(adv_pred).mean() < attack.l2_norm_batch(clean_pred).mean()
    assert float(attack.l2_norm_batch(res.delta).max()) <= eps + 1e-6


def test_pgd_mse_objective_grows_error(radon):
    adapter, y_clean, clean_pred = _clean_setup(radon)
    x_gt = torch.zeros_like(clean_pred)
    eps = float(3.0 * attack.l2_norm_batch(y_clean).mean())
    res = attack.pgd_attack(adapter, x_gt=x_gt, y_clean=y_clean, clean_pred=clean_pred,
                            eps=eps, alpha=0.1 * eps, steps=150, restarts=1,
                            objective="mse", shift_weight=0.0)
    with torch.no_grad():
        adv_pred, _, _ = adapter.forward(res.y_adv, project=False)
    assert attack.l2_norm_batch(adv_pred - x_gt).mean() > attack.l2_norm_batch(clean_pred - x_gt).mean()


# --------------------------------------------------------------------------- #
# small reduction helpers
# --------------------------------------------------------------------------- #
def test_reduction_helpers():
    x = _rand(3, 1, IMG, IMG)
    y = _rand(3, 1, IMG, IMG)
    assert torch.allclose(attack.reduce_loss((x - y) ** 2), ((x - y) ** 2).mean())
    v = _rand(5)
    assert torch.allclose(attack.reduce_loss(v), v.mean())
    pe = attack.per_example_mse(x, y)
    assert pe.shape == (3,)
    assert torch.allclose(pe[0], ((x[0] - y[0]) ** 2).mean())
    assert torch.allclose(attack.batch_mean_abs(x)[1], x[1].abs().mean())


def test_confidence_interval_95():
    mean, half = attack.confidence_interval_95([1.0, 1.0, 1.0])
    assert math.isclose(mean, 1.0) and math.isclose(half, 0.0, abs_tol=1e-12)
    mean1, half1 = attack.confidence_interval_95([5.0])
    assert math.isclose(mean1, 5.0) and half1 == 0.0
    mean2, half2 = attack.confidence_interval_95([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(mean2, 2.5)
    assert half2 > 0.0


def test_stack_chunks_concatenates_and_drops_channel():
    a = _rand(2, 1, 3, 3)
    b = _rand(1, 1, 3, 3)
    out = attack._stack_chunks([a, b])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 3, 3)
    assert np.allclose(out, torch.cat([a, b], dim=0)[:, 0].numpy())


# --------------------------------------------------------------------------- #
# evaluate_batch + summarize_metrics (real metric path on a fake operator)
# --------------------------------------------------------------------------- #
def test_evaluate_batch_and_summarize():
    r = FakeRadon(img=8, seed=3)   # 8x8 so skimage SSIM's 7x7 window fits
    B = 3
    x_gt = torch.rand(B, 1, r.IMG, r.IMG, dtype=torch.float64)
    clean_init = x_gt + 0.01 * _rand(B, 1, r.IMG, r.IMG)
    clean_pred = x_gt + 0.02 * _rand(B, 1, r.IMG, r.IMG)
    adv_init = x_gt + 0.05 * _rand(B, 1, r.IMG, r.IMG)
    adv_pred = x_gt + 0.20 * _rand(B, 1, r.IMG, r.IMG)
    clean_y = r.forward_la(x_gt)
    adv_y = clean_y + 0.05 * _rand(B, 1, r.LA_ROWS, 1)
    delta = adv_y - clean_y

    rows = attack.evaluate_batch(
        x_gt=x_gt, clean_init=clean_init, clean_y=clean_y, clean_pred=clean_pred,
        adv_init=adv_init, adv_y=adv_y, adv_pred=adv_pred, delta=delta,
        success_mse_factor=2.0, radon=r,
    )
    assert len(rows) == B
    row = rows[0]
    assert row["adv_mse"] > row["clean_mse"]
    assert 0.0 <= row["adv_e_nul_frac"] <= 1.0 + 1e-9
    tot2 = row["adv_e_ran_l2"] ** 2 + row["adv_e_nul_l2"] ** 2
    e_l2 = np.linalg.norm((adv_pred[0] - x_gt[0]).reshape(-1).numpy())
    assert math.isclose(math.sqrt(tot2), e_l2, rel_tol=1e-5, abs_tol=1e-6)

    summary = attack.summarize_metrics(rows)
    assert summary["num_examples"] == B
    for key in ["adv_rel_l2_mean", "adv_rel_l2_median", "adv_e_nul_l2_mean", "delta_l2_mean"]:
        assert key in summary and math.isfinite(summary[key])


# --------------------------------------------------------------------------- #
# artifact round-trip: the attack -> visualise on-disk contract
# --------------------------------------------------------------------------- #
def test_rows_bundle_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{
        "x_gt": rng.standard_normal((8, 8)),
        "delta": rng.standard_normal((8, 8)),
        "m_adv_pred": {"rel_l2": 0.5, "psnr": 20.0, "ssim": 0.5},
        "worst_score": 0.001,
    }]
    write_rows_bundle(tmp_path / "examples.npz", tmp_path / "examples.json", rows)
    back = read_rows_bundle(tmp_path / "examples.npz", tmp_path / "examples.json")
    assert len(back) == 1
    assert set(back[0].keys()) == set(rows[0].keys())
    assert np.allclose(back[0]["x_gt"], rows[0]["x_gt"])
    assert back[0]["m_adv_pred"]["psnr"] == 20.0
    assert math.isclose(back[0]["worst_score"], 0.001)


def test_transfer_bundle_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    K = 3
    gt = np.stack([rng.standard_normal((8, 8)) for _ in range(K)])
    recon = {"clean__nsn": np.stack([rng.standard_normal((8, 8)) for _ in range(K)]),
             "pred__nsn__resnet": np.stack([rng.standard_normal((8, 8)) for _ in range(K)])}
    write_transfer_bundle(tmp_path / "transfer.npz", tmp_path / "transfer.json",
                          model_names=["nsn", "resnet"],
                          attack_name="adversarial", eps=0.01, T=2, n_ex=2,
                          gt_stack=gt, recon=recon)
    meta, data = read_transfer_bundle(tmp_path / "transfer.npz", tmp_path / "transfer.json")
    assert meta["model_names"] == ["nsn", "resnet"] and meta["T"] == 2
    assert np.allclose(data["gt"], gt)
    assert np.allclose(data["pred__nsn__resnet"], recon["pred__nsn__resnet"])


def test_read_metric_rows_returns_floats(tmp_path):
    import csv
    p = tmp_path / "per_sample_metrics.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["adv_rel_l2", "clean_rel_l2"])
        w.writeheader()
        w.writerow({"adv_rel_l2": 0.5, "clean_rel_l2": 0.1})
        w.writerow({"adv_rel_l2": 0.7, "clean_rel_l2": 0.2})
    rows = read_metric_rows(p)
    assert len(rows) == 2
    assert all(isinstance(v, float) for r in rows for v in r.values())
    assert rows[1]["adv_rel_l2"] == 0.7


# --------------------------------------------------------------------------- #
# suite wiring
# --------------------------------------------------------------------------- #
def test_targeted_attacks_registered_in_suite():
    assert "adversarial_target_zero" in attack._SUITE_ATTACKS
    assert "adversarial_target_sample" in attack._SUITE_ATTACKS
    assert attack._SUITE_OBJECTIVE["adversarial_target_zero"] == "zero"
    assert attack._SUITE_OBJECTIVE["adversarial_target_sample"] == "target"
    assert attack._SUITE_TARGETED_ATTACKS == {"adversarial_target_sample"}
    assert attack._SUITE_ATTACKS[0] == "adversarial"
    assert set(attack._SUITE_ATTACKS) == set(attack._SUITE_OBJECTIVE)


def test_all_suite_objectives_are_valid(radon):
    pred, gt, clean = _rand(2, 1, IMG, IMG), _rand(2, 1, IMG, IMG), _rand(2, 1, IMG, IMG)
    tgt = _rand(2, 1, IMG, IMG)
    for name in attack._SUITE_OBJECTIVE.values():
        val = attack.attack_objective(pred, gt, clean, name, 0.25, radon=radon, target=tgt)
        assert torch.isfinite(val), f"objective {name!r} produced non-finite value"


def test_make_other_sample_target_is_derangement():
    x_gt = torch.arange(5, dtype=torch.float64).reshape(5, 1, 1, 1).repeat(1, 1, IMG, IMG)
    torch.manual_seed(0)
    target = attack.make_other_sample_target(x_gt)
    assert target.shape == x_gt.shape
    for i in range(5):
        assert not torch.allclose(target[i], x_gt[i]), f"row {i} maps to itself"
        assert any(torch.allclose(target[i], x_gt[j]) for j in range(5) if j != i)
    single = x_gt[:1]
    assert torch.allclose(attack.make_other_sample_target(single), single)

# =========================================================================== #
# Real-operator / real-model integration tests.
#
# These need the actual MatrixRadonAdapter (built via astra) and, for the astra
# path, AstraRadonAdapter. They are gated by importorskip so the fast fake-based
# unit tests above still run without astra / scipy. Everything runs on CPU in
# float64 at a tiny resolution so it stays fast.
#
# Note on the truncated SVD: MatrixRadonAdapter keeps singular values above
# svd_threshold, so A_la @ (P_null x) is NOT exactly zero — only its projection
# onto the retained range, proj_ran(A_la @ P_null x), is exactly zero (the
# sub-cutoff trace remains, exactly as the ghost construction relies on). The
# measured data the pipeline uses is proj_ran(forward_la(.)), so the invariants
# below are asserted through proj_ran.
# =========================================================================== #
_RES = 16
_DET = 24                                  # >= ceil(sqrt(2)*16)=23, avoids clip warning
_N_ANGLES = 40
_PHI = (0.0, float(np.pi / 3.0))           # limited-angle window


def _angles():
    return np.linspace(0.0, np.pi, _N_ANGLES, endpoint=False)


@pytest.fixture(scope="session")
def matrix_radon():
    """Real SVD-backed limited-angle operator. Building the system matrices
    needs astra; the truncated SVD needs scipy."""
    pytest.importorskip("astra")
    pytest.importorskip("scipy")
    from src.radon import MatrixRadonAdapter
    try:
        return MatrixRadonAdapter(
            resolution=_RES, angles=_angles(), det_count=_DET, phi=_PHI,
            svd_threshold=0.02, dx=1.0, estimate_norm=False,
            device=torch.device("cpu"), dtype=torch.float64, cache_dir=None,
        )
    except Exception as exc:                       # pragma: no cover - env dependent
        pytest.skip(f"could not build MatrixRadonAdapter: {exc}")


@pytest.fixture(scope="session")
def astra_radon():
    """Real astra-backed operator (FP/BP adjoint pair)."""
    pytest.importorskip("astra")
    from src.radon import AstraRadonAdapter
    try:
        return AstraRadonAdapter(
            resolution=_RES, angles=_angles(), det_count=_DET,
            dx=1.0, estimate_norm=False, device=torch.device("cpu"),
            dtype=torch.float64, phi=_PHI,
        )
    except Exception as exc:                       # pragma: no cover - env dependent
        pytest.skip(f"could not build AstraRadonAdapter: {exc}")


def _img(*shape):
    return torch.rand(*shape, dtype=torch.float64)


def _measured(radon, v):
    """The measured data of an image: proj_ran(A_la v) — the retained-range
    trace the reconstruction pipeline actually sees."""
    return radon.proj_ran(radon.forward_la(v))


# --------------------------------------------------------------------------- #
# MatrixRadonAdapter: exact operator identities (through proj_ran)
# --------------------------------------------------------------------------- #
def test_matrix_null_projection_is_idempotent_and_invisible(matrix_radon):
    r = matrix_radon
    x = _img(2, 1, _RES, _RES)
    p = r.proj_null_image(x)
    # orthogonal projector => idempotent
    assert torch.allclose(r.proj_null_image(p), p, atol=1e-9)
    # the null component carries no measured (retained-range) signal
    meas = torch.linalg.norm(_measured(r, p).reshape(-1))
    ref = torch.linalg.norm(_measured(r, x).reshape(-1)).clamp_min(1e-12)
    assert float(meas / ref) < 1e-8, f"proj_ran(A_la P_null x) should vanish, ratio {meas/ref}"


def test_matrix_decompose_error_is_orthogonal(matrix_radon):
    r = matrix_radon
    e = _img(2, 1, _RES, _RES) - 0.5
    e_ran, e_nul = r.decompose_error(e)
    # exact reconstruction e = e_ran + e_nul
    assert torch.allclose(e, e_ran + e_nul, atol=1e-9)
    # orthogonal split: <e_ran, e_nul> ~ 0
    ip = float((e_ran * e_nul).sum())
    denom = float(torch.linalg.norm(e.reshape(-1)) ** 2)
    assert abs(ip) / max(denom, 1e-12) < 1e-8
    # the null part carries no measured signal
    assert float(torch.linalg.norm(_measured(r, e_nul).reshape(-1))) < 1e-8


def test_attack_objective_null_range_with_real_operator(matrix_radon):
    r = matrix_radon
    pred, gt, clean = _img(2, 1, _RES, _RES), _img(2, 1, _RES, _RES), _img(2, 1, _RES, _RES)
    null = attack.attack_objective(pred, gt, clean, "null", 0.0, radon=r)
    rng = attack.attack_objective(pred, gt, clean, "range", 0.0, radon=r)
    e = pred - gt
    en = r.proj_null_image(e)
    assert torch.allclose(null, (en ** 2).mean(), atol=1e-10)
    # orthogonal channels sum to the total mean-square error
    assert torch.allclose((e ** 2).mean(), null + rng, atol=1e-9)


# --------------------------------------------------------------------------- #
# Real models through the real ModelAttackAdapter
# --------------------------------------------------------------------------- #
def _build_model(radon, name):
    from src.utils import build_models
    model = build_models([name], radon=radon)[name]
    return model.double().eval()      # match the float64 operator


def _pinv_adapter(radon, model):
    init = attack.InitReconstructor(init_method="pinv",
                                    summary={"operator_norm_A2": 1.0}, radon=radon)
    return attack.ModelAttackAdapter(model=model, init_reconstructor=init,
                                     projector=lambda y: radon.proj_ran(y),
                                     attack_init_mode="exact")


def test_nsn_correction_lives_in_nullspace(matrix_radon):
    """Training-independent NSN invariant: NSN(x) - x = P_null(UNet(x)) lies in
    null(A_la), so it cannot change the measured data. Holds for any weights."""
    r = matrix_radon
    nsn = _build_model(r, "nsn")
    x = _img(2, 1, _RES, _RES)
    with torch.no_grad():
        out = nsn(x)
        corr = out - x
    # correction invisible to the measured (retained-range) data ...
    assert float(torch.linalg.norm(_measured(r, corr).reshape(-1))) < 1e-6
    # ... i.e. the reconstruction is measurement-consistent with the input
    assert torch.allclose(_measured(r, out), _measured(r, x), atol=1e-6)


def test_model_attack_adapter_forward_shapes(matrix_radon):
    r = matrix_radon
    resnet = _build_model(r, "resnet")
    adapter = _pinv_adapter(r, resnet)
    x0 = _img(2, 1, _RES, _RES)
    y_clean = r.forward_la(x0)
    pred, x_init, y_used = adapter.forward(y_clean, mode="exact")
    assert pred.shape == (2, 1, _RES, _RES)
    assert x_init.shape == (2, 1, _RES, _RES)
    # exact pinv init == backward_la of the projected sinogram
    assert torch.allclose(x_init, r.backward_la(r.proj_ran(y_clean)), atol=1e-8)


@pytest.mark.parametrize("model_name", ["nsn", "resnet"])
def test_pgd_end_to_end_real_model(matrix_radon, model_name):
    """A few PGD steps through the real init + real (untrained) network. We only
    assert the mechanical contract — it runs, keeps the shape, respects the L2
    budget, and stays in the measured subspace — not a claim about untrained
    weights."""
    r = matrix_radon
    model = _build_model(r, model_name)
    adapter = _pinv_adapter(r, model)
    x0 = _img(2, 1, _RES, _RES)
    y_clean = r.proj_ran(r.forward_la(x0))
    with torch.no_grad():
        clean_pred, _, _ = adapter.forward(y_clean, mode="exact")
    eps = float(0.1 * attack.l2_norm_batch(y_clean).mean())
    res = attack.pgd_attack(adapter, x_gt=x0, y_clean=y_clean, clean_pred=clean_pred,
                            eps=eps, alpha=0.3 * eps, steps=3, restarts=1,
                            objective="null", shift_weight=0.0)
    assert res.y_adv.shape == y_clean.shape
    assert float(attack.l2_norm_batch(res.delta).max()) <= eps + 1e-6
    # the perturbation stays in the measured subspace (projector applied)
    assert torch.allclose(res.delta, r.proj_ran(res.delta), atol=1e-7)


# --------------------------------------------------------------------------- #
# AstraRadonAdapter: adjointness + limited-angle masking
# --------------------------------------------------------------------------- #
def test_astra_forward_backward_are_adjoint(astra_radon):
    r = astra_radon
    x = _img(2, 1, _RES, _RES)
    y = _img(2, 1, _N_ANGLES, _DET)
    lhs = float((r.forward(x) * y).sum())      # <A x, y>
    rhs = float((x * r.backward(y)).sum())      # <x, A^T y>
    scale = max(abs(lhs), abs(rhs), 1e-12)
    assert abs(lhs - rhs) / scale < 2e-3, f"FP/BP not adjoint: {lhs} vs {rhs}"


def test_astra_forward_la_masks_unmeasured_angles(astra_radon):
    r = astra_radon
    x = _img(2, 1, _RES, _RES)
    y_la = r.forward_la(x)
    # forward_la = forward * ran_mask, so the complement (null) angles are exactly 0
    assert float(r.proj_nsn(y_la).abs().max()) == 0.0


# =========================================================================== #
# Feature-level tests (formerly test_todos.py).
# =========================================================================== #



# --------------------------------------------------------------------------- #
# Training loss. The configurable data-fidelity loss (TODO 3) was removed with
# src.utils.reconstruction_loss: training is L2/MSE only, so what remains to
# check is mse_loss itself and the TV helper the attack objectives still use.
# --------------------------------------------------------------------------- #
def test_mse_loss_matches_the_training_objective():
    """train.py's loss_fn is written out inline as mean((pred-target)^2); this is
    the same quantity, and the attack side's l2 objective assumes it."""
    p, t = torch.randn(3, 1, 8, 8), torch.randn(3, 1, 8, 8)
    assert torch.allclose(mse_loss(p, t), ((p - t) ** 2).mean())


# --------------------------------------------------------------------------- #
# TODO 2 — cross-attack aggregation.
# --------------------------------------------------------------------------- #
def _make_summary(root, init, attack_name, obj, models):
    d = root / f"init_{init}" / attack_name
    d.mkdir(parents=True)
    summ = {"attack": attack_name, "objective": obj, "eps": 0.05, "models": {}}
    for name, base in models.items():
        m = {"num_examples": 64}
        for k in attack._AGGREGATE_METRICS:
            m[f"{k}_mean"] = base
            m[f"{k}_median"] = base * 0.5
        summ["models"][name] = m
    with open(d / "summary.json", "w") as f:
        json.dump(summ, f)


def test_aggregate_from_disk(tmp_path):
    root = tmp_path / "attacks_n0.05"
    _make_summary(root, "fbp", "adversarial", "mse", {"resnet": 1.0, "nsn": 0.5})
    _make_summary(root, "fbp", "adversarial_null", "null", {"resnet": 2.0})
    recs = attack.aggregate_from_disk(root)
    assert len(recs) == 3
    nsn = [r for r in recs if r["model"] == "nsn"][0]
    assert nsn["adv_rel_l2_mean"] == 0.5 and nsn["adv_rel_l2_median"] == 0.25
    assert nsn["init"] == "fbp" and nsn["attack"] == "adversarial"


def test_write_aggregate_summary(tmp_path):
    root = tmp_path / "attacks_n0.05"
    _make_summary(root, "fbp", "adversarial", "mse", {"resnet": 1.0})
    _make_summary(root, "pinv", "adversarial", "mse", {"resnet": 1.5})
    recs = attack.write_aggregate_summary(root)
    assert len(recs) == 2
    assert (root / "aggregate_summary.csv").exists()
    assert (root / "aggregate_summary.json").exists()
    rows = list(csv.DictReader(open(root / "aggregate_summary.csv")))
    assert len(rows) == 2
    assert "adv_rel_l2_mean" in rows[0] and "adv_rel_l2_median" in rows[0]
    order = [(r["init"], r["attack"], r["model"]) for r in rows]
    assert order == sorted(order)
    nested = json.load(open(root / "aggregate_summary.json"))
    assert set(nested) == {"fbp", "pinv"}


def test_write_aggregate_summary_empty(tmp_path):
    assert attack.write_aggregate_summary(tmp_path) == []
    assert not (tmp_path / "aggregate_summary.csv").exists()


def test_aggregate_missing_keys_become_nan(tmp_path):
    root = tmp_path / "a"
    d = root / "init_lw" / "adversarial"
    d.mkdir(parents=True)
    with open(d / "summary.json", "w") as f:
        json.dump({"attack": "adversarial", "objective": "mse", "eps": 0.05,
                   "models": {"resnet": {"num_examples": 5,
                                         "adv_rel_l2_mean": 3.0,
                                         "adv_rel_l2_median": 2.0}}}, f)
    recs = attack.aggregate_from_disk(root)
    assert recs[0]["adv_rel_l2_mean"] == 3.0
    assert math.isnan(recs[0]["adv_e_nul_l2_mean"])


# --------------------------------------------------------------------------- #
# TODO 1 — create_phantom_data.py --pinv_mode flag.
# Imports odl/dival (heavy), so gated separately.
# --------------------------------------------------------------------------- #
def _parse_cpd(argv):
    pytest.importorskip("odl")
    pytest.importorskip("dival")
    import src.create_phantom_data as cpd
    old = sys.argv
    sys.argv = ["create_phantom_data"] + argv
    try:
        return cpd.parse_args()
    finally:
        sys.argv = old


def test_pinv_mode_default_thresholded():
    assert _parse_cpd([]).pinv_mode == "thresholded"


def test_pinv_mode_unthresholded():
    assert _parse_cpd(["--pinv_mode", "unthresholded"]).pinv_mode == "unthresholded"


def test_pinv_mode_invalid_rejected():
    pytest.importorskip("odl")
    pytest.importorskip("dival")
    import src.create_phantom_data as cpd
    old = sys.argv
    sys.argv = ["create_phantom_data", "--pinv_mode", "bogus"]
    try:
        with pytest.raises(SystemExit):
            cpd.parse_args()
    finally:
        sys.argv = old


# --------------------------------------------------------------------------- #
# TODOs 4–6 — data-consistency overview, null-structure analysis, attack
# overview.  All read only saved artifacts; they need matplotlib + numpy but no
# trained model or radon operator.
# --------------------------------------------------------------------------- #
def _vis():
    """Import the visualisation module with a headless backend, skipping if
    matplotlib is unavailable."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import src.visualisations as V
    return V


def _make_attack_tree(root, init, attack_name, models):
    """Write a summary.json + per_sample_metrics.csv + examples.npz per model,
    mirroring what the attack suite emits, for one (init, attack)."""
    import numpy as np
    d = root / f"init_{init}" / attack_name
    d.mkdir(parents=True)
    summ = {"attack": attack_name, "objective": "mse", "eps": 0.05, "models": {}}
    rng = np.random.default_rng(0)
    for name, (nul_frac, cons) in models.items():
        md = d / name
        md.mkdir()
        rows = [{"adv_e_nul_frac": nul_frac, "clean_e_nul_frac": 0.3,
                 "adv_consistency_rel": cons, "clean_consistency_rel": 0.01,
                 "adv_rel_l2": 0.4, "clean_rel_l2": 0.1} for _ in range(20)]
        with open(md / "per_sample_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(md / "examples.npz", ex0__adv_pred=rng.random((16, 16)))
        summ["models"][name] = {"num_examples": 20, "adv_rel_l2_median": 0.4,
                                "adv_ssim_median": 0.7, "adv_e_nul_frac_median": nul_frac,
                                "adv_consistency_rel_median": cons}
    with open(d / "summary.json", "w") as f:
        json.dump(summ, f)


def test_collect_attack_overview_and_markdown(tmp_path):
    V = _vis()
    root = tmp_path / "attacks_n0.05"
    _make_attack_tree(root, "fbp", "adversarial", {"resnet": (0.6, 0.25), "nsn": (0.95, 0.02)})
    _make_attack_tree(root, "fbp", "adversarial_null", {"resnet": (0.7, 0.30), "nsn": (0.97, 0.02)})
    recs = V.collect_attack_overview(root)
    assert len(recs) == 4
    assert {r["model"] for r in recs} == {"resnet", "nsn"}
    V.write_overview_markdown(root, recs, root / "overview.md")
    md = (root / "overview.md").read_text()
    assert "| attack | model |" in md and "adversarial" in md and "nsn" in md


def test_write_overview_markdown_empty(tmp_path):
    V = _vis()
    V.write_overview_markdown(tmp_path, [], tmp_path / "overview.md")
    assert "No attack summaries" in (tmp_path / "overview.md").read_text()


def test_save_attack_overview_produces_montage_and_index(tmp_path):
    V = _vis()
    root = tmp_path / "attacks_n0.05"
    _make_attack_tree(root, "fbp", "adversarial", {"resnet": (0.6, 0.25), "nsn": (0.95, 0.02)})
    V.save_attack_overview(root)
    assert (root / "overview.md").exists()
    assert (root / "overview_fbp.png").exists()


def test_save_consistency_overview_and_ghost_structure(tmp_path):
    V = _vis()
    root = tmp_path / "attacks_n0.05"
    _make_attack_tree(root, "fbp", "adversarial", {"resnet": (0.6, 0.25), "nsn": (0.95, 0.02)})
    init_dir = root / "init_fbp"
    rows_by_model = {
        m: [{k: float(v) for k, v in r.items()}
            for r in csv.DictReader(open(init_dir / "adversarial" / m / "per_sample_metrics.csv"))]
        for m in ("resnet", "nsn")
    }
    V.save_ghost_structure_plot(init_dir / "adversarial", rows_by_model, 0.05, "adversarial")
    assert (init_dir / "adversarial" / "ghost_structure.png").exists()
    V.save_consistency_overview(init_dir, {"adversarial": rows_by_model}, 0.05)
    assert (init_dir / "consistency_overview.png").exists()


def test_clean_consistency_in_aggregate_metrics():
    assert "clean_consistency_rel" in attack._AGGREGATE_METRICS
    assert "adv_consistency_rel" in attack._AGGREGATE_METRICS


# --------------------------------------------------------------------------- #
# TODO 9 — dpnsn / dpnsn_res coverage.
# --------------------------------------------------------------------------- #
def test_build_models_all_four():
    from src.utils import build_models

    class _FakeRadon:  # only stored by the wrappers, never called at init
        pass

    models = build_models(["resnet", "nsn"],
                          radon=_FakeRadon())
    assert set(models) == {"resnet", "nsn"}


def test_detect_suite_models_finds_all_four(tmp_path):
    init = "fbp"
    ck = tmp_path / f"init_{init}" / "checkpoints"
    ck.mkdir(parents=True)
    for m in ("resnet", "nsn"):
        (ck / f"{m}_best.pt").write_bytes(b"x")
    found = attack.detect_suite_models(str(tmp_path), init)
    assert set(found) == {"resnet", "nsn"}


# --------------------------------------------------------------------------- #
# TODO 7 — worst-case example presentation.
# --------------------------------------------------------------------------- #
def _make_worst_bundle(model_dir, rel_l2):
    import numpy as np
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / "worst.npz",
                        ex0__adv_pred=np.random.default_rng(0).random((16, 16)))
    (model_dir / "worst.json").write_text(json.dumps(
        [{"worst_score": 9.0, "m_adv_pred": {"rel_l2": rel_l2, "psnr": 10.0, "ssim": 0.2}}]))


def test_worst_rel_l2_reads_bundle(tmp_path):
    V = _vis()
    md = tmp_path / "init_fbp" / "adversarial" / "resnet"
    _make_worst_bundle(md, 0.91)
    rec = {"worst_json": str(md / "worst.json")}
    assert abs(V._worst_rel_l2(rec) - 0.91) < 1e-9
    assert math.isnan(V._worst_rel_l2({"worst_json": str(tmp_path / "nope.json")}))


def test_attack_overview_includes_worst_montage(tmp_path):
    V = _vis()
    root = tmp_path / "attacks_n0.05"
    _make_attack_tree(root, "fbp", "adversarial", {"resnet": (0.6, 0.25), "nsn": (0.95, 0.02)})
    for m in ("resnet", "nsn"):
        _make_worst_bundle(root / "init_fbp" / "adversarial" / m, 0.9)
    V.save_attack_overview(root)
    assert (root / "overview_worst_fbp.png").exists()
    md = (root / "overview.md").read_text()
    assert "worst rel-L2" in md


# --------------------------------------------------------------------------- #
# TODO 8 — per-epoch weight tracking + epoch-attack study.
# --------------------------------------------------------------------------- #
def test_detect_epoch_checkpoints_orders_and_ignores_best(tmp_path):
    ck = tmp_path / "init_fbp" / "checkpoints"
    ck.mkdir(parents=True)
    for e in (10, 1, 5, 2):
        (ck / f"nsn_epoch{e:03d}.pt").write_bytes(b"x")
    (ck / "nsn_best.pt").write_bytes(b"x")  # must NOT be treated as an epoch
    got = attack.detect_epoch_checkpoints(str(tmp_path), "fbp", "nsn")
    assert [e for e, _ in got] == [1, 2, 5, 10]


def test_load_epoch_history(tmp_path):
    ck = tmp_path / "init_fbp" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "nsn_history.json").write_text(json.dumps({
        "best_epoch": 5,
        "history": [{"epoch": e, "train": 1.0 / e, "val": 0.5 / e} for e in (1, 2, 5)]}))
    hist, best = attack.load_epoch_history(str(tmp_path), "fbp", "nsn")
    assert best == 5
    assert set(hist) == {1, 2, 5}
    assert hist[2] == (0.5, 0.25)


def test_load_epoch_history_missing(tmp_path):
    hist, best = attack.load_epoch_history(str(tmp_path), "fbp", "nsn")
    assert hist == {} and best is None


def test_epoch_study_plot(tmp_path):
    V = _vis()
    sd = tmp_path / "epoch_study"
    sd.mkdir()
    rows = [{"epoch": e, "train_loss": 1.0 / e, "val_loss": 0.5 / e,
             "is_best": 1 if e == 3 else 0, "clean_rel_l2_median": 0.1,
             "adv_rel_l2_median": 0.2 + 0.02 * e, "rel_l2_ratio_median": 2 + e,
             "adv_e_nul_frac_median": 0.8, "adv_consistency_rel_median": 0.02}
            for e in (1, 2, 3, 4)]
    with open(sd / "fbp_nsn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    V.save_epoch_study_plots(tmp_path)
    assert (sd / "fbp_nsn.png").exists()
    assert len(V.read_epoch_study_csv(sd / "fbp_nsn.csv")) == 4


# --------------------------------------------------------------------------- #
# Run-log follow-ups (job 20585). Each test pins one defect the six-day full run
# exposed, so a future run cannot silently reintroduce it.
# --------------------------------------------------------------------------- #

def test_suite_step_size_scales_with_the_budget():
    """The classic 2.5*eps*||y||/steps, in the same units as the ball radius."""
    a = attack.suite_step_size(eps_nominal=0.01, mean_sino_norm=100.0, steps=50)
    assert a == pytest.approx(2.5 * 0.01 * 100.0 / 50)
    assert attack.suite_step_size(0.02, 100.0, 50) == pytest.approx(2 * a)


def test_suite_eps_batch_is_a_relative_l2_budget():
    """A saturated ball carries exactly eps*||y_i|| of L2 mass, per sample."""
    torch.manual_seed(0)
    y = torch.randn(3, 1, 180, 182)
    eps = attack.suite_eps_batch(y, 0.01)
    delta = attack.proj_l2_ball(torch.randn_like(y) * 1e6, eps)
    rel = delta.reshape(3, -1).norm(dim=1) / y.reshape(3, -1).norm(dim=1)
    assert torch.allclose(rel, torch.full((3,), 0.01), atol=1e-5)


# 5 — consistency against the *clean* measurement.
def test_adv_consistency_vs_clean_is_not_trivially_zero():
    """adv_consistency_rel scores x_adv against y_adv, which a hard-consistent
    model drives to ~0 whatever the attack did. Against the true y it must not."""
    radon = _IdentityRadon()
    x = torch.zeros(1, 1, 8, 8)          # >=7x7: ssim() needs a 7-wide window
    y_clean = torch.zeros(1, 1, 8, 8)
    y_adv = torch.full((1, 1, 8, 8), 0.5)
    pred_adv = y_adv.clone()          # perfectly consistent with the *attacked* y
    row = attack.evaluate_batch(
        x_gt=x, clean_init=x, clean_y=y_clean, clean_pred=x,
        adv_init=pred_adv, adv_y=y_adv, adv_pred=pred_adv,
        delta=y_adv - y_clean, success_mse_factor=2.0, radon=radon)[0]
    assert row["adv_consistency_rel"] == pytest.approx(0.0, abs=1e-9)
    assert row["adv_consistency_vs_clean_rel"] > 0.1


def test_adv_consistency_vs_clean_in_aggregate_metrics():
    assert "adv_consistency_vs_clean_rel" in attack._AGGREGATE_METRICS


# --- tiny stand-in operators -------------------------------------------------
class _IdentityRadon:
    """A = I: forward_la and proj_ran are the identity, so the data-consistency
    residual reduces to ||pred - y|| / ||y||. With a trivial null space the error
    decomposition is all range, no null."""
    @staticmethod
    def forward_la(x):
        return x.clone()

    @staticmethod
    def proj_ran(y):
        return y.clone()

    @staticmethod
    def proj_null_image(x):
        return torch.zeros_like(x)

    @staticmethod
    def decompose_error(e, iters=None, tol=None):
        return e.clone(), torch.zeros_like(e)


# 8 / tooling — render progress reporting (the render task's sign of life).
def test_count_render_steps_counts_attacks_plus_tree_steps(tmp_path):
    V = _vis()
    for init in ("init_pinv", "init_fbp"):
        for attack_name in ("adversarial", "adversarial_null", "adversarial_range"):
            d = tmp_path / init / attack_name
            d.mkdir(parents=True)
            (d / "summary.json").write_text("{}")
    (tmp_path / "init_pinv" / "not_an_attack").mkdir()   # no summary.json
    # 2 inits x 3 attacks, + attack-overview + epoch-study steps
    assert V.count_render_steps(tmp_path) == 8


def test_render_tree_emits_parseable_progress(tmp_path, capsys):
    """The Slurm render task's only sign of life for 35+ minutes."""
    import re
    V = _vis()
    d = tmp_path / "init_pinv" / "adversarial"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"attack": "adversarial", "models": {}}))
    V.render_tree(tmp_path)
    lines = [l for l in capsys.readouterr().out.splitlines()
             if l.startswith("[visualise][progress]")]
    assert lines, "no progress lines emitted"
    for line in lines:
        assert re.match(r"^\[visualise\]\[progress\] \d+/\d+ \S", line), line
    # one attack dir + the two tree-level steps (overview, epoch study)
    assert lines[0].startswith("[visualise][progress] 1/3 init_pinv/adversarial")
    assert [l.split()[1] for l in lines] == ["1/3", "2/3", "3/3"]


# Tooling — PowerShell source hygiene.
def test_powershell_scripts_are_ascii_outside_comments():
    """Non-ASCII in a .ps1 string literal can silently unbalance quotes.

    Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI. An em-dash is UTF-8
    E2 80 94, and byte 0x94 is a right double-quote in CP1252 — so a dash inside
    a double-quoted string decodes to a stray quote and the parse breaks several
    hundred lines later, nowhere near the cause. Comments are exempt: a spurious
    quote there is harmless, which is why only string literals bite.
    """
    from pathlib import Path
    offenders = []
    for ps1 in sorted(Path(__file__).parent.glob("*.ps1")):
        for n, line in enumerate(ps1.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            bad = {c for c in line if ord(c) > 127}
            if bad:
                offenders.append(f"{ps1.name}:{n}: {''.join(sorted(bad))!r} in {line.strip()[:70]}")
    assert not offenders, (
        "non-ASCII outside comments in PowerShell source (use ASCII '--' etc.):\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------------------------- #
# Source hygiene — dependency boundaries the architecture depends on.
# --------------------------------------------------------------------------- #
def test_attack_module_does_not_import_matplotlib():
    """src/attack.py documents that it never draws a figure. That was once false:
    it imported the bundle writers from src/visualisations.py, so every array
    task paid for matplotlib and risked a backend failure on a headless node.
    """
    import subprocess
    code = ("import sys; import src.attack; "
            "sys.exit(1 if any(m.startswith('matplotlib') for m in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0, \
        "importing src.attack pulled in matplotlib"


def test_artifacts_module_is_dependency_free():
    """The on-disk contract is shared by the attack driver and the renderer, so
    it must not drag either side's heavy dependency into the other."""
    import subprocess
    code = ("import sys; import src.artifacts; "
            "sys.exit(1 if ('torch' in sys.modules or "
            "any(m.startswith('matplotlib') for m in sys.modules)) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0, \
        "src.artifacts pulled in torch or matplotlib"


# =========================================================================== #
# Models — the four architectures under study (src/wrappers.py).
#
# These were entirely untested: RESNET, DPNSN and DPNSN_RES were never even
# named by the suite, yet the paper's whole comparison rests on how their
# corrections are constrained. The properties below are the definitions, not
# implementation details -- if one breaks, the results mean something else.
# =========================================================================== #
class _ConstUNet(torch.nn.Module):
    """A 'network' returning a fixed correction, so each wrapper's algebra can be
    checked exactly rather than approximately."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return self.value.expand_as(x).clone()


class ModelRadon:
    """Operator stub following the *real* sinogram convention.

    MatrixRadonAdapter.forward_la returns a full-shape sinogram with the
    unmeasured rows zeroed, so forward() and forward_la() outputs can be added
    and subtracted. FakeRadon above instead emits only the measured rows, which
    is fine for the attack tests but makes DPNSN_RES -- which computes
    forward(res) - y_delta -- a shape error. Hence a second stub, rather than a
    change to the one that forty-odd existing tests depend on.
    """

    def __init__(self, img=16, seed=0, dtype=torch.float64):
        self.IMG = img
        self.N = img * img
        self.M_FULL = (3 * self.N) // 2
        self.LA_ROWS = (3 * self.N) // 4
        A, A_la, P_null = _build_operator(self.N, self.M_FULL, self.LA_ROWS, seed)
        self.dtype = dtype
        self._A = torch.tensor(A, dtype=dtype)
        self._P = torch.tensor(P_null, dtype=dtype)
        self._A_pinv = torch.tensor(np.linalg.pinv(A), dtype=dtype)
        self._A_la_pinv = torch.tensor(np.linalg.pinv(A_la), dtype=dtype)
        self.norm_A2 = 1.0

    def _flat(self, v):
        return v.reshape(v.shape[0], -1).to(self.dtype)

    def forward(self, x):
        y = self._flat(x) @ self._A.T
        return y.reshape(x.shape[0], 1, self.M_FULL, 1).to(x.dtype)

    def forward_la(self, x):
        return self.proj_ran(self.forward(x))

    def proj_ran(self, y):
        out = torch.zeros_like(y)
        flat_out, flat_in = out.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1)
        flat_out[:, : self.LA_ROWS] = flat_in[:, : self.LA_ROWS]
        return flat_out.reshape(y.shape)

    def proj_nsn(self, y):
        return y - self.proj_ran(y)

    def proj_null_image(self, v):
        return (self._flat(v) @ self._P.T).reshape(v.shape).to(v.dtype)

    def fbp(self, y):
        x = self._flat(y) @ self._A_pinv.T
        return x.reshape(y.shape[0], 1, self.IMG, self.IMG).to(y.dtype)

    def fbp_la(self, y):
        rows = y.reshape(y.shape[0], -1)[:, : self.LA_ROWS].to(self.dtype)
        return (rows @ self._A_la_pinv.T).reshape(
            y.shape[0], 1, self.IMG, self.IMG).to(y.dtype)

    def decompose_error(self, e, iters=50, tol=1e-6):
        e_nul = self.proj_null_image(e)
        return e - e_nul, e_nul


def _unet_returning(radon, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, 1, radon.IMG, radon.IMG, generator=g, dtype=radon.dtype) * scale
    return _ConstUNet(v)


@pytest.fixture(scope="module")
def model_radon():
    # 16x16: build_models wires a real UNet, which downsamples four times and
    # cannot take the 6x6 images the attack fixtures use.
    return ModelRadon(img=16)


def test_model_radon_stub_matches_the_real_sinogram_convention(model_radon):
    """Guards the stub itself: forward_la must be full-shape with the unmeasured
    rows zeroed, or every model property below is verified against a fiction."""
    x = torch.randn(2, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    y, y_la = model_radon.forward(x), model_radon.forward_la(x)
    assert y.shape == y_la.shape
    la = model_radon.LA_ROWS
    flat, flat_full = y_la.reshape(2, -1), y.reshape(2, -1)
    assert torch.allclose(flat[:, la:], torch.zeros_like(flat[:, la:]))
    assert torch.allclose(flat[:, :la], flat_full[:, :la])
    # A_la annihilates the image-domain null space, which is what makes a ghost
    # invisible and an NSN correction free.
    n = model_radon.proj_null_image(x)
    assert torch.allclose(model_radon.forward_la(n), torch.zeros_like(y), atol=1e-8)


def test_proj_l2_ball_is_identity_inside_and_shrinks_outside():
    from src.wrappers import _proj_l2_ball
    v = torch.tensor([[[[3.0, 4.0]]]])                        # norm 5
    assert torch.allclose(_proj_l2_ball(v, 10.0), v)          # inside: untouched
    out = _proj_l2_ball(v, 1.0)
    assert out.norm().item() == pytest.approx(1.0, rel=1e-6)  # outside: on the sphere
    assert torch.allclose(out, v / 5.0)                       # direction preserved


def test_proj_l2_ball_is_per_sample():
    """A batched projection must not let one large sample shrink the others."""
    from src.wrappers import _proj_l2_ball
    v = torch.stack([torch.full((1, 2, 2), 0.1), torch.full((1, 2, 2), 100.0)])
    out = _proj_l2_ball(v, 1.0)
    assert torch.allclose(out[0], v[0])
    assert out[1].norm().item() == pytest.approx(1.0, rel=1e-6)


def test_resnet_is_exactly_input_plus_correction(model_radon):
    from src.wrappers import RESNET
    unet = _unet_returning(model_radon)
    x = torch.randn(3, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    assert torch.allclose(RESNET(unet)(x, None), x + unet.value.expand_as(x))


def test_resnet_is_not_data_consistent(model_radon):
    """The control case: RESNET's correction is unconstrained, so it may move the
    measured channel. If this ever passed, the NSN comparison would be vacuous."""
    from src.wrappers import RESNET
    x = torch.randn(2, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    out = RESNET(_unet_returning(model_radon))(x, None)
    assert not torch.allclose(model_radon.forward_la(out),
                              model_radon.forward_la(x), atol=1e-6)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_nsn_output_is_data_consistent_by_construction(model_radon, scale):
    """The defining NSN property: A_la f(x) == A_la x for *any* network output.

    This is why an NSN's measurement residual stays ~0 whatever an attack does,
    and hence why adv_consistency_rel had to be measured against the clean y to
    say anything (see test_adv_consistency_vs_clean_is_not_trivially_zero)."""
    from src.wrappers import NSN
    x = torch.randn(4, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    out = NSN(_unet_returning(model_radon, scale=scale), model_radon)(x, None)
    assert torch.allclose(model_radon.forward_la(out),
                          model_radon.forward_la(x), atol=1e-7)


def test_nsn_correction_lies_in_the_null_space(model_radon):
    from src.wrappers import NSN
    x = torch.randn(2, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    correction = NSN(_unet_returning(model_radon), model_radon)(x, None) - x
    # already in null(A_la), so projecting again must change nothing
    assert torch.allclose(model_radon.proj_null_image(correction), correction, atol=1e-8)


def test_dpnsn_measured_correction_is_capped_by_beta(model_radon):
    """DPNSN's whole point: however large the network output, the measured part
    of the correction is clipped to the L2 ball of radius beta."""
    from src.wrappers import DPNSN, _proj_l2_ball
    x = torch.zeros(2, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    beta = 0.05
    unet = _unet_returning(model_radon, scale=1e4)
    out = DPNSN(unet, model_radon, beta)(x, None)
    # reproduce the measured branch and confirm the clip actually bound it
    res = unet.value.expand(2, 1, model_radon.IMG, model_radon.IMG)
    y = model_radon.forward(res)
    clipped = _proj_l2_ball(model_radon.proj_ran(y), beta)
    assert clipped.reshape(2, -1).norm(dim=1).max().item() <= beta * (1 + 1e-9)
    expected = x + model_radon.fbp_la(clipped) + model_radon.fbp(model_radon.proj_nsn(y))
    assert torch.allclose(out, expected, atol=1e-8)


def test_dpnsn_larger_beta_permits_a_larger_measured_correction(model_radon):
    from src.wrappers import DPNSN
    x = torch.zeros(1, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    unet = _unet_returning(model_radon, scale=1e4)
    tight = DPNSN(unet, model_radon, 0.01)(x, None)
    loose = DPNSN(unet, model_radon, 10.0)(x, None)
    assert (loose - x).norm().item() > (tight - x).norm().item()


def test_dpnsn_res_uses_the_measured_sinogram(model_radon):
    """DPNSN_RES is the only model whose output depends on y_delta; were that
    argument ignored, its data-proximal term would be doing nothing."""
    from src.wrappers import DPNSN_RES
    x = torch.randn(2, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    model = DPNSN_RES(_unet_returning(model_radon), model_radon, beta=0.5)
    y1 = model_radon.forward_la(x)
    assert not torch.allclose(model(x, y1), model(x, y1 + 1.0))


def test_dpnsn_res_requires_y_delta(model_radon):
    """Unlike the other three it has no default, so a caller that forgets the
    sinogram fails loudly instead of silently reconstructing from nothing."""
    from src.wrappers import DPNSN_RES
    x = torch.randn(1, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    with pytest.raises(TypeError):
        DPNSN_RES(_unet_returning(model_radon), model_radon, beta=0.5)(x)


@pytest.mark.parametrize("name", ["resnet", "nsn", "dpnsn", "dpnsn_res"])
def test_every_model_preserves_shape_and_is_differentiable(model_radon, name):
    """Preconditions for both training and attacking: PGD differentiates the
    reconstruction with respect to the sinogram, so a broken graph on any one
    model would silently produce zero-gradient 'attacks'."""
    from src.utils import build_models
    model = build_models([name], radon=model_radon)[name].to(torch.float64)
    x = torch.randn(2, 1, model_radon.IMG, model_radon.IMG,
                    dtype=model_radon.dtype, requires_grad=True)
    y = model_radon.forward_la(x).detach()
    out = model(x, y)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0, "zero gradient: an attack here would be a no-op"


# =========================================================================== #
# UNet building blocks (src/unet.py).
#
# Standard blocks, but every wrapper above is "x + N(x)" with N this network, so
# a shape or skip-alignment mistake here changes what the correction *is*. The
# properties below are the ones the wrappers rely on: spatial size in == out,
# the encoder halves and the decoder doubles, and the skip concatenation lines
# up even when a level has an odd spatial size.
# =========================================================================== #
def test_double_conv_keeps_spatial_size_and_sets_channels():
    from src.unet import DoubleConv
    x = torch.randn(2, 3, 9, 7)
    out = DoubleConv(3, 8)(x)
    assert out.shape == (2, 8, 9, 7)          # padding=1 on 3x3 => same size


def test_double_conv_output_is_nonnegative():
    """It ends in a ReLU, so every sign in a model's correction comes from
    OutConv's 1x1 — which is why that layer is the only one with a bias."""
    from src.unet import DoubleConv
    out = DoubleConv(1, 4)(torch.randn(2, 1, 6, 6))
    assert (out >= 0).all()


def test_double_conv_mid_channels_defaults_to_out_channels():
    from src.unet import DoubleConv
    default, explicit = DoubleConv(1, 8), DoubleConv(1, 8, mid_channels=3)
    assert default.net[0].out_channels == 8
    assert explicit.net[0].out_channels == 3 and explicit.net[2].out_channels == 8


def test_down_halves_the_spatial_size():
    from src.unet import Down
    assert Down(2, 5)(torch.randn(1, 2, 8, 12)).shape == (1, 5, 4, 6)


def test_up_doubles_and_matches_the_skip():
    from src.unet import Up
    x1 = torch.randn(1, 8, 4, 4)              # coarse level
    x2 = torch.randn(1, 4, 8, 8)              # skip from the encoder
    assert Up(12, 6)(x1, x2).shape == (1, 6, 8, 8)


def test_up_pads_an_odd_sized_skip():
    """An odd level cannot be recovered by a x2 upsample, so Up pads to the
    skip's size. Without it the concatenation is a shape error, which is the
    failure mode for any image size that is not a multiple of 16."""
    from src.unet import Up
    x1 = torch.randn(1, 8, 4, 4)              # upsamples to 8x8
    x2 = torch.randn(1, 4, 9, 9)              # encoder level was 9x9
    assert Up(12, 6)(x1, x2).shape == (1, 6, 9, 9)


def test_up_without_bilinear_uses_a_learned_upsample():
    from src.unet import Up
    up = Up(16, 8, bilinear=False)
    assert isinstance(up.up, torch.nn.ConvTranspose2d)
    x1, x2 = torch.randn(1, 16, 4, 4), torch.randn(1, 8, 8, 8)
    assert up(x1, x2).shape == (1, 8, 8, 8)


def test_out_conv_is_pointwise():
    """A 1x1 convolution acts on each pixel independently, so permuting the
    pixels must permute the output — the property that makes the final layer a
    per-pixel channel mix rather than more spatial filtering."""
    from src.unet import OutConv
    conv = OutConv(4, 1)
    x = torch.randn(1, 4, 5, 5)
    flipped = torch.flip(x, dims=[-1, -2])
    assert torch.allclose(conv(flipped), torch.flip(conv(x), dims=[-1, -2]), atol=1e-6)


def test_unet_preserves_shape_and_passes_gradient():
    """The contract every wrapper assumes: N(x) is the same shape as x, and the
    whole path is differentiable so PGD can attack through it."""
    from src.unet import UNet
    net = UNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    out = net(x)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


# =========================================================================== #
# Pipeline setup — what every run mode resolves before doing any work.
#
# prepare_run / build_init_inputs are shared by the attack suite, the epoch
# study. A mistake here does not crash: it silently changes
# which data, which operator or which output directory a six-day job uses.
# =========================================================================== #
def _summary_dict(**over):
    d = {"dataset": "ellipses", "img_size": 8, "num_angles": 12, "det_count": 10,
         "angles": [0.0] * 12, "dx": 1.0, "phi": [0.0, 2.0], "matrix_mode": 1,
         "noise_sigma_rel": 0.02, "mean_norm_y": 7.0,
         "mean_norm_y_minus_y_delta": 0.5}
    d.update(over)
    return d


def _data_root(tmp_path, inits=("pinv", "fbp"), **over):
    """Minimal data directory: summary.json plus one .npy per init folder, which
    is all detect_data_inits looks at."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(_summary_dict(**over)))
    for init in inits:
        d = root / init
        d.mkdir(exist_ok=True)
        np.save(d / "00000.npy", np.zeros((8, 8), dtype=np.float32))
    return root


def _prep_args(root, **over):
    a = dict(data_root=str(root), model_dir=None, out_dir=None, type=None, init=None,
             seed=0, fp64=False, sparse_radon=False, suite_eps=None)
    a.update(over)
    return argparse.Namespace(**a)


def test_load_summary_reads_the_data_root(tmp_path):
    root = _data_root(tmp_path)
    assert attack.load_summary(str(root))["num_angles"] == 12


def test_detect_data_inits_finds_only_populated_folders(tmp_path):
    root = _data_root(tmp_path, inits=("pinv",))
    (root / "fbp").mkdir()                     # exists but holds no .npy
    got = attack.detect_data_inits(str(root))
    assert got == ["pinv"], "an empty init folder must not be treated as data"


def test_detect_data_inits_is_ordered_deterministically(tmp_path):
    """Init order decides the order of every downstream artifact, so it must not
    depend on filesystem enumeration."""
    root = _data_root(tmp_path, inits=("pinv", "fbp"))
    assert attack.detect_data_inits(str(root)) == attack.detect_data_inits(str(root))


def test_prepare_run_resolves_the_shared_context(tmp_path, monkeypatch):
    root = _data_root(tmp_path)
    monkeypatch.setattr(attack, "build_radon", lambda *a, **k: object())
    s = attack.prepare_run(_prep_args(root))
    assert s.example == "ellipses"
    assert s.beta == 0.5 and s.noise_rel == 0.02 and s.mean_sino_norm == 7.0
    assert set(s.inits) == {"pinv", "fbp"}


def test_prepare_run_maps_rectangles_onto_the_ellipses_loader(tmp_path, monkeypatch):
    """Rectangles share the on-disk layout; picking a 'rectangles' loader would
    fail, so the mapping is deliberate and worth pinning."""
    root = _data_root(tmp_path, dataset="rectangles")
    monkeypatch.setattr(attack, "build_radon", lambda *a, **k: object())
    assert attack.prepare_run(_prep_args(root)).example == "ellipses"


def test_prepare_run_out_root_defaults_to_the_noise_level(tmp_path, monkeypatch):
    root = _data_root(tmp_path)
    monkeypatch.setattr(attack, "build_radon", lambda *a, **k: object())
    assert attack.prepare_run(_prep_args(root)).out_root.name == "attacks_n0.02"
    explicit = attack.prepare_run(_prep_args(root, out_dir="somewhere"))
    assert explicit.out_root.name == "somewhere"


def test_prepare_run_honours_an_explicit_init(tmp_path, monkeypatch):
    root = _data_root(tmp_path)
    monkeypatch.setattr(attack, "build_radon", lambda *a, **k: object())
    assert attack.prepare_run(_prep_args(root, init="PINV")).inits == ["pinv"]


def test_prepare_run_rejects_a_data_root_without_inits(tmp_path, monkeypatch):
    """Silently looping zero times is how the epoch study used
    to fail; all three modes must now say so."""
    root = _data_root(tmp_path, inits=())
    monkeypatch.setattr(attack, "build_radon", lambda *a, **k: object())
    with pytest.raises(FileNotFoundError):
        attack.prepare_run(_prep_args(root))


def test_prepare_run_requires_a_data_root():
    with pytest.raises(ValueError):
        attack.prepare_run(_prep_args("", data_root=None))


@pytest.mark.parametrize("fp64,expected", [(True, torch.float64), (False, torch.float32)])
def test_prepare_run_passes_the_dtype_through(tmp_path, monkeypatch, fp64, expected):
    root = _data_root(tmp_path)
    seen = {}
    monkeypatch.setattr(attack, "build_radon",
                        lambda summary, **k: seen.update(k) or object())
    attack.prepare_run(_prep_args(root, fp64=fp64))
    assert seen["dtype"] is expected
    assert seen["dense"] is True          # --sparse-radon off => dense operators


def test_build_input_cache_stops_at_max_samples():
    """The budget bounds a multi-hour run; an off-by-one here silently changes
    every headline number's sample count."""
    batches = [(torch.zeros(4, 1, 4, 4), torch.zeros(4, 1, 4, 4), torch.zeros(4, 1, 3, 3))
               for _ in range(10)]
    cache = attack.build_input_cache(lambda y: y, batches, max_samples=6,
                                     device=torch.device("cpu"))
    assert sum(c[0].shape[0] for c in cache) >= 6
    assert len(cache) == 2, "should stop as soon as the budget is met"


def test_build_input_cache_applies_the_range_projector():
    """Every run projects the noisy sinogram onto range(A_la) first; skipping it
    would attack a measurement the operator cannot produce."""
    batches = [(torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 3, 3))]
    cache = attack.build_input_cache(lambda y: y * 0.0, batches, max_samples=2,
                                     device=torch.device("cpu"))
    assert torch.allclose(cache[0][2], torch.zeros_like(cache[0][2]))


# =========================================================================== #
# Null-restricted Lipschitz estimate.
#
# This produces a headline claim ("NSN < 1 < ResNet") and had no test at all.
# The cases below have an analytically known answer, so the estimator is checked
# against arithmetic rather than against itself.
# =========================================================================== #
class _GainModel(torch.nn.Module):
    """f(x) = x + gain * P_N(x).

    P_N is idempotent, so the null-restricted correction g(x) = P_N(f(x) - x)
    equals gain * P_N(x): a linear map whose operator norm on null(A_la) is
    exactly ``gain``. That is the number the estimator must return.
    """

    def __init__(self, radon, gain):
        super().__init__()
        self.radon, self.gain = radon, gain

    def forward(self, x, y=None):
        return x + self.gain * self.radon.proj_null_image(x)


def _lipschitz_cache(radon, n=2):
    x = torch.randn(n, 1, radon.IMG, radon.IMG, dtype=radon.dtype)
    return [(x, x.clone(), radon.forward_la(x))]


@pytest.mark.parametrize("gain", [0.5, 1.0, 2.5])
def test_lipschitz_recovers_a_known_null_space_gain(model_radon, gain):
    res = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, gain),
        clean_cache=_lipschitz_cache(model_radon), radon=model_radon,
        n_samples=2, n_iter=12)
    assert res["mean"] == pytest.approx(gain, rel=0.05), res
    assert res["max"] == pytest.approx(gain, rel=0.05)
    assert res["n"] == 2


def test_lipschitz_is_zero_for_a_model_that_does_nothing(model_radon):
    """An identity model has no learned correction, so its null-space gain is 0.
    A non-zero answer here would mean the estimator is measuring the operator
    rather than the network."""
    res = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, 0.0),
        clean_cache=_lipschitz_cache(model_radon), radon=model_radon,
        n_samples=2, n_iter=8)
    assert res["mean"] == pytest.approx(0.0, abs=1e-6)


def test_lipschitz_orders_models_by_amplification(model_radon):
    """The comparison the figure actually makes: a model that amplifies null
    directions must score above one that suppresses them."""
    cache = _lipschitz_cache(model_radon)
    quiet = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, 0.3), clean_cache=cache,
        radon=model_radon, n_samples=1, n_iter=10)["mean"]
    loud = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, 3.0), clean_cache=cache,
        radon=model_radon, n_samples=1, n_iter=10)["mean"]
    assert quiet < 1.0 < loud


def test_lipschitz_respects_the_sample_budget(model_radon):
    x = torch.randn(8, 1, model_radon.IMG, model_radon.IMG, dtype=model_radon.dtype)
    cache = [(x, x.clone(), model_radon.forward_la(x))]
    res = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, 1.0), clean_cache=cache,
        radon=model_radon, n_samples=3, n_iter=6)
    assert res["n"] == 3, "the budget bounds a slow power iteration"


# =========================================================================== #
# Operator construction — which backend a run uses is decided by summary.json.
# =========================================================================== #
def test_build_radon_dispatches_on_matrix_mode(monkeypatch):
    """matrix_mode picks the whole numerical backend. Getting it wrong does not
    crash; it silently reconstructs with a different operator."""
    seen = {}

    class _Marker:
        def __init__(self, **kw):
            seen.update(kw)
            seen["cls"] = self.__class__.__name__

    monkeypatch.setattr(attack, "MatrixRadonAdapter", type("M", (_Marker,), {}))
    monkeypatch.setattr(attack, "AstraRadonAdapter", type("A", (_Marker,), {}))

    attack.build_radon(_summary_dict(matrix_mode=1), device=torch.device("cpu"))
    assert seen["cls"] == "M"
    seen.clear()
    attack.build_radon(_summary_dict(matrix_mode=0), device=torch.device("cpu"))
    assert seen["cls"] == "A"


def test_build_radon_passes_the_geometry_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(attack, "MatrixRadonAdapter",
                        lambda **kw: seen.update(kw) or object())
    attack.build_radon(_summary_dict(matrix_mode=1, img_size=8, det_count=10),
                       device=torch.device("cpu"))
    assert seen["resolution"] == 8 and seen["det_count"] == 10
    assert len(seen["angles"]) == 12


# =========================================================================== #
# Attack primitives that bound what an attacker may do.
# =========================================================================== #
def test_random_start_stays_inside_the_ball():
    """PGD starts from a random point in the ball; a start outside it would mean
    the reported eps is not the budget actually spent."""
    y = torch.zeros(4, 1, 6, 6)
    eps = 0.25
    d = attack.random_start_like(y, eps, projector=lambda v: v)
    assert d.reshape(4, -1).norm(dim=1).max().item() <= eps * (1 + 1e-6)


def test_random_start_is_zero_for_a_zero_budget():
    y = torch.zeros(2, 1, 4, 4)
    d = attack.random_start_like(y, 0.0, projector=lambda v: v)
    assert torch.allclose(d, torch.zeros_like(d))


def test_random_start_applies_the_projector():
    """The start must already satisfy the range constraint, else the first step
    is spent projecting rather than attacking."""
    y = torch.ones(2, 1, 4, 4)
    d = attack.random_start_like(y, 1.0, projector=lambda v: torch.zeros_like(v))
    assert torch.allclose(d, torch.zeros_like(d))


# =========================================================================== #
# Data loading — every number in the study enters through here.
# =========================================================================== #
def _npy_dataset(tmp_path, n=6, img=5, sino=(4, 3)):
    root = tmp_path / "ds"
    for sub in ("gt", "pinv", "sino"):
        (root / sub).mkdir(parents=True)
    for i in range(n):
        np.save(root / "gt" / f"{i:05d}.npy", np.full((img, img), float(i), dtype=np.float32))
        np.save(root / "pinv" / f"{i:05d}.npy", np.full((img, img), -float(i), dtype=np.float32))
        np.save(root / "sino" / f"{i:05d}.npy", np.full(sino, float(i), dtype=np.float32))
    return root


def test_dataset_returns_aligned_triples(tmp_path):
    """gt / init / sinogram must be the *same* sample: a sort mismatch would
    silently pair every image with someone else's measurements."""
    from src.ellipse_dataloader import EllipsesGTInitDataset
    ds = EllipsesGTInitDataset(_npy_dataset(tmp_path), init="pinv")
    assert len(ds) == 6
    for i in (0, 3, 5):
        x_gt, x_init, y = ds[i]
        assert x_gt.shape == (1, 5, 5) and y.shape == (1, 4, 3)
        assert x_gt.flatten()[0].item() == pytest.approx(float(i))
        assert x_init.flatten()[0].item() == pytest.approx(-float(i))
        assert y.flatten()[0].item() == pytest.approx(float(i))


def test_dataset_selects_the_requested_init(tmp_path):
    from src.ellipse_dataloader import EllipsesGTInitDataset
    root = _npy_dataset(tmp_path)
    (root / "fbp").mkdir()
    for i in range(6):
        np.save(root / "fbp" / f"{i:05d}.npy", np.full((5, 5), 99.0, dtype=np.float32))
    assert EllipsesGTInitDataset(root, init="fbp")[0][1].flatten()[0].item() == 99.0


def test_dataloader_splits_train_and_test_disjointly(tmp_path):
    """Attacking on samples the model trained on would flatter every result."""
    from src.ellipse_dataloader import get_ellipse_dataloader
    root = _npy_dataset(tmp_path, n=6)
    common = dict(init_recon="pinv", batch_size=1, n_train=4, n_test=2,
                  shuffle=False, num_workers=0, data_root=str(root))
    ids = {}
    for split in ("train", "test"):
        loader = get_ellipse_dataloader(split=split, **common)
        ids[split] = [float(b[0].flatten()[0]) for b in loader]
    assert len(ids["train"]) == 4 and len(ids["test"]) == 2
    assert not (set(ids["train"]) & set(ids["test"])), "train and test overlap"


def test_dataloader_is_deterministic_without_shuffle(tmp_path):
    """The suite relies on every model seeing byte-identical inputs."""
    from src.ellipse_dataloader import get_ellipse_dataloader
    root = _npy_dataset(tmp_path, n=6)
    kw = dict(init_recon="pinv", batch_size=2, split="test", n_train=4, n_test=2,
              shuffle=False, num_workers=0, data_root=str(root))
    a = [float(b[0].flatten()[0]) for b in get_ellipse_dataloader(**kw)]
    b = [float(x.flatten()[0]) for x in
         (batch[0] for batch in get_ellipse_dataloader(**kw))]
    assert a == b


def test_lipschitz_empty_result_is_still_readable(model_radon):
    """An empty cache (max_samples=0, or a loader that yielded nothing) must not
    produce a dict the plotter cannot read."""
    res = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, 1.0), clean_cache=[], radon=model_radon,
        n_samples=4)
    assert res["n"] == 0
    assert math.isnan(res["mean"])
    json.dumps(res)


def test_lipschitz_plot_renders(tmp_path):
    V = _vis()
    lip = {"nsn": {"mean": 0.6, "std": 0.1, "max": 0.8, "n": 4},
           "resnet": {"mean": 2.1, "std": 0.3, "max": 2.9, "n": 4}}
    V.save_lipschitz_plot(tmp_path, lip)
    assert (tmp_path / "lipschitz_nullspace.png").exists()


def test_lipschitz_result_is_json_serialisable(model_radon):
    """It is written straight to lipschitz_nullspace.json; a non-serialisable
    field would fail only at the very end of a suite run."""
    res = attack.estimate_lipschitz_nullspace(
        model=_GainModel(model_radon, 1.0), clean_cache=_lipschitz_cache(model_radon),
        radon=model_radon, n_samples=1, n_iter=4)
    json.dumps(res)


# =========================================================================== #
# Radon operators — AstraRadonAdapter / MatrixRadonAdapter identities and the
# FBP filter construction. Tiny synthetic limited-angle operators, built on the
# spot: no data directory, summary.json or checkpoint involved.
# =========================================================================== #
def rel(a, b):
    """Relative error ||a - b|| / ||b||."""
    return float((a - b).norm() / (b.norm() + 1e-12))

# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------

def make_phantom(res, device, dtype):
    """Simple disk + rectangle phantom, shape (1, 1, res, res)."""
    lin = torch.linspace(-1, 1, res, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    img = ((xx ** 2 + yy ** 2) < 0.6 ** 2).to(dtype)
    img += 0.5 * ((xx.abs() < 0.3) & (yy.abs() < 0.2)).to(dtype)
    return img.unsqueeze(0).unsqueeze(0)

# ---------------------------------------------------------------------------
# Pytest fixtures — tiny synthetic limited-angle operators
#
# The math tests below verify the AstraRadonAdapter / MatrixRadonAdapter
# operator identities directly, with no --data-dir, summary.json or trained
# checkpoint. Building the operators needs astra (forward/back-projection) and,
# for the truncated SVD, scipy; both are gated with importorskip so the module
# skips cleanly wherever they are absent. Everything runs on CPU in float64 at a
# tiny resolution to stay fast.
# ---------------------------------------------------------------------------
_RES = 16
_DET = 24                                  # >= ceil(sqrt(2)*16), avoids clip warning
_N_ANGLES = 40
_PHI = (0.0, float(np.pi / 3.0))           # limited-angle window
_SVD_THRESH = 4e-3                          # matches create_phantom_data.py default


def _fixture_angles():
    return np.linspace(0.0, np.pi, _N_ANGLES, endpoint=False)


@pytest.fixture(scope="module")
def svd_thresh():
    return _SVD_THRESH


@pytest.fixture(scope="module")
def astra_r():
    """Astra-backed forward/back-projection operator."""
    pytest.importorskip("astra")
    from src.radon import AstraRadonAdapter
    try:
        return AstraRadonAdapter(
            resolution=_RES, angles=_fixture_angles(), det_count=_DET, phi=_PHI,
            dx=1.0, estimate_norm=True, device=torch.device("cpu"),
            dtype=torch.float64,
        )
    except Exception as exc:                       # pragma: no cover - env dependent
        pytest.skip(f"could not build AstraRadonAdapter: {exc}")


@pytest.fixture(scope="module")
def matrix_r():
    """SVD-backed limited-angle system matrices (needs astra + scipy)."""
    pytest.importorskip("astra")
    pytest.importorskip("scipy")
    from src.radon import MatrixRadonAdapter
    try:
        return MatrixRadonAdapter(
            resolution=_RES, angles=_fixture_angles(), det_count=_DET, phi=_PHI,
            svd_threshold=_SVD_THRESH, dx=1.0, estimate_norm=True,
            device=torch.device("cpu"), dtype=torch.float64, cache_dir=None,
        )
    except Exception as exc:                       # pragma: no cover - env dependent
        pytest.skip(f"could not build MatrixRadonAdapter: {exc}")


@pytest.fixture(scope="module")
def matrix_r_dense(matrix_r):
    """Dense-layout float32 twin of matrix_r (the attack.py fast path)."""
    from src.radon import MatrixRadonAdapter
    return MatrixRadonAdapter(
        resolution=_RES, angles=_fixture_angles(), det_count=_DET, phi=_PHI,
        svd_threshold=_SVD_THRESH, dx=1.0, estimate_norm=True,
        device=torch.device("cpu"), dtype=torch.float32, dense=True,
        cache_dir=None,
    )


@pytest.fixture
def x():
    """Deterministic disk+rectangle phantom, shape (1, 1, _RES, _RES)."""
    return make_phantom(_RES, torch.device("cpu"), torch.float64)


@pytest.fixture
def v():
    """Deterministic random image for null-space / decomposition tests."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 1, _RES, _RES, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Tests — Radon operator identities
# ---------------------------------------------------------------------------

def test_shapes(astra_r, matrix_r, x):
    """forward / forward_la / backward all return the expected shapes."""
    B, C, H, W = x.shape
    n_a = len(astra_r.angles)
    nd = astra_r.det_count

    y_a = astra_r.forward(x)
    y_m = matrix_r.forward(x)
    y_la = matrix_r.forward_la(x)

    assert tuple(y_a.shape) == (B, C, n_a, nd)
    assert tuple(y_m.shape) == (B, C, n_a, nd)
    assert tuple(y_la.shape) == (B, C, n_a, nd)

    assert tuple(matrix_r.backward(y_m).shape) == (B, C, H, W)
    assert tuple(matrix_r.backward_la(y_la).shape) == (B, C, H, W)


def test_forward_consistency(astra_r, matrix_r, x):
    """MatrixRadon forward matches the astra forward projection."""
    y_astra = astra_r.forward(x).to(dtype=torch.float64)
    y_matrix = matrix_r.forward(x)
    assert rel(y_matrix, y_astra) < 1e-3


def test_forward_la_rows(matrix_r, x):
    """forward_la returns exactly the measured-angle rows of forward."""
    y_full = matrix_r.forward(x)
    y_la = matrix_r.forward_la(x)
    ang_mask = (matrix_r.angles >= matrix_r.phi[0]) & (matrix_r.angles < matrix_r.phi[1])
    assert rel(y_la[:, :, ang_mask, :], y_full[:, :, ang_mask, :]) < 1e-8


def test_svd_reconstruction(matrix_r, x):
    """Truncated SVD factors reproduce the sparse system matrices (full + la)."""
    assert hasattr(matrix_r, "_U_k"), "SVD not built (svd_threshold == 0)"
    _, _, H, W = x.shape
    x_flat = x.reshape(1, H * W).to(dtype=matrix_r.dtype, device=matrix_r.device)

    # Full A
    y_sparse = torch.sparse.mm(matrix_r._A, x_flat.t()).t()
    y_svd = (matrix_r._U_k * matrix_r._s_k) @ (matrix_r._Vt_k @ x_flat.t())
    assert rel(y_svd.t(), y_sparse) < 1e-2

    # Limited-angle A_la
    y_la_sparse = torch.sparse.mm(matrix_r._A_la, x_flat.t()).t()
    y_la_svd = (matrix_r._U_k_la * matrix_r._s_k_la) @ (matrix_r._Vt_k_la @ x_flat.t())
    assert rel(y_la_svd.t(), y_la_sparse) < 1e-2


def test_dense_layout_matches_sparse(matrix_r, matrix_r_dense, x):
    """dense=True/float32 applies the same operators as sparse CSR/float64.

    Guards the attack.py fast path: the dense float32 adapter must agree with
    the legacy sparse float64 layout to float32 accuracy on every operator
    that touches A / A_la.
    """
    assert matrix_r._A.layout != torch.strided
    assert matrix_r_dense._A.layout == torch.strided

    y = matrix_r.forward(x)
    for name, arg in [("forward", x), ("forward_la", x), ("fbp", y),
                      ("fbp_la", y), ("backward", y), ("backward_la", y),
                      ("proj_ran", y), ("proj_null", x)]:
        out_ref = getattr(matrix_r, name)(arg)
        out_dense = getattr(matrix_r_dense, name)(arg.to(torch.float32))
        scale = out_ref.double().norm().clamp_min(arg.double().norm())
        err = (out_dense.double() - out_ref.double()).norm() / scale
        assert err < 1e-4, f"{name}: dense/f32 deviates from sparse/f64 by {err:.3e}"

    # power iteration runs on the dense layout too and lands on the same norm
    rel_norm = abs(matrix_r_dense.norm_A - matrix_r.norm_A) / matrix_r.norm_A
    assert rel_norm < 1e-2

    # autograd flows through the dense operator (the PGD backward path)
    xg = x.to(torch.float32).clone().requires_grad_(True)
    (g,) = torch.autograd.grad(matrix_r_dense.forward(xg).pow(2).sum(), xg)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_pseudoinverse_range_consistency(matrix_r, x, svd_thresh):
    """A A^+ A x == A x on the retained range, and backward stays finite."""
    assert hasattr(matrix_r, "_U_k"), "SVD not built"
    tol = max(1e-6, svd_thresh)

    # Full
    y = matrix_r.forward(x)
    x_back = matrix_r.backward(y)
    assert torch.isfinite(x_back).all(), "backward() produced NaN/Inf"
    assert rel(matrix_r.forward(x_back), y) < tol

    # Limited-angle
    y_la = matrix_r.forward_la(x)
    x_back_la = matrix_r.backward_la(y_la)
    assert torch.isfinite(x_back_la).all(), "backward_la() produced NaN/Inf"
    assert rel(matrix_r.forward_la(x_back_la), y_la) < tol


def test_null_space(matrix_r, v):
    """Null-space projections carry no *measured* (retained-range) signal.

    Under a truncated SVD the raw ``A_la @ proj_null_la(v)`` is not exactly zero
    — a sub-threshold trace survives — so the exact, threshold-independent
    invariant the reconstruction pipeline relies on is that the projection onto
    the *retained range* (i.e. the measured data) vanishes. This mirrors the
    proj_ran-based checks in test_attack.py.
    """
    assert hasattr(matrix_r, "_U_k"), "SVD not built"

    # null(A_la): invisible to the measured limited-angle data proj_ran(A_la ·).
    v_null_la = matrix_r.proj_null_la(v)
    meas_null = matrix_r.proj_ran(matrix_r.forward_la(v_null_la)).norm()
    meas_ref = matrix_r.proj_ran(matrix_r.forward_la(v)).norm().clamp_min(1e-12)
    assert float(meas_null / meas_ref) < 1e-8

    # null(A): invisible to the full operator's retained range (project A v onto U_k).
    def measured_full(image):
        y = torch.sparse.mm(matrix_r._A, image.reshape(1, -1).to(matrix_r.dtype).t()).t()
        return ((y @ matrix_r._U_k) @ matrix_r._U_k.t()).norm()

    v_null = matrix_r.proj_null(v)
    assert float(measured_full(v_null) / measured_full(v).clamp_min(1e-12)) < 1e-8


def test_decomposition(matrix_r, v):
    """A_la^+ A_la v + proj_null_la(v) == v (orthogonal range/null split)."""
    assert hasattr(matrix_r, "_U_k_la"), "SVD not built"
    range_comp = matrix_r.backward_la(matrix_r.forward_la(v))
    null_comp = matrix_r.proj_null_la(v)
    assert rel(range_comp + null_comp, v) < 1e-6


def test_operator_norm(astra_r, matrix_r):
    """Both adapters yield a positive, finite, mutually-consistent ‖A‖."""
    assert astra_r.norm_A is not None and astra_r.norm_A > 0
    assert matrix_r.norm_A is not None and matrix_r.norm_A > 0
    ratio = matrix_r.norm_A / astra_r.norm_A
    assert 0.8 < ratio < 1.2, f"norm_A ratio {ratio:.3f} out of range"


# ---------------------------------------------------------------------------
# FBP filtering helpers.
#
# construct_fourier_filter_torch / filter_sinogram are pure torch — no astra, no
# operator — so unlike everything above they run everywhere. They are asserted
# through their defining properties rather than against stored numbers: the ramp
# is the |omega| response built from the Ram-Lak impulse response, every named
# window attenuates it, and the sinogram filter is linear and per-row.
# ---------------------------------------------------------------------------
from src.radon import construct_fourier_filter_torch, filter_sinogram   # noqa: E402

_FILTERS = ["ramp", "ram-lak", "shepp-logan", "cosine", "hamming", "hann"]


def _filt(size=64, name="ramp", dtype=torch.float64):
    return construct_fourier_filter_torch(size, name, device=torch.device("cpu"), dtype=dtype)


@pytest.mark.parametrize("name", _FILTERS)
def test_fourier_filter_is_real_and_nonnegative(name):
    """A frequency response that only attenuates: real-valued and |omega|-like,
    so never negative — a negative band would invert that part of the spectrum."""
    f = _filt(name=name)
    assert f.shape == (64,) and not f.is_complex()
    assert torch.isfinite(f).all()
    assert (f >= -1e-12).all(), "a negative frequency response inverts that band"


@pytest.mark.parametrize("name", ["ramp", "ram-lak", "shepp-logan", "hamming", "hann"])
def test_fourier_filter_is_symmetric_about_nyquist(name):
    """f[k] == f[size-k]: the filter is zero-phase, so it may blur but must not
    shift. An asymmetric response would displace the backprojection sideways."""
    f = _filt(name=name)
    assert torch.allclose(f[1:32], f.flip(0)[:31], atol=1e-12)


def test_cosine_filter_is_asymmetric_known_defect():
    """`cosine` is the one option that fails the test above. It windows with
    fftshift(sin(linspace(0, pi, size))), a half-sine whose centre lands half a
    bin off the FFT's Nyquist, so the response is neither symmetric nor peaked
    at Nyquist (it peaks at bin 46 of 64). Nothing in the pipeline uses it —
    everything asks for 'ram-lak' — so this pins the behaviour rather than
    hiding it; fix the window before using this filter for anything.
    """
    f = _filt(name="cosine")
    assert float((f[1:32] - f.flip(0)[:31]).abs().max()) > 1e-3
    assert int(f.argmax()) != 32


@pytest.mark.parametrize("name", _FILTERS)
def test_every_filter_suppresses_dc(name):
    """The ramp is |omega|, so DC is suppressed — which is why FBP cannot recover
    the image mean from the filtered data. The discrete Ram-Lak kernel leaves a
    small residue rather than an exact zero; what matters is that it is
    negligible against the ramp's own peak, which is the scale every window
    starts from."""
    f = _filt(name=name)
    assert float(f[0]) < 0.01 * float(_filt(name="ramp").max())


def test_ramp_filter_rises_monotonically_to_nyquist():
    """The defining shape of |omega|: no dips, peak at the Nyquist bin."""
    f = _filt(name="ramp")
    assert int(f.argmax()) == 32                     # size // 2 = Nyquist
    assert (f[1:33].diff() > 0).all()


def test_ram_lak_is_an_alias_for_ramp():
    assert torch.equal(_filt(name="ramp"), _filt(name="ram-lak"))


@pytest.mark.parametrize("name", ["shepp-logan", "cosine", "hamming", "hann"])
def test_windowed_filters_attenuate_the_ramp(name):
    """Every non-ramp option is the ramp times a window in [0, 1]: it may only
    take energy out, and must take a visible amount out at Nyquist — that is the
    entire point of choosing one (noise suppression at the cost of resolution)."""
    ramp, f = _filt(name="ramp"), _filt(name=name)
    assert (f <= ramp + 1e-9).all()
    assert float(f[32]) < 0.9 * float(ramp[32])


def test_fourier_filter_rejects_odd_size_and_unknown_name():
    """Odd sizes would make the symmetric construction meaningless, and a typo in
    a filter name must not silently fall back to the ramp."""
    with pytest.raises(ValueError):
        _filt(size=63)
    with pytest.raises(ValueError):
        _filt(name="gaussian")


def test_fourier_filter_matches_across_dtypes():
    f32 = _filt(dtype=torch.float32)
    f64 = _filt(dtype=torch.float64)
    assert f32.dtype == torch.float32 and f64.dtype == torch.float64
    assert torch.allclose(f32.double(), f64, atol=1e-5)


def test_filter_sinogram_preserves_shape_and_dtype():
    for dtype in (torch.float32, torch.float64):
        Y = torch.randn(2, 1, 7, 20, dtype=dtype)
        out = filter_sinogram(Y)
        assert out.shape == Y.shape and out.dtype == dtype


def test_filter_sinogram_requires_4d():
    """Shape (B, C, angles, detectors) is load-bearing: the filter runs along the
    last axis, so a 3-D sinogram would be filtered along the wrong one."""
    with pytest.raises(ValueError):
        filter_sinogram(torch.randn(3, 8, 8))


def test_filter_sinogram_is_linear():
    """Filtering is a convolution, so superposition must hold exactly — this is
    what lets the adapters treat FBP as a linear operator when composing it with
    projections."""
    a, b = 2.5, -0.75
    Y1 = torch.randn(2, 1, 5, 16, dtype=torch.float64)
    Y2 = torch.randn(2, 1, 5, 16, dtype=torch.float64)
    lhs = filter_sinogram(a * Y1 + b * Y2)
    rhs = a * filter_sinogram(Y1) + b * filter_sinogram(Y2)
    assert rel(lhs, rhs) < 1e-10


def test_filter_sinogram_treats_rows_independently():
    """Each projection angle is filtered on its own; a leak between rows would
    mix angles before backprojection."""
    Y = torch.randn(1, 1, 4, 16, dtype=torch.float64)
    full = filter_sinogram(Y)
    one = filter_sinogram(Y[:, :, 2:3, :] * 4)      # same row, scaled, alone
    # 4 angles vs 1 changes only the pi/(2*n_angles) normalisation.
    assert rel(one / 4 * (1 / 4), full[:, :, 2:3, :]) < 1e-10


def test_filter_sinogram_scales_with_the_angle_count():
    """The pi/(2*n_angles) factor is the FBP normalisation: doubling the number
    of projections must halve each one's contribution, or the reconstruction
    scales with how finely the scan was sampled."""
    row = torch.randn(1, 1, 1, 16, dtype=torch.float64)
    one = filter_sinogram(row)
    four = filter_sinogram(row.repeat(1, 1, 4, 1))
    assert rel(four[:, :, 0:1, :] * 4, one) < 1e-10


@pytest.mark.parametrize("name", _FILTERS)
def test_filter_sinogram_cache_is_populated_and_used(name):
    """The cache is keyed by padded size / name / device / dtype. A key collision
    would silently apply the wrong filter, so check both that a hit is reused and
    that a different filter does not hit the same entry."""
    cache = {}
    Y = torch.randn(1, 1, 3, 12, dtype=torch.float64)
    first = filter_sinogram(Y, name, fourier_filter_cache=cache)
    assert len(cache) == 1
    second = filter_sinogram(Y, name, fourier_filter_cache=cache)
    assert len(cache) == 1 and torch.equal(first, second)
    assert torch.equal(first, filter_sinogram(Y, name))     # cache changes nothing
    filter_sinogram(Y, "hann", fourier_filter_cache=cache)
    assert len(cache) == (1 if name == "hann" else 2)


def test_filter_sinogram_pads_to_at_least_64():
    """Short detector rows are zero-padded to 64 before the FFT so the circular
    convolution does not wrap the row onto itself."""
    narrow = torch.zeros(1, 1, 1, 4, dtype=torch.float64)
    narrow[0, 0, 0, 0] = 1.0
    out = filter_sinogram(narrow)
    assert out.shape == narrow.shape and torch.isfinite(out).all()
    # An impulse at one end must not produce its ringing at the other end with
    # the same magnitude, which is what wrap-around would look like.
    assert abs(float(out[0, 0, 0, -1])) < abs(float(out[0, 0, 0, 0]))

# =========================================================================== #
# Numeric helpers — the image-quality metrics and small tensor utilities in
# src/utils.py, which every metric row in the suite is built from.
# =========================================================================== #
def test_rel_l2_np_zero_for_identical_and_scales_linearly():
    y = np.array([[3.0, -4.0], [0.0, 5.0]])
    assert utils.rel_l2_np(y, y) == pytest.approx(0.0)
    # ‖x - y‖ / ‖y‖ with a known perturbation: y is 3-4-5 style, ‖y‖ = sqrt(50).
    x = y.copy()
    x[0, 0] += math.sqrt(50.0)  # perturbation norm == ‖y‖  -> ratio 1.0
    assert utils.rel_l2_np(x, y) == pytest.approx(1.0)


def test_rel_l2_np_clamps_near_zero_reference():
    # All-zero reference would blow the ratio up; the 1e-3*sqrt(N) floor caps it.
    y = np.zeros((128, 128))
    x = np.full((128, 128), 0.01)
    num = np.linalg.norm(x - y)             # 0.01 * 128 = 1.28
    floor = 1e-3 * math.sqrt(y.size)        # 0.128
    assert utils.rel_l2_np(x, y) == pytest.approx(num / floor)


def test_psnr_infinite_when_identical_and_matches_formula():
    y = np.array([[0.0, 1.0], [0.5, 0.25]])
    assert utils.psnr(y, y) == float("inf")

    x = y + 0.1
    mse = float(np.mean((x - y) ** 2))
    data_range = float(y.max() - y.min())
    expected = 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
    assert utils.psnr(x, y) == pytest.approx(expected)


def test_psnr_zero_data_range_falls_back_to_unit_range():
    y = np.full((4, 4), 2.0)   # constant reference -> data_range 0 -> treated as 1
    x = y + 0.5
    mse = 0.25
    assert utils.psnr(x, y) == pytest.approx(-10.0 * math.log10(mse))


def test_ssim_perfect_match_is_one_or_nan_without_skimage():
    y = np.random.default_rng(0).random((16, 16))
    val = utils.ssim(y, y)
    if utils._HAS_SKIMAGE:
        assert val == pytest.approx(1.0, abs=1e-6)
    else:
        assert math.isnan(val)


def test_mae_and_max_abs_err():
    x = np.array([1.0, -2.0, 3.0])
    y = np.array([1.0, 0.0, 0.0])
    assert utils.mae(x, y) == pytest.approx((0 + 2 + 3) / 3)
    assert utils.max_abs_err(x, y) == pytest.approx(3.0)


def test_max_abs_err_empty_is_nan():
    assert math.isnan(utils.max_abs_err(np.array([]), np.array([])))


def test_rmse_and_nrmse():
    x = np.array([[0.0, 2.0], [0.0, 0.0]])
    y = np.zeros((2, 2))
    expected_rmse = math.sqrt(np.mean((x - y) ** 2))   # sqrt(4/4) = 1.0
    assert utils.rmse(x, y) == pytest.approx(expected_rmse)
    # nrmse divides by data_range; here reference is constant -> range floored to 1.
    assert utils.nrmse(x, y) == pytest.approx(expected_rmse)


def test_nrmse_normalises_by_data_range():
    y = np.array([[0.0, 4.0], [0.0, 0.0]])   # data_range = 4
    x = y.copy()
    x[0, 0] = 2.0                             # single error of 2 over 4 pixels
    expected = math.sqrt(4.0 / 4.0) / 4.0
    assert utils.nrmse(x, y) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# tensor helpers — src/utils.py
# --------------------------------------------------------------------------- #
def test_to_4d_promotes_2d_and_3d_and_leaves_4d():
    assert utils.to_4d(torch.zeros(8, 8)).shape == (1, 1, 8, 8)
    assert utils.to_4d(torch.zeros(3, 8, 8)).shape == (3, 1, 8, 8)
    already = torch.zeros(2, 1, 8, 8)
    assert utils.to_4d(already).shape == (2, 1, 8, 8)


def test_set_seed_makes_torch_and_numpy_reproducible():
    utils.set_seed(123)
    a_t, a_n = torch.randn(4), np.random.rand(4)
    utils.set_seed(123)
    b_t, b_n = torch.randn(4), np.random.rand(4)
    assert torch.equal(a_t, b_t)
    assert np.array_equal(a_n, b_n)


def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    utils.ensure_dir(target)
    assert target.is_dir()
    # idempotent — calling again on an existing dir must not raise.
    utils.ensure_dir(target)
