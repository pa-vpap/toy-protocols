#!/usr/bin/env python3
"""
Protocol C (Toy Model): Boltzmann-weighted selection toward "structured" point clouds
===============================================================================

This is NOT causal-set Protocol C. It's an environment-friendly toy analogue:

State space:
  - Configurations are point clouds C ∈ R^{N×D}
  - Two ensembles:
      * Entropic: isotropic Gaussian in ambient D
      * Sprinkled-like: intrinsic-d manifold projected into ambient D (+ small noise)

Energy / "action" S(C) (scale-invariant by construction):
  Choose one:
    - "knn_graph": mean kNN edge length (scale-normalized cloud)
    - "mst": mean MST edge length (approx via Prim; O(N^2), ok for N~100)
        - "pair_dist": mean random-pair squared distance (exactly constant under RMS normalization, up to sampling noise)

Boltzmann weighting:
  w(C; β) ∝ exp(-β S(C))

Diagnostics vs β:
  1) Manifoldness drift: TwoNN intrinsic dimension estimate (optionally on PCA-reduced coords)
  2) Prevalence: P(sprinkled | β) using TRUE labels (not thresholding S)
  3) Landscape / ruggedness: Metropolis acceptance rate + Var(ΔS) under local moves

Key fixes vs the earlier prototype:
  - Scale normalization per cloud (removes trivial "select smaller variance")
  - Uses true labels for prevalence
  - Fast-ish S computation (pair sampling / kNN graph)
  - Ruggedness uses ΔS / acceptance (not just Var(S under noise))
"""

from __future__ import annotations

import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, replace
from typing import Literal, Tuple, Optional, List


# ----------------------------
# Config
# ----------------------------

ActionType = Literal["knn_graph", "mst", "pair_dist"]

@dataclass
class ToyProtocolCConfig:
    num_configs: int = 4000             # total configs (half entropic, half sprinkled)
    ambient_dim: int = 50               # D
    intrinsic_dim: int = 5              # d_true for sprinkled-like
    num_points: int = 120               # N points per cloud
    sprinkled_noise: float = 0.10       # small ambient noise on sprinkled
    betas: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 10.0, 20))


    action_type: ActionType = "knn_graph"
    action_knn_k: int = 8               # for knn_graph action

    # Speed knobs
    pair_samples_for_action: int = 4000 # used if action_type == "pair_dist"
    twonn_subsample: int = 200          # number of points to use for TwoNN (<= N); set <=0 to use all
    pca_dim_for_twonn: int = 0          # 0 disables PCA, else reduce to this many dims
    twonn_trim_frac: float = 0.05       # 0 disables trimming; else trim extremes of log(mu)

    # Sampling from Ω at each beta
    num_weighted_samples: int = 800     # how many configs to sample from Boltzmann weights

    # Metropolis ruggedness probe
    metro_steps: int = 300              # per beta
    local_move_sigma: float = 0.02      # perturbation strength for local moves
    metro_measure_every: int = 1        # measure each step (keep 1)

    seed: int = 42


# ----------------------------
# Ensemble generators
# ----------------------------

def generate_entropic_cloud(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return rng.standard_normal((n, d))

def generate_sprinkled_cloud(
    rng: np.random.Generator,
    n: int,
    ambient_d: int,
    intrinsic_d: int,
    noise: float
) -> np.ndarray:
    # Sample in intrinsic space
    low = rng.standard_normal((n, intrinsic_d))
    # Random linear embedding into ambient space
    proj = rng.standard_normal((ambient_d, intrinsic_d))
    embedded = low @ proj.T
    return embedded + noise * rng.standard_normal((n, ambient_d))


# ----------------------------
# Preprocessing: scale normalization (critical!)
# ----------------------------

def normalize_cloud(C: np.ndarray) -> np.ndarray:
    """
    Remove translation and normalize RMS radius to 1.

    This kills the trivial effect where exp(-β S) just picks smaller-variance clouds.
    """
    # Be defensive: if clouds are stored in a dtype=object container (as we do for Ω),
    # individual clouds can arrive as object-typed arrays, which will break linear algebra.
    try:
        C = np.asarray(C, dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        raise TypeError(
            "normalize_cloud expected a numeric (N,D) array-like cloud; "
            f"got type={type(C)!r}."
        ) from e
    if C.ndim != 2:
        raise ValueError(f"normalize_cloud expected a 2D array (N,D); got shape={C.shape!r}")

    X = C - C.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(X * X, axis=1)))
    if rms < 1e-12:
        return X
    return X / rms


# ----------------------------
# Utilities: distances without full NxN when possible
# ----------------------------

def sample_pairwise_distances(
    rng: np.random.Generator, X: np.ndarray, n_pairs: int
) -> np.ndarray:
    n = X.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    # ensure i != j for most pairs
    mask = (i == j)
    if np.any(mask):
        j[mask] = (j[mask] + 1) % n
    diff = X[i] - X[j]
    # Return squared distances.
    # For centered data, E||xi-xj||^2 = 2 E||x||^2, so after RMS-radius normalization this
    # is ~2 for *any* cloud geometry (good negative-control action).
    return np.sum(diff * diff, axis=1)

def full_distance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))



# ----------------------------
# Action definitions (scale-invariant)
# ----------------------------

def action_pair_dist(
    rng: np.random.Generator, X: np.ndarray, n_pairs: int
) -> float:
    # After normalize_cloud (RMS radius = 1), mean squared pair distance is ~2 regardless of
    # intrinsic structure. This is a good "wrong action" control.
    d2 = sample_pairwise_distances(rng, X, n_pairs)
    return float(np.mean(d2))

def action_knn_graph(
    X: np.ndarray, k: int
) -> float:
    """
    Mean kNN edge length (excluding self).
    Uses full NxN distances for simplicity; for N~120 it's fine.
    """
    D = full_distance_matrix(X)
    np.fill_diagonal(D, np.inf)
    # k nearest distances per point
    knn = np.partition(D, kth=k-1, axis=1)[:, :k]
    return float(np.mean(knn))

def action_mst_mean_edge(
    X: np.ndarray
) -> float:
    """
    Mean edge length of MST using dense Prim (O(N^2)).
    For N ~ 100-200, fine. Uses full distances.
    """
    D = full_distance_matrix(X)
    n = D.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    total = 0.0
    edges = 0

    for _ in range(n):
        u = int(np.argmin(np.where(in_mst, np.inf, min_edge)))
        in_mst[u] = True
        if min_edge[u] != 0.0 and np.isfinite(min_edge[u]):
            total += float(min_edge[u])
            edges += 1
        # relax
        du = D[u]
        min_edge = np.minimum(min_edge, du)

    if edges == 0:
        return 0.0
    return total / edges


def compute_action(
    rng: np.random.Generator,
    cloud: np.ndarray,
    action_type: ActionType,
    knn_k: int,
    pair_samples: int
) -> float:
    X = normalize_cloud(cloud)
    if action_type == "pair_dist":
        return action_pair_dist(rng, X, pair_samples)
    if action_type == "knn_graph":
        return action_knn_graph(X, knn_k)
    if action_type == "mst":
        return action_mst_mean_edge(X)
    raise ValueError(f"Unknown action_type: {action_type}")


# ----------------------------
# TwoNN intrinsic dimension estimator (robust-ish)
# ----------------------------

def pca_reduce(X: np.ndarray, out_dim: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"pca_reduce expected a 2D array (N,D); got shape={X.shape!r}")
    if out_dim <= 0 or out_dim >= X.shape[1]:
        return X
    # SVD on centered data
    Y = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    return Y @ Vt[:out_dim].T

def estimate_intrinsic_dim_twonn(
    rng: np.random.Generator,
    cloud: np.ndarray,
    subsample: int = 0,
    pca_dim: int = 0,
    trim_frac: float = 0.05
) -> float:
    """
    TwoNN estimator:
      d ≈ 1 / mean(log(mu))   (one common variant)
    Your previous variant used -2/mean(log(mu)). Variants exist; scaling differs.

    We:
      - normalize cloud
      - optionally subsample points
      - optionally PCA-reduce
      - compute full NxN distances for the subsample
      - trim extremes of log(mu) for stability
    """
    X = normalize_cloud(cloud)
    n = X.shape[0]
    if subsample and 0 < subsample < n:
        idx = rng.choice(n, size=subsample, replace=False)
        X = X[idx]

    X = pca_reduce(X, pca_dim)

    D = full_distance_matrix(X)
    np.fill_diagonal(D, np.inf)
    nn = np.partition(D, kth=1, axis=1)[:, :2]  # two nearest distances
    d1 = nn[:, 0]
    d2 = nn[:, 1]
    # avoid zeros / degeneracy
    d1 = np.maximum(d1, 1e-12)
    mu = d2 / d1
    logmu = np.log(np.maximum(mu, 1.0 + 1e-12))

    # trim outliers
    if 0.0 < trim_frac < 0.5:
        lo = np.quantile(logmu, trim_frac)
        hi = np.quantile(logmu, 1.0 - trim_frac)
        logmu = logmu[(logmu >= lo) & (logmu <= hi)]
        if logmu.size == 0:
            return np.nan

    m = float(np.mean(logmu))
    if m <= 1e-12:
        return np.nan
    return float(1.0 / m)


# ----------------------------
# Boltzmann sampling over a fixed Ω
# ----------------------------

def softmax_logweights(logw: np.ndarray) -> np.ndarray:
    a = logw - np.max(logw)
    w = np.exp(a)
    s = np.sum(w)
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


# ----------------------------
# Metropolis ruggedness probe (local move on clouds)
# ----------------------------

def metropolis_ruggedness_probe(
    rng: np.random.Generator,
    clouds: np.ndarray,
    actions: np.ndarray,
    beta: float,
    action_type: ActionType,
    knn_k: int,
    pair_samples: int,
    steps: int,
    sigma: float
) -> Tuple[float, float]:
    """
    Pick a random cloud from Ω and do local Gaussian perturbations:
      C' = C + sigma * N(0,1)
    Accept with Metropolis prob exp(-β (S' - S)).

    Returns:
      (accept_rate, var_deltaS)
    """
    idx = int(rng.integers(0, len(clouds)))
    C = clouds[idx].copy()
    S = float(actions[idx])

    accepted = 0
    deltaS_list: List[float] = []

    for _ in range(steps):
        Cp = C + sigma * rng.standard_normal(C.shape)
        Sp = compute_action(rng, Cp, action_type, knn_k, pair_samples)
        dS = Sp - S
        deltaS_list.append(dS)

        if dS <= 0 or rng.random() < np.exp(-beta * dS):
            C = Cp
            S = Sp
            accepted += 1

    return accepted / max(steps, 1), float(np.var(deltaS_list)) if deltaS_list else np.nan


# ----------------------------
# Main experiment
# ----------------------------

def run_toy_protocol_c(
    cfg: ToyProtocolCConfig,
    *,
    do_metropolis: bool = True,
    report_twonn_baseline_by_label: bool = False,
    twonn_baseline_samples_per_label: int = 100,
):
    rng = np.random.default_rng(cfg.seed)

    n_ent = cfg.num_configs // 2
    n_spr = cfg.num_configs - n_ent

    # Build Ω and labels
    clouds = []
    labels = []  # 0 entropic, 1 sprinkled
    for _ in range(n_ent):
        clouds.append(generate_entropic_cloud(rng, cfg.num_points, cfg.ambient_dim))
        labels.append(0)
    for _ in range(n_spr):
        clouds.append(generate_sprinkled_cloud(rng, cfg.num_points, cfg.ambient_dim, cfg.intrinsic_dim, cfg.sprinkled_noise))
        labels.append(1)

    # Shapes are fixed here, so prefer a dense numeric array for speed + safety.
    # Result shape: (M, N, D)
    clouds = np.stack(clouds, axis=0).astype(np.float64, copy=False)
    labels = np.array(labels, dtype=int)

    # Precompute actions S(C) on Ω (most expensive part)
    S_vals = np.empty(cfg.num_configs, dtype=float)
    for i in range(cfg.num_configs):
        S_vals[i] = compute_action(
            rng, clouds[i], cfg.action_type, cfg.action_knn_k, cfg.pair_samples_for_action
        )

    # Baseline: compare S distributions by label
    S_ent = S_vals[labels == 0]
    S_spr = S_vals[labels == 1]

    # Informativeness check: correlation between action and true labels.
    # labels are {0=entropic, 1=sprinkled}. A strongly *negative* correlation means
    # sprinkled configs tend to have lower S (desired for Protocol C mechanism).
    if np.std(S_vals) < 1e-15 or np.std(labels) < 1e-15:
        corr_S_label = np.nan
    else:
        corr_S_label = float(np.corrcoef(S_vals, labels)[0, 1])

    baseline = {
        "S_ent_mean": float(np.mean(S_ent)),
        "S_spr_mean": float(np.mean(S_spr)),
        "S_ent_std": float(np.std(S_ent)),
        "S_spr_std": float(np.std(S_spr)),
        "corr_S_label": corr_S_label,
    }

    twonn_baseline = None
    if report_twonn_baseline_by_label:
        # Use a dedicated RNG so baseline reporting doesn't perturb the main run's RNG stream.
        rng_twonn_base = np.random.default_rng(cfg.seed + 12345)

        def _twonn_summary_for_indices(idxs: np.ndarray) -> Tuple[float, float, int]:
            if idxs.size == 0:
                return (np.nan, np.nan, 0)
            k = int(min(max(twonn_baseline_samples_per_label, 0), idxs.size))
            if k <= 0:
                return (np.nan, np.nan, 0)
            chosen = rng_twonn_base.choice(idxs, size=k, replace=False)
            vals: List[float] = []
            for j in chosen:
                d_hat = estimate_intrinsic_dim_twonn(
                    rng_twonn_base,
                    clouds[int(j)],
                    subsample=cfg.twonn_subsample,
                    pca_dim=cfg.pca_dim_for_twonn,
                    trim_frac=cfg.twonn_trim_frac,
                )
                if np.isfinite(d_hat):
                    vals.append(float(d_hat))
            if not vals:
                return (np.nan, np.nan, 0)
            arr = np.array(vals, dtype=np.float64)
            return (float(np.mean(arr)), float(np.std(arr)), int(arr.size))

        ent_idxs = np.where(labels == 0)[0]
        spr_idxs = np.where(labels == 1)[0]
        ent_mean, ent_std, ent_n = _twonn_summary_for_indices(ent_idxs)
        spr_mean, spr_std, spr_n = _twonn_summary_for_indices(spr_idxs)
        twonn_baseline = {
            "id_ent_mean": ent_mean,
            "id_ent_std": ent_std,
            "id_ent_n": ent_n,
            "id_spr_mean": spr_mean,
            "id_spr_std": spr_std,
            "id_spr_n": spr_n,
            "samples_per_label": int(twonn_baseline_samples_per_label),
            "twonn_subsample": int(cfg.twonn_subsample),
            "pca_dim": int(cfg.pca_dim_for_twonn),
            "trim_frac": float(cfg.twonn_trim_frac),
        }

    # Diagnostics vs β
    mean_id = []
    std_id = []
    p_sprinkled = []
    mean_S = []
    std_S = []
    accept_rate = []
    var_deltaS = []

    for beta in cfg.betas:
        logw = -beta * S_vals
        w = softmax_logweights(logw)

        # Sample indices from Ω by Boltzmann weights
        idxs = rng.choice(cfg.num_configs, size=cfg.num_weighted_samples, replace=True, p=w)
        sampled_clouds = clouds[idxs]
        sampled_S = S_vals[idxs]
        sampled_labels = labels[idxs]

        mean_S.append(float(np.mean(sampled_S)))
        std_S.append(float(np.std(sampled_S)))
        p_sprinkled.append(float(np.mean(sampled_labels == 1)))

        # TwoNN intrinsic dim (subsample + optional PCA)
        ids = []
        for C in sampled_clouds:
            d_hat = estimate_intrinsic_dim_twonn(
                rng,
                C,
                subsample=cfg.twonn_subsample,
                pca_dim=cfg.pca_dim_for_twonn,
                trim_frac=cfg.twonn_trim_frac,
            )
            if np.isfinite(d_hat):
                ids.append(d_hat)
        if ids:
            mean_id.append(float(np.mean(ids)))
            std_id.append(float(np.std(ids)))
        else:
            mean_id.append(np.nan)
            std_id.append(np.nan)

        if do_metropolis and cfg.metro_steps > 0:
            # Ruggedness probe by Metropolis local moves
            acc, vds = metropolis_ruggedness_probe(
                rng,
                clouds=clouds,
                actions=S_vals,
                beta=float(beta),
                action_type=cfg.action_type,
                knn_k=cfg.action_knn_k,
                pair_samples=cfg.pair_samples_for_action,
                steps=cfg.metro_steps,
                sigma=cfg.local_move_sigma
            )
            accept_rate.append(acc)
            var_deltaS.append(vds)
        else:
            accept_rate.append(np.nan)
            var_deltaS.append(np.nan)

    results = {
        "baseline": baseline,
        "twonn_baseline": twonn_baseline,
        "betas": cfg.betas,
        "mean_id": np.array(mean_id),
        "std_id": np.array(std_id),
        # TwoNN prefactor ambiguity: some conventions use d_hat = 2/<log(mu)> instead of 1/<log(mu)>.
        # This rescales estimates by 2× (including the spread) without changing the trend vs beta.
        "mean_id_x2": 2.0 * np.array(mean_id),
        "std_id_x2": 2.0 * np.array(std_id),
        "p_sprinkled": np.array(p_sprinkled),
        "mean_S": np.array(mean_S),
        "std_S": np.array(std_S),
        "accept_rate": np.array(accept_rate),
        "var_deltaS": np.array(var_deltaS),
    }
    return results


def plot_results(cfg: ToyProtocolCConfig, results):
    betas = results["betas"]

    plt.figure(figsize=(16, 10))

    # 1) Manifoldness drift (TwoNN)
    ax1 = plt.subplot(2, 2, 1)
    ax1.errorbar(betas, results["mean_id"], yerr=results["std_id"], marker="o")
    ax1.axhline(cfg.intrinsic_dim, linestyle="--", label=f"true intrinsic dim={cfg.intrinsic_dim}")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel("Estimated intrinsic dimension (TwoNN)")
    ax1.set_title("Manifoldness drift")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Prevalence of sprinkled configs
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(betas, results["p_sprinkled"], marker="o")
    ax2.axhline(0.5, linestyle="--", alpha=0.6, label="Ω baseline (50%)")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("P(sprinkled | β)")
    ax2.set_title("Prevalence shift (true labels)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Mean action under weighting
    ax3 = plt.subplot(2, 2, 3)
    ax3.errorbar(betas, results["mean_S"], yerr=results["std_S"], marker="s")
    ax3.set_xlabel(r"$\beta$")
    ax3.set_ylabel("⟨S⟩ under Boltzmann sampling")
    ax3.set_title(f"Action vs β   (action={cfg.action_type})")
    ax3.grid(True, alpha=0.3)

    # 4) Ruggedness / landscape probe
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(betas, results["accept_rate"], marker="o", label="Metropolis acceptance rate")
    ax4b = ax4.twinx()
    ax4b.plot(betas, results["var_deltaS"], marker="s", linestyle="--", label="Var(ΔS) under local moves")
    ax4.set_xlabel(r"$\beta$")
    ax4.set_ylabel("Acceptance rate")
    ax4b.set_ylabel("Var(ΔS)")

    ax4.set_title("Landscape probe")
    ax4.grid(True, alpha=0.3)

    # joint legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    backend = matplotlib.get_backend()
    if "agg" in backend.lower():
        out_path = "protocol_c_results.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path} (matplotlib backend: {backend})")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Toy Protocol C: Boltzmann-weighted selection toward structured point clouds"
    )
    parser.add_argument(
        "--action-type",
        choices=["knn_graph", "mst", "pair_dist"],
        default=None,
        help="Which action S(C) to use. Use 'pair_dist' as a negative control (after normalization it is ~constant).",
    )
    parser.add_argument(
        "--sanity-twonn",
        action="store_true",
        help="Run a quick TwoNN honesty check: disables PCA (pca_dim=0), disables trimming (trim=0), "
             "skips Metropolis probe, and suppresses plotting.",
    )
    parser.add_argument(
        "--twonn-baseline-n",
        type=int,
        default=100,
        help="Number of Ω configs per label for the unweighted TwoNN baseline summary (used in --sanity-twonn).",
    )
    args = parser.parse_args()

    cfg = ToyProtocolCConfig(
        num_configs=4000,
        ambient_dim=50,
        intrinsic_dim=5,
        num_points=120,
        sprinkled_noise=0.10,
        betas=np.linspace(0.1, 10.0, 20),

        # Try "knn_graph" first; it's meaningfully geometry-sensitive after normalization.
        action_type="knn_graph",
        action_knn_k=8,

        # TwoNN settings
        twonn_subsample=120,       # use all points (N=120)
        pca_dim_for_twonn=15,      # helps TwoNN in high ambient dim; set 0 to disable

        # Sampling
        num_weighted_samples=800,

        # Metropolis probe
        metro_steps=300,
        local_move_sigma=0.02,

        seed=42,
    )

    if args.action_type is not None:
        cfg = replace(cfg, action_type=args.action_type)

    do_plot = True
    do_metropolis = True
    report_twonn_baseline_by_label = False
    if args.sanity_twonn:
        # "Honesty" run: remove PCA + trimming biases; keep everything else identical.
        # Use `replace` to keep the config immutable-ish and make runs easier to reason about.
        cfg = replace(cfg, pca_dim_for_twonn=0, twonn_trim_frac=0.0, metro_steps=0)
        do_metropolis = False
        do_plot = False
        report_twonn_baseline_by_label = True

    results = run_toy_protocol_c(
        cfg,
        do_metropolis=do_metropolis,
        report_twonn_baseline_by_label=report_twonn_baseline_by_label,
        twonn_baseline_samples_per_label=(args.twonn_baseline_n if args.sanity_twonn else 0),
    )

    b = results["baseline"]
    print("\n=== Toy Protocol C Baseline (Ω) ===")
    print(f"Action type: {cfg.action_type}")
    print(f"S(entropic)   = {b['S_ent_mean']:.4f} ± {b['S_ent_std']:.4f}")
    print(f"S(sprinkled)  = {b['S_spr_mean']:.4f} ± {b['S_spr_std']:.4f}")
    print(f"Corr(S, label) = {b['corr_S_label']:.4f}   (label: 0=entropic, 1=sprinkled)")
    print("Expectation for a 'good' action: S(sprinkled) < S(entropic) on average.\n")

    # Copy/paste-friendly summaries (for quick inspection / sharing)
    print("p_sprinkled =", np.array2string(results["p_sprinkled"], precision=4, separator=", "))
    print("mean_id     =", np.array2string(results["mean_id"], precision=4, separator=", "))
    print("mean_id_x2  =", np.array2string(results["mean_id_x2"], precision=4, separator=", "))
    print("mean_S      =", np.array2string(results["mean_S"], precision=4, separator=", "))

    if args.sanity_twonn and results.get("twonn_baseline"):
        tb = results["twonn_baseline"]
        print(
            "TwoNN baseline (Ω, unweighted; mean±std; same TwoNN settings as run): "
            f"entropic={tb['id_ent_mean']:.4f}±{tb['id_ent_std']:.4f} (std, n={tb['id_ent_n']}), "
            f"sprinkled={tb['id_spr_mean']:.4f}±{tb['id_spr_std']:.4f} (std, n={tb['id_spr_n']}); "
            f"settings(subsample={tb['twonn_subsample']}, pca_dim={tb['pca_dim']}, trim={tb['trim_frac']})"
        )

    if do_plot:
        plot_results(cfg, results)


if __name__ == "__main__":
    main()
