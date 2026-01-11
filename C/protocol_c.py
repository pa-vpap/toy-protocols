# -*- coding: utf-8 -*-
"""
Protocol C — Suppression of non-manifold interval statistics (on-the-fly)

Runs the SAME coordinate MCMC as Protocol A, but during sampling it measures
(open) interval-size statistics k = |(y,x)| for sampled comparable pairs y≺x,
and compares to reference ensembles:

  (1) 4D Minkowski sprinklings in the same Alexandrov interval (same N, same T)
  (2) Optional non-manifold baseline (KR-like 3-layer random posets)

Core outputs per (N,beta):
  - JS divergence to 4D reference: JS(P_sel, P_4D)
  - Optional JS divergence to KR reference: JS(P_sel, P_KR)
  - Small-interval mass p_small(K)=Σ_{k<=K} P(k), for K in {3,5,10,20}
  - Acceptance + ESS (ESS computed on JS time series per chain)

Plots:
  - JS vs beta (per N)
  - p_small(20) vs beta (per N) with 4D reference band
  - Interval spectrum overlays for a few betas (per N)

Conventions match Protocol A:
  - Open interval size: k(y,x) = popcount(reach[y] & past[x])
  - Sprinkling in Alexandrov interval (causal diamond) height T
  - Causality from Minkowski metric
  - BD-style action from interval abundances N_k (k=open interval size)

Usage example:
  python protocol_c.py --kmax 3 --alpha "1,-9,16,-8" \
    --betas 0.001,0.003,0.0042,0.006,0.01 --Ns 120,180,250,350 --chains 5 \
    --mix_local 0.5 --ref_sets 80 --ref_pairs 20000 --sample_pairs 20000

Tip:
  - Keep k_cap moderate (e.g. 200) and use tail-binning for stable JS.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------
# Causal set bitset representation
# -----------------------------
@dataclass
class CausalSet:
    N: int
    reach: List[int]  # reach[i] bitset of all j with i ≺ j
    past: List[int]   # past[j] bitset of all i with i ≺ j


def build_past_from_reach(N: int, reach: List[int]) -> List[int]:
    past = [0] * N
    for i in range(N):
        x = reach[i]
        while x:
            lsb = x & -x
            j = lsb.bit_length() - 1
            past[j] |= (1 << i)
            x ^= lsb
    return past


# -----------------------------
# Alexandrov interval volume + density scaling (same as Protocol A)
# -----------------------------
def sphere_area(n: int) -> float:
    return 2.0 * math.pi ** ((n + 1) / 2.0) / math.gamma((n + 1) / 2.0)


def diamond_volume(dim: int, T: float) -> float:
    Sd_2 = sphere_area(dim - 2)
    return 2.0 * Sd_2 / (dim * (dim - 1)) * (0.5 * T) ** dim


def T_for_fixed_density(dim: int, N: int, rho: float) -> float:
    Sd_2 = sphere_area(dim - 2)
    return 2.0 * ((N * dim * (dim - 1)) / (2.0 * rho * Sd_2)) ** (1.0 / dim)


# -----------------------------
# Alexandrov interval sprinkling (same as Protocol A)
# -----------------------------
def inside_alexandrov(pt: np.ndarray, dim: int, T: float) -> bool:
    half = 0.5 * T
    t = float(pt[0])
    r2 = float(np.sum(pt[1:] ** 2))
    return ((t + half) ** 2 > r2) and ((half - t) ** 2 > r2) and (-half < t < half)


def sample_point_in_diamond(dim: int, T: float, rng: np.random.Generator) -> np.ndarray:
    half = 0.5 * T
    while True:
        t = rng.uniform(-half, half)
        x = rng.uniform(-half, half, size=(dim - 1,))
        pt = np.concatenate([[t], x])
        if inside_alexandrov(pt, dim, T):
            return pt


def sprinkle_diamond(N: int, dim: int, T: float, rng: np.random.Generator) -> np.ndarray:
    pts = np.zeros((N, dim), dtype=float)
    for i in range(N):
        pts[i] = sample_point_in_diamond(dim, T, rng)
    pts = pts[np.argsort(pts[:, 0])]
    return pts


def causalset_from_coords(coords: np.ndarray) -> CausalSet:
    N, dim = coords.shape
    t = coords[:, 0]
    x = coords[:, 1:]

    reach = [0] * N
    for i in range(N - 1):
        dt = t[i + 1:] - t[i]
        dx = x[i + 1:] - x[i]
        tau2 = dt * dt - np.sum(dx * dx, axis=1)
        js = np.where(tau2 > 0.0)[0] + (i + 1)
        bits = 0
        for j in js:
            bits |= (1 << int(j))
        reach[i] = bits

    past = build_past_from_reach(N, reach)
    return CausalSet(N=N, reach=reach, past=past)


# -----------------------------
# BD-style action via interval abundances (same as Protocol A)
# -----------------------------
def bd_action_interval_abundances(C: CausalSet, alpha: np.ndarray, k_max: int) -> float:
    N = C.N
    reach, past = C.reach, C.past

    Nk = [0] * (k_max + 1)
    for x in range(N):
        ybits = past[x]
        while ybits:
            lsb = ybits & -ybits
            y = lsb.bit_length() - 1
            k = (reach[y] & past[x]).bit_count()
            if k <= k_max:
                Nk[k] += 1
            ybits ^= lsb

    return float(np.dot(alpha[: k_max + 1], np.array(Nk, dtype=float)))


# -----------------------------
# Mixed coordinate proposal (same as Protocol A)
# -----------------------------
def propose_one_point(
    coords: np.ndarray,
    dim: int,
    T: float,
    rng: np.random.Generator,
    move: str,
    local_sigma: float,
) -> np.ndarray:
    N = coords.shape[0]
    idx = int(rng.integers(0, N))
    new_coords = coords.copy()

    if move == "resample":
        new_coords[idx] = sample_point_in_diamond(dim, T, rng)
    elif move == "local":
        pt = new_coords[idx].copy()
        for _ in range(200):
            cand = pt + rng.normal(0.0, local_sigma, size=(dim,))
            if inside_alexandrov(cand, dim, T):
                new_coords[idx] = cand
                break
        else:
            new_coords[idx] = sample_point_in_diamond(dim, T, rng)
    else:
        raise ValueError("move must be 'resample' or 'local'")

    new_coords = new_coords[np.argsort(new_coords[:, 0])]
    return new_coords


# -----------------------------
# ESS (same as Protocol A; for JS series)
# -----------------------------
def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.ones(n)
    x = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=m)[:n]
    ac /= ac[0] if ac[0] != 0 else 1.0
    return ac


def ess_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float(n)
    rho = _autocorr_fft(x)
    s = 0.0
    for t in range(1, n):
        if rho[t] <= 0.0:
            break
        s += rho[t]
    tau = 1.0 + 2.0 * s
    if tau <= 1e-12:
        return float(n)
    return float(n / tau)


# -----------------------------
# Protocol C: interval histogram sampling
# -----------------------------
def _random_set_bit_index(bits: int, r: int) -> int:
    """Return index of r-th set bit in bits (0-based), assuming 0 <= r < popcount(bits)."""
    b = bits
    while True:
        lsb = b & -b
        if r == 0:
            return lsb.bit_length() - 1
        b ^= lsb
        r -= 1


def sample_interval_histogram(
    C: CausalSet,
    rng: np.random.Generator,
    n_pairs: int,
    k_cap: int,
) -> np.ndarray:
    """
    Sample n_pairs comparable pairs (y ≺ x) and histogram their open interval sizes k.
    Returns hist[0..k_cap], with overflow accumulated in hist[k_cap].
    """
    N = C.N
    reach, past = C.reach, C.past
    hist = np.zeros(k_cap + 1, dtype=np.int64)

    xs = [x for x in range(N) if past[x] != 0]
    if not xs:
        return hist

    for _ in range(n_pairs):
        x = xs[int(rng.integers(0, len(xs)))]
        ybits = past[x]
        m = ybits.bit_count()
        if m == 0:
            continue
        r = int(rng.integers(0, m))
        y = _random_set_bit_index(ybits, r)

        k = (reach[y] & past[x]).bit_count()
        if k > k_cap:
            k = k_cap
        hist[k] += 1

    return hist


def default_rebin_edges(N: int, k_cap: int) -> List[Tuple[int, int]]:
    """
    Tail-stable binning:
      - exact bins 0..20
      - then coarse bins increasing roughly logarithmically up to min(N, k_cap)
    """
    hi = min(N, k_cap)
    edges: List[Tuple[int, int]] = [(k, k) for k in range(0, min(21, hi + 1))]
    if hi <= 20:
        return edges

    tail = [(21, 30), (31, 45), (46, 70), (71, 110), (111, 170), (171, 260), (261, hi)]
    for a, b in tail:
        if a > hi:
            break
        edges.append((a, min(b, hi)))
    # Ensure last bin ends exactly at hi
    if edges[-1][1] != hi:
        edges[-1] = (edges[-1][0], hi)
    return edges


def rebin_hist(hist: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    out = np.zeros(len(edges), dtype=float)
    for i, (a, b) in enumerate(edges):
        a = max(a, 0)
        b = min(b, len(hist) - 1)
        if a <= b:
            out[i] = float(hist[a : b + 1].sum())
    return out


def normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = float(p.sum())
    if s <= eps:
        return np.ones_like(p) / max(1, len(p))
    return p / s


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = normalize(p, eps=eps)
    q = normalize(q, eps=eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def p_small_from_hist(hist: np.ndarray, K: int) -> float:
    K = int(K)
    K = max(0, min(K, len(hist) - 1))
    return float(hist[: K + 1].sum() / max(1.0, float(hist.sum())))


# -----------------------------
# Optional KR-like baseline generator
# -----------------------------
def kr_like_causalset(N: int, rng: np.random.Generator, p: float = 0.5) -> CausalSet:
    """
    Simple KR-like 3-layer random poset:
      - partition into L0 < L1 < L2 (sizes ~ N/4, N/2, N/4)
      - for each pair (i in L0, j in L1) add i≺j with prob p
      - for each pair (i in L1, j in L2) add i≺j with prob p
      - take transitive closure approximately by adding implied L0->L2 edges
        whenever i->k and k->j exist (computed via bitset multiplication)
    """
    n0 = N // 4
    n1 = N // 2
    n2 = N - n0 - n1

    L0 = list(range(0, n0))
    L1 = list(range(n0, n0 + n1))
    L2 = list(range(n0 + n1, N))

    reach = [0] * N

    # L0 -> L1
    for i in L0:
        bits = 0
        for j in L1:
            if rng.random() < p:
                bits |= (1 << j)
        reach[i] = bits

    # L1 -> L2
    for i in L1:
        bits = 0
        for j in L2:
            if rng.random() < p:
                bits |= (1 << j)
        reach[i] = bits

    # L0 -> L2 via closure: reach[i] gets union of reach[k] over k in (L0->L1 successors)
    # This is a fast bitset closure step for this layered structure.
    for i in L0:
        succ_L1 = reach[i]
        # iterate set bits in succ_L1; union their reach (which only targets L2)
        b = succ_L1
        implied = 0
        while b:
            lsb = b & -b
            k = lsb.bit_length() - 1
            implied |= reach[k]
            b ^= lsb
        reach[i] |= implied

    past = build_past_from_reach(N, reach)
    return CausalSet(N=N, reach=reach, past=past)


# -----------------------------
# Reference ensembles (4D and optional KR)
# -----------------------------
def reference_interval_distribution_4d(
    N: int,
    dim: int,
    T: float,
    rng: np.random.Generator,
    n_sets: int,
    n_pairs: int,
    k_cap: int,
    edges: List[Tuple[int, int]],
    progress_label: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[int, Tuple[float, float]]]:
    """
    Returns:
      - P4D (rebinned normalized distribution)
      - small-mass bands: dict K -> (mean, std) across sets (computed on UN-rebinned hist)
    """
    Ps = []
    smalls: Dict[int, List[float]] = {3: [], 5: [], 10: [], 20: []}

    it_range = range(n_sets)
    if tqdm is not None and progress_label is not None:
        it_range = tqdm(it_range, desc=progress_label, leave=False)

    for _ in it_range:
        coords = sprinkle_diamond(N, dim, T, rng)
        C = causalset_from_coords(coords)
        hist = sample_interval_histogram(C, rng, n_pairs=n_pairs, k_cap=k_cap)
        for K in smalls.keys():
            smalls[K].append(p_small_from_hist(hist, K))
        pr = rebin_hist(hist, edges)
        Ps.append(normalize(pr))

    P_mean = normalize(np.mean(np.stack(Ps, axis=0), axis=0))
    bands = {K: (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)) for K, v in smalls.items()}
    return P_mean, bands


def reference_interval_distribution_kr(
    N: int,
    rng: np.random.Generator,
    n_sets: int,
    n_pairs: int,
    k_cap: int,
    edges: List[Tuple[int, int]],
    p: float = 0.5,
    progress_label: Optional[str] = None,
) -> np.ndarray:
    Ps = []
    it_range = range(n_sets)
    if tqdm is not None and progress_label is not None:
        it_range = tqdm(it_range, desc=progress_label, leave=False)

    for _ in it_range:
        C = kr_like_causalset(N, rng, p=p)
        hist = sample_interval_histogram(C, rng, n_pairs=n_pairs, k_cap=k_cap)
        pr = rebin_hist(hist, edges)
        Ps.append(normalize(pr))
    return normalize(np.mean(np.stack(Ps, axis=0), axis=0))


# -----------------------------
# MCMC chain for Protocol C (collects histograms + JS series)
# -----------------------------
@dataclass
class ChainCOut:
    js_to_4d: np.ndarray
    js_to_kr: Optional[np.ndarray]
    psmall20: np.ndarray
    accept_rate: float
    # for overlay plots / debugging
    mean_hist_rebinned: np.ndarray


def run_protocol_c_chain(
    N: int,
    dim: int,
    T: float,
    beta: float,
    alpha: np.ndarray,
    k_max: int,
    burn_in: int,
    steps: int,
    thin: int,
    mix_local: float,
    local_sigma: float,
    rng: np.random.Generator,
    sample_pairs: int,
    k_cap: int,
    edges: List[Tuple[int, int]],
    P4D: np.ndarray,
    PKR: Optional[np.ndarray],
    show_progress: bool = False,
) -> ChainCOut:
    coords = sprinkle_diamond(N, dim, T, rng)
    C = causalset_from_coords(coords)
    S = bd_action_interval_abundances(C, alpha, k_max)

    accepts = 0
    props = 0

    js4_list: List[float] = []
    jskr_list: List[float] = []
    p20_list: List[float] = []
    hsum = np.zeros(len(edges), dtype=float)

    total_iters = burn_in + steps
    it_range = range(total_iters)
    if show_progress and tqdm is not None:
        it_range = tqdm(it_range, desc=f"C-Chain N={N} β={beta:.4f}", leave=False)

    for it in it_range:
        move = "local" if (mix_local > 0 and rng.random() < mix_local) else "resample"
        coords_prop = propose_one_point(coords, dim, T, rng, move, local_sigma)
        C_prop = causalset_from_coords(coords_prop)
        S_prop = bd_action_interval_abundances(C_prop, alpha, k_max)

        dS = S_prop - S
        if beta == 0.0:
            a = 1.0
        else:
            expo = -beta * dS
            expo = max(min(expo, 700.0), -700.0)
            a = math.exp(expo)

        if a >= 1.0 or rng.random() < a:
            coords = coords_prop
            C = C_prop
            S = S_prop
            accepts += 1
        props += 1

        if it >= burn_in and ((it - burn_in) % thin == 0):
            hist = sample_interval_histogram(C, rng, n_pairs=sample_pairs, k_cap=k_cap)
            pr = rebin_hist(hist, edges)
            prn = normalize(pr)
            hsum += prn

            js4_list.append(js_divergence(prn, P4D))
            p20_list.append(p_small_from_hist(hist, 20))
            if PKR is not None:
                jskr_list.append(js_divergence(prn, PKR))

    js4 = np.array(js4_list, dtype=float)
    jskr = np.array(jskr_list, dtype=float) if PKR is not None else None
    p20 = np.array(p20_list, dtype=float)
    mean_hist = normalize(hsum)

    return ChainCOut(
        js_to_4d=js4,
        js_to_kr=jskr,
        psmall20=p20,
        accept_rate=accepts / max(1, props),
        mean_hist_rebinned=mean_hist,
    )


# -----------------------------
# Multi-chain summarization
# -----------------------------
@dataclass
class GridPointCStats:
    js4_mean: float
    js4_stderr: float
    js4_ess: float
    jskr_mean: float
    jskr_stderr: float
    jskr_ess: float
    p20_mean: float
    p20_stderr: float
    p20_ess: float
    acc_mean: float
    n_samples: int
    mean_hist_rebinned: np.ndarray


def _summarize_series(chains: List[np.ndarray]) -> Tuple[float, float, float, int]:
    if not chains:
        return float("nan"), float("nan"), 0.0, 0
    allv = np.concatenate(chains) if len(chains) > 1 else chains[0]
    if allv.size == 0:
        return float("nan"), float("nan"), 0.0, 0
    ess_total = float(sum(ess_1d(c) for c in chains if c.size > 0))
    mean = float(allv.mean())
    var = float(allv.var(ddof=1)) if allv.size > 1 else 0.0
    stderr = math.sqrt(var / max(1.0, ess_total))
    return mean, stderr, ess_total, int(allv.size)


def summarize_c_chains(chains: List[ChainCOut]) -> GridPointCStats:
    js4_ch = [c.js_to_4d for c in chains]
    js4_mean, js4_stderr, js4_ess, n = _summarize_series(js4_ch)

    if chains and chains[0].js_to_kr is not None:
        jskr_ch = [c.js_to_kr for c in chains if c.js_to_kr is not None]  # type: ignore
        jskr_mean, jskr_stderr, jskr_ess, _ = _summarize_series(jskr_ch)  # type: ignore
    else:
        jskr_mean, jskr_stderr, jskr_ess = float("nan"), float("nan"), 0.0

    p20_ch = [c.psmall20 for c in chains]
    p20_mean, p20_stderr, p20_ess, _ = _summarize_series(p20_ch)

    acc_mean = float(np.mean([c.accept_rate for c in chains])) if chains else float("nan")
    mean_hist = normalize(np.mean(np.stack([c.mean_hist_rebinned for c in chains], axis=0), axis=0)) if chains else np.array([])

    return GridPointCStats(
        js4_mean=js4_mean,
        js4_stderr=js4_stderr,
        js4_ess=js4_ess,
        jskr_mean=jskr_mean,
        jskr_stderr=jskr_stderr,
        jskr_ess=jskr_ess,
        p20_mean=p20_mean,
        p20_stderr=p20_stderr,
        p20_ess=p20_ess,
        acc_mean=acc_mean,
        n_samples=n,
        mean_hist_rebinned=mean_hist,
    )


# -----------------------------
# Plotting
# -----------------------------
def make_plots(
    outdir: str,
    dim: int,
    Ns: List[int],
    betas: List[float],
    js4: Dict[int, np.ndarray],
    js4_err: Dict[int, np.ndarray],
    p20: Dict[int, np.ndarray],
    p20_err: Dict[int, np.ndarray],
    acc: Dict[int, np.ndarray],
    ref_p20_band: Dict[int, Tuple[float, float]],
    overlays: Dict[int, Dict[float, np.ndarray]],
    edges: List[Tuple[int, int]],
    include_kr: bool,
    jskr: Dict[int, np.ndarray],
    jskr_err: Dict[int, np.ndarray],
) -> str:
    if plt is None:
        raise RuntimeError("Plotting requires matplotlib. Install it or run with plotting disabled.")
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, "protocolC_interval_stats.png")

    x = np.arange(len(edges), dtype=float)

    fig = plt.figure(figsize=(18, 10))

    # JS to 4D vs beta
    ax1 = fig.add_subplot(2, 2, 1)
    for N in Ns:
        ax1.errorbar(betas, js4[N], yerr=js4_err[N], marker="o", capsize=3, label=f"N={N}")
    ax1.set_title("Protocol C: JS(P_selected, P_4D) vs beta")
    ax1.set_xlabel("beta")
    ax1.set_ylabel("JS divergence to 4D")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # p_small(20) vs beta with 4D band
    ax2 = fig.add_subplot(2, 2, 2)
    for N in Ns:
        ax2.errorbar(betas, p20[N], yerr=p20_err[N], marker="o", capsize=3, label=f"N={N}")
        mu, sd = ref_p20_band[N]
        ax2.fill_between(betas, [mu - 2 * sd] * len(betas), [mu + 2 * sd] * len(betas), alpha=0.12)
    ax2.set_title("Protocol C: p_small(20) vs beta (shaded: 4D ref ±2σ)")
    ax2.set_xlabel("beta")
    ax2.set_ylabel("p_small(20)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Acceptance vs beta
    ax3 = fig.add_subplot(2, 2, 3)
    for N in Ns:
        ax3.plot(betas, acc[N], marker="o", label=f"N={N}")
    ax3.set_title("Acceptance rate vs beta")
    ax3.set_xlabel("beta")
    ax3.set_ylabel("accept rate")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Interval spectrum overlay (use N=max(Ns), show few betas)
    ax4 = fig.add_subplot(2, 2, 4)
    N0 = max(Ns)
    ol = overlays.get(N0, {})
    for b in sorted(ol.keys()):
        ax4.plot(x, ol[b], marker="o", label=f"sel N={N0} β={b:.4g}")
    ax4.set_title(f"Rebinned interval spectrum overlays (N={N0})")
    ax4.set_xlabel("bin index (rebinned k)")
    ax4.set_ylabel("P(bin)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpng, dpi=250)
    plt.close(fig)

    # Optional KR comparison plot
    if include_kr:
        outpng2 = os.path.join(outdir, "protocolC_js_to_kr.png")
        fig2 = plt.figure(figsize=(12, 6))
        ax = fig2.add_subplot(1, 1, 1)
        for N in Ns:
            ax.errorbar(betas, jskr[N], yerr=jskr_err[N], marker="o", capsize=3, label=f"N={N}")
        ax.set_title("Protocol C: JS(P_selected, P_KR) vs beta")
        ax.set_xlabel("beta")
        ax.set_ylabel("JS divergence to KR")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(outpng2, dpi=250)
        plt.close(fig2)

    return outpng


# -----------------------------
# CLI parsing helpers
# -----------------------------
def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Protocol C (on-the-fly interval statistics during coordinate MCMC)")

    # Core geometry / grid
    ap.add_argument("--dim", type=int, default=4, help="Target Minkowski dimension (default 4)")
    ap.add_argument("--T", type=float, default=1.0, help="Diamond height T (used unless density is fixed)")
    ap.add_argument("--Ns", type=str, default="120,180,250,350", help="Comma-separated N list")
    ap.add_argument("--betas", type=str, default="0.001,0.003,0.0042,0.006,0.01", help="Comma-separated beta list")

    # Action
    ap.add_argument("--kmax", type=int, default=3, help="Interval cutoff k_max for action (default 3)")
    ap.add_argument("--alpha", type=str, default="1,-9,16,-8", help="Comma-separated alpha_k list length kmax+1")

    # MCMC
    ap.add_argument("--burn", type=int, default=1000, help="MCMC burn-in steps")
    ap.add_argument("--steps", type=int, default=16000, help="MCMC sampling steps after burn-in")
    ap.add_argument("--thin", type=int, default=40, help="Record every thin steps")
    ap.add_argument("--chains", type=int, default=5, help="Independent chains per (N,beta)")
    ap.add_argument("--mix_local", type=float, default=0.5, help="P(local Gaussian move), else resample")
    ap.add_argument("--local_sigma", type=float, default=0.03, help="Sigma for local Gaussian proposal")
    ap.add_argument("--seed", type=int, default=1234, help="Master RNG seed")
    ap.add_argument("--progress", action="store_true", help="Show per-chain progress bars")

    # Protocol C sampling controls
    ap.add_argument("--sample_pairs", type=int, default=20000, help="Comparable pairs sampled per saved configuration")
    ap.add_argument("--k_cap", type=int, default=200, help="Max open interval size bucket (overflow into last bin)")
    ap.add_argument("--no_rebin", action="store_true", help="Disable rebinning (not recommended for JS stability)")

    # Reference ensembles
    ap.add_argument("--ref_sets", type=int, default=80, help="# reference 4D sprinklings per N")
    ap.add_argument("--ref_pairs", type=int, default=20000, help="Comparable pairs sampled per reference causal set")

    # Optional KR baseline
    ap.add_argument("--with_kr", action="store_true", help="Also compute KR-like reference and JS(P_sel, P_KR)")
    ap.add_argument("--kr_sets", type=int, default=80, help="# KR reference posets per N (if --with_kr)")
    ap.add_argument("--kr_p", type=float, default=0.5, help="Edge probability between adjacent KR layers (default 0.5)")

    # Density control (same as Protocol A)
    ap.add_argument("--fix_density", action="store_true",
                    help="Keep sprinkling density fixed across N by scaling T(N) using Nref,Tref.")
    ap.add_argument("--Nref", type=int, default=120, help="Reference N for density calibration")
    ap.add_argument("--Tref", type=float, default=1.0, help="Reference T for density calibration")
    ap.add_argument("--rho", type=float, default=None,
                    help="If set, keep density fixed at rho across N by scaling T(N). Overrides --fix_density.")

    # Output
    ap.add_argument("--outdir", type=str, default="protocolC_out", help="Output directory")

    args = ap.parse_args()

    dim = int(args.dim)
    Ns = parse_csv_ints(args.Ns)
    betas = parse_csv_floats(args.betas)

    k_max = int(args.kmax)
    alpha_list = parse_csv_floats(args.alpha)
    alpha = np.array(alpha_list, dtype=float)
    if len(alpha) != k_max + 1:
        raise ValueError(f"--alpha must have length kmax+1 ({k_max+1}), got {len(alpha)}")

    if not (0.0 <= args.mix_local <= 1.0):
        raise ValueError("--mix_local must be in [0,1]")

    # RNG master
    master_rng = np.random.default_rng(int(args.seed))

    # density calibration
    if args.rho is not None:
        rho0 = float(args.rho)
        density_mode = f"fixed rho={rho0:g}"
    elif args.fix_density:
        rho0 = float(args.Nref) / diamond_volume(dim, float(args.Tref))
        density_mode = f"fixed rho from (Nref={args.Nref},Tref={args.Tref}) -> rho={rho0:g}"
    else:
        rho0 = None
        density_mode = "varying rho (T fixed)"

    print("\n--- Protocol C (Interval statistics suppression; on-the-fly) ---")
    print(f"DIM={dim} | NS={Ns} | betas=[{betas[0]}, …, {betas[-1]}] ({len(betas)} points)")
    print(f"burn={args.burn} steps={args.steps} thin={args.thin} chains={args.chains}")
    print(f"moves=mixed(p_local={args.mix_local:.2f}, sigma={args.local_sigma})")
    print(f"action: k_max={k_max} | alpha={alpha.tolist()}")
    print(f"interval sampling: sample_pairs={args.sample_pairs} k_cap={args.k_cap} rebin={'no' if args.no_rebin else 'yes'}")
    print(f"4D reference: ref_sets={args.ref_sets} ref_pairs={args.ref_pairs}")
    print(f"KR baseline: {'enabled' if args.with_kr else 'disabled'}")
    print(f"density: {density_mode}\n")

    os.makedirs(args.outdir, exist_ok=True)

    # Storage for grid outputs
    js4: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    js4_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    p20: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    p20_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    acc: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}

    jskr: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    jskr_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}

    # For p_small(20) band from 4D reference
    ref_p20_band: Dict[int, Tuple[float, float]] = {}

    # For overlay plots (store selected beta histograms)
    overlays: Dict[int, Dict[float, np.ndarray]] = {N: {} for N in Ns}

    # Precompute references per N (independent of beta)
    refs_4d: Dict[int, np.ndarray] = {}
    refs_kr: Dict[int, np.ndarray] = {}
    edges_by_N: Dict[int, List[Tuple[int, int]]] = {}

    for N in Ns:
        if rho0 is None:
            T_used = float(args.T)
        else:
            T_used = T_for_fixed_density(dim, N, rho0)

        edges = default_rebin_edges(N=N, k_cap=int(args.k_cap)) if not args.no_rebin else [(k, k) for k in range(0, int(args.k_cap) + 1)]
        edges_by_N[N] = edges

        # separate RNG stream for references
        ref_seed = int(master_rng.integers(0, 2**32 - 1))
        ref_rng = np.random.default_rng(ref_seed)

        P4D, bands = reference_interval_distribution_4d(
            N=N,
            dim=dim,
            T=T_used,
            rng=ref_rng,
            n_sets=int(args.ref_sets),
            n_pairs=int(args.ref_pairs),
            k_cap=int(args.k_cap),
            edges=edges,
            progress_label=f"4D ref N={N}",
        )
        refs_4d[N] = P4D
        ref_p20_band[N] = bands[20]

        if args.with_kr:
            kr_seed = int(master_rng.integers(0, 2**32 - 1))
            kr_rng = np.random.default_rng(kr_seed)
            PKR = reference_interval_distribution_kr(
                N=N,
                rng=kr_rng,
                n_sets=int(args.kr_sets),
                n_pairs=int(args.ref_pairs),
                k_cap=int(args.k_cap),
                edges=edges,
                p=float(args.kr_p),
                progress_label=f"KR ref N={N}",
            )
            refs_kr[N] = PKR

    # Choose a few betas for overlay storage (first, middle-ish, last)
    if len(betas) >= 3:
        overlay_betas = [betas[0], betas[len(betas)//2], betas[-1]]
    else:
        overlay_betas = betas[:]

    # Run grid
    for N in Ns:
        if rho0 is None:
            T_used = float(args.T)
        else:
            T_used = T_for_fixed_density(dim, N, rho0)

        edges = edges_by_N[N]
        P4D = refs_4d[N]
        PKR = refs_kr.get(N) if args.with_kr else None

        for bi, beta in enumerate(betas):
            chains_out: List[ChainCOut] = []
            for ci in range(int(args.chains)):
                child_seed = int(master_rng.integers(0, 2**32 - 1))
                rng = np.random.default_rng(child_seed)
                chains_out.append(
                    run_protocol_c_chain(
                        N=N,
                        dim=dim,
                        T=T_used,
                        beta=float(beta),
                        alpha=alpha,
                        k_max=k_max,
                        burn_in=int(args.burn),
                        steps=int(args.steps),
                        thin=int(args.thin),
                        mix_local=float(args.mix_local),
                        local_sigma=float(args.local_sigma),
                        rng=rng,
                        sample_pairs=int(args.sample_pairs),
                        k_cap=int(args.k_cap),
                        edges=edges,
                        P4D=P4D,
                        PKR=PKR,
                        show_progress=bool(args.progress),
                    )
                )

            stats = summarize_c_chains(chains_out)
            js4[N][bi] = stats.js4_mean
            js4_err[N][bi] = stats.js4_stderr
            p20[N][bi] = stats.p20_mean
            p20_err[N][bi] = stats.p20_stderr
            acc[N][bi] = stats.acc_mean

            if args.with_kr:
                jskr[N][bi] = stats.jskr_mean
                jskr_err[N][bi] = stats.jskr_stderr

            # store overlays at selected betas
            if any(abs(beta - ob) < 1e-15 for ob in overlay_betas):
                overlays[N][float(beta)] = stats.mean_hist_rebinned

            # Print line
            if args.with_kr:
                print(
                    f"N={N:4d} beta={beta:7.4f} | T={T_used:.4f} | "
                    f"JS4={stats.js4_mean:.4f} ± {stats.js4_stderr:.4f} (ESS≈{stats.js4_ess:.1f}) | "
                    f"JSKR={stats.jskr_mean:.4f} ± {stats.jskr_stderr:.4f} (ESS≈{stats.jskr_ess:.1f}) | "
                    f"p20={stats.p20_mean:.4f} ± {stats.p20_stderr:.4f} (ESS≈{stats.p20_ess:.1f}) | "
                    f"acc={stats.acc_mean:.3f} n={stats.n_samples}"
                )
            else:
                print(
                    f"N={N:4d} beta={beta:7.4f} | T={T_used:.4f} | "
                    f"JS4={stats.js4_mean:.4f} ± {stats.js4_stderr:.4f} (ESS≈{stats.js4_ess:.1f}) | "
                    f"p20={stats.p20_mean:.4f} ± {stats.p20_stderr:.4f} (ESS≈{stats.p20_ess:.1f}) | "
                    f"acc={stats.acc_mean:.3f} n={stats.n_samples}"
                )

    # Save a quick CSV summary
    csv_path = os.path.join(args.outdir, "protocolC_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        if args.with_kr:
            f.write("N,beta,T_used,JS4,JS4_stderr,JSKR,JSKR_stderr,p20,p20_stderr,acc\n")
        else:
            f.write("N,beta,T_used,JS4,JS4_stderr,p20,p20_stderr,acc\n")
        for N in Ns:
            T_used = float(args.T) if rho0 is None else T_for_fixed_density(dim, N, rho0)
            for bi, beta in enumerate(betas):
                if args.with_kr:
                    f.write(
                        f"{N},{beta},{T_used},"
                        f"{js4[N][bi]},{js4_err[N][bi]},"
                        f"{jskr[N][bi]},{jskr_err[N][bi]},"
                        f"{p20[N][bi]},{p20_err[N][bi]},"
                        f"{acc[N][bi]}\n"
                    )
                else:
                    f.write(
                        f"{N},{beta},{T_used},"
                        f"{js4[N][bi]},{js4_err[N][bi]},"
                        f"{p20[N][bi]},{p20_err[N][bi]},"
                        f"{acc[N][bi]}\n"
                    )

    # Plotting
    outpng = make_plots(
        outdir=args.outdir,
        dim=dim,
        Ns=Ns,
        betas=betas,
        js4=js4,
        js4_err=js4_err,
        p20=p20,
        p20_err=p20_err,
        acc=acc,
        ref_p20_band=ref_p20_band,
        overlays=overlays,
        edges=edges_by_N[max(Ns)],
        include_kr=bool(args.with_kr),
        jskr=jskr,
        jskr_err=jskr_err,
    )

    print(f"\nSaved figure: {outpng}")
    print(f"Saved CSV: {csv_path}")

    print("\nProtocol C interpretation guide:")
    print("- Success signal: in the same beta window where Protocol A shows d≈4, JS4 should decrease and/or reach a minimum,")
    print("  and p_small(20) should move toward the 4D reference band (shaded).")
    if args.with_kr:
        print("- Stronger check: in that window, JS4 should be smaller than JSKR (closer to 4D than to KR-like non-manifold).")
    print("- If JS4 is flat or worsens while A 'passes', you're likely seeing a non-manifold structure masquerading as 4D in d_MM.")


if __name__ == "__main__":
    main()
