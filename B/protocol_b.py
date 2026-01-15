# -*- coding: utf-8 -*-
"""
Protocol B — Defect-induced transport validation (single-file, Protocol C style)

Implements Protocol B on top of the SAME coordinate MCMC engine used in Protocols A/C.
At each saved post-burn configuration, it:

  1) Defines defect "charge" q(x) (two operational definitions):
       - degree anomaly (indeg+outdeg) robust z-score
       - interval-participation proxy (indeg*outdeg) robust z-score

  2) Coarse-grains into mesoscopic cells (time-slice bins) at two scales.

  3) Defines a minimal defect-coupled transport:
       - node transitions to a small set of "local" future neighbors
       - T_lambda(x->y) ∝ exp(lambda * q(y))
       - induces a cell-level transition matrix Tcell

  4) Evolves a cell distribution p by a short mixing run and measures:
       - divergence of antisymmetric flux:
           div_i = Σ_j [ p_i T_ij - p_j T_ji ]

  5) Tests source law per sample:
       div_i = kappa * rho_Q(i)
     (no intercept; you can optionally extend to intercept later)

  6) Produces per-(N,beta) summaries with multi-chain ESS-adjusted stderr:
       - kappa_mean ± stderr, z-score
       - R^2 mean, ΔR^2 mean vs lambda=0 (same charge+cellsize)
       - acceptance rate, ESS

Controls (optional flags):
  --shuffle_charge : shuffle q(x) across elements (histogram preserved)
  --excise_defects : set q(x)=0 wherever |z|>tau (neutralize defect cores)

Notes / Caveats:
  - The neighbor kernel is a *local scan* in coordinate time ordering; it is not an exact link set.
    It is intended as a minimal, fast "local future" neighborhood for first-pass Protocol B runs.
  - For a stronger implementation, replace neighbor selection with true links
    using bitset link test: link iff popcount(reach[x] & past[y])==0.

Usage example:
  python protocol_b.py \
    --kmax 3 --alpha "1,-9,16,-8" \
    --betas 0.002,0.0025,0.003,0.0035,0.004 --Ns 180,250,350 --chains 5 \
    --mix_local 0.5 --cell_sizes 25,45 --lambdas 0,0.1,-0.1 \
    --nbr_rank 12 --scan_window 80 --mix_steps 60 --tau 3.0 --qmax 5.0

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
# Alexandrov interval volume + density scaling (same as A/C)
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
# Alexandrov interval sprinkling (same as A/C)
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
# BD-style action via interval abundances (same as A/C)
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
# Mixed coordinate proposal (same as A/C)
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
# ESS (same as A/C)
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
# Protocol B: charge definitions
# -----------------------------
def robust_zscore(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + eps
    return (v - med) / mad


def charge_degree(C: CausalSet, tau: float, qmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Degree anomaly charge: q(x) from robust z-score of indeg+outdeg.
    Returns (q, z) where z is the underlying robust z-score (used for excision).
    """
    N = C.N
    indeg = np.array([C.past[i].bit_count() for i in range(N)], dtype=float)
    outdeg = np.array([C.reach[i].bit_count() for i in range(N)], dtype=float)
    deg = indeg + outdeg
    z = robust_zscore(deg)
    q = np.zeros(N, dtype=float)
    mask = np.abs(z) > float(tau)
    q[mask] = np.clip(z[mask], -float(qmax), float(qmax))
    return q, z


def charge_interval_participation(C: CausalSet, tau: float, qmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interval-participation proxy: q(x) from robust z-score of indeg*outdeg.
    Returns (q, z).
    """
    N = C.N
    indeg = np.array([C.past[i].bit_count() for i in range(N)], dtype=float)
    outdeg = np.array([C.reach[i].bit_count() for i in range(N)], dtype=float)
    ip = indeg * outdeg
    z = robust_zscore(ip)
    q = np.zeros(N, dtype=float)
    mask = np.abs(z) > float(tau)
    q[mask] = np.clip(z[mask], -float(qmax), float(qmax))
    return q, z


# -----------------------------
# Protocol B: coarse graining into cells
# -----------------------------
def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def make_time_cells(N: int, cell_size: int) -> List[np.ndarray]:
    cells: List[np.ndarray] = []
    for start in range(0, N, cell_size):
        end = min(N, start + cell_size)
        cells.append(np.arange(start, end, dtype=int))
    return cells


def rho_Q_from_q(cells: List[np.ndarray], q: np.ndarray) -> np.ndarray:
    rho = np.zeros(len(cells), dtype=float)
    for i, idx in enumerate(cells):
        rho[i] = float(np.mean(q[idx])) if idx.size else 0.0
    return rho


def element_to_cell_map(cells: List[np.ndarray], N: int) -> np.ndarray:
    which = np.empty(N, dtype=int)
    for ci, idx in enumerate(cells):
        which[idx] = ci
    return which


# -----------------------------
# Protocol B: neighbor selection (fast local scan in time order)
# -----------------------------
def _is_causal(coords: np.ndarray, i: int, j: int) -> bool:
    # assumes coords time-sorted; checks Minkowski causality i ≺ j
    dt = coords[j, 0] - coords[i, 0]
    if dt <= 0:
        return False
    dx = coords[j, 1:] - coords[i, 1:]
    tau2 = dt * dt - float(np.dot(dx, dx))
    return tau2 > 0.0


def neighbors_local_future_scan(
    coords: np.ndarray,
    nbr_rank: int,
    scan_window: int,
) -> List[np.ndarray]:
    """
    For each i, scan forward up to scan_window indices and collect up to nbr_rank causal futures.
    This approximates a local neighborhood without building full link sets.
    """
    N = coords.shape[0]
    nbrs: List[np.ndarray] = []
    nbr_rank = int(nbr_rank)
    scan_window = int(scan_window)

    for i in range(N):
        js: List[int] = []
        stop = min(N, i + 1 + scan_window)
        for j in range(i + 1, stop):
            if _is_causal(coords, i, j):
                js.append(j)
                if len(js) >= nbr_rank:
                    break
        nbrs.append(np.array(js, dtype=int))
    return nbrs


# -----------------------------
# Protocol B: transport kernel and induced cell Markov chain
# -----------------------------
def build_T_lambda_node(
    nbrs: List[np.ndarray],
    q: np.ndarray,
    lam: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Per-node transition lists: (targets, probs).
    T_lambda(x->y) ∝ exp(lam * q(y)) over y in nbrs[x].
    """
    lam = float(lam)
    T: List[Tuple[np.ndarray, np.ndarray]] = []
    for js in nbrs:
        if js.size == 0:
            T.append((js, np.array([], dtype=float)))
            continue
        w = np.exp(lam * q[js])
        s = float(w.sum())
        if s <= 0.0 or not np.isfinite(s):
            p = np.ones_like(w, dtype=float) / float(len(w))
        else:
            p = (w / s).astype(float)
        T.append((js, p))
    return T


def induced_cell_transition(
    cells: List[np.ndarray],
    T_node: List[Tuple[np.ndarray, np.ndarray]],
    N: int,
) -> np.ndarray:
    """
    Tcell[i,j] = average prob of jumping from a random element in cell i into cell j.
    """
    m = len(cells)
    which = element_to_cell_map(cells, N)
    Tcell = np.zeros((m, m), dtype=float)

    for ci, idx in enumerate(cells):
        if idx.size == 0:
            continue
        acc = np.zeros(m, dtype=float)
        for x in idx:
            js, p = T_node[int(x)]
            for y, py in zip(js, p):
                acc[which[int(y)]] += float(py)
        Tcell[ci, :] = acc / float(idx.size)

    # renormalize rows (handle dead rows)
    rs = Tcell.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    Tcell = Tcell / rs
    return Tcell


def evolve_cell_chain(Tcell: np.ndarray, steps: int, p0: np.ndarray) -> np.ndarray:
    p = p0.copy()
    for _ in range(int(steps)):
        p = p @ Tcell
    return p


def divergence_from_Tcell(Tcell: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    div_i = Σ_j [ p_i T_ij - p_j T_ji ]
    """
    m = Tcell.shape[0]
    div = np.zeros(m, dtype=float)
    # vectorized form
    for i in range(m):
        div[i] = float(np.sum(p[i] * Tcell[i, :] - p * Tcell[:, i]))
    return div


# -----------------------------
# Protocol B: regression div = kappa * rho
# -----------------------------
def linreg_kappa(div: np.ndarray, rho: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit div = kappa * rho (no intercept).
    Returns (kappa_hat, se_hat, r2).
    """
    x = np.asarray(rho, dtype=float)
    y = np.asarray(div, dtype=float)

    # drop NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return float("nan"), float("nan"), float("nan")

    k = float(np.dot(x, y) / denom)
    resid = y - k * x
    rss = float(np.dot(resid, resid))
    sigma2 = rss / max(1, n - 1)
    se = math.sqrt(sigma2 / denom)

    tss = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - (rss / tss) if tss > 1e-12 else float("nan")
    return k, se, r2


# -----------------------------
# Protocol B chain output
# -----------------------------
@dataclass
class ChainBOut:
    accept_rate: float
    # key -> series arrays (per saved config)
    kappa: Dict[str, np.ndarray]
    r2: Dict[str, np.ndarray]
    dr2: Dict[str, np.ndarray]  # delta R2 vs lambda=0 (same charge+cellsize)


def _key(charge_name: str, cell_size: int, lam: float) -> str:
    return f"{charge_name}|cell{int(cell_size)}|lam{float(lam):+.6g}"


def _key0(charge_name: str, cell_size: int) -> str:
    return f"{charge_name}|cell{int(cell_size)}|lam0"


def run_protocol_b_chain(
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
    cell_sizes: List[int],
    lambdas: List[float],
    nbr_rank: int,
    scan_window: int,
    mix_steps: int,
    tau: float,
    qmax: float,
    shuffle_charge: bool,
    excise_defects: bool,
    show_progress: bool = False,
) -> ChainBOut:
    coords = sprinkle_diamond(N, dim, T, rng)
    C = causalset_from_coords(coords)
    S = bd_action_interval_abundances(C, alpha, k_max)

    accepts = 0
    props = 0

    # storage per key
    kappa_series: Dict[str, List[float]] = {}
    r2_series: Dict[str, List[float]] = {}
    dr2_series: Dict[str, List[float]] = {}

    total_iters = burn_in + steps
    it_range = range(total_iters)
    if show_progress and tqdm is not None:
        it_range = tqdm(it_range, desc=f"B-Chain N={N} β={beta:.4f}", leave=False)

    lambdas = [float(l) for l in lambdas]
    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas

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

        if it < burn_in or ((it - burn_in) % thin != 0):
            continue

        # neighbor lists for this configuration (shared across charges/lambdas/cells)
        nbrs = neighbors_local_future_scan(coords, nbr_rank=nbr_rank, scan_window=scan_window)

        # charges
        charges: List[Tuple[str, np.ndarray, np.ndarray]] = []
        q1, z1 = charge_degree(C, tau=tau, qmax=qmax)
        charges.append(("deg", q1, z1))
        q2, z2 = charge_interval_participation(C, tau=tau, qmax=qmax)
        charges.append(("ip", q2, z2))

        for charge_name, q, z in charges:
            q_use = q.copy()

            if shuffle_charge:
                q_use = q_use.copy()
                rng.shuffle(q_use)

            if excise_defects:
                # neutralize defect cores: set q=0 wherever |z|>tau (i.e. where q!=0 for our definitions)
                q_use[np.abs(z) > float(tau)] = 0.0

            for cell_size in cell_sizes:
                cells = make_time_cells(N, int(cell_size))
                m = len(cells)
                rho = rho_Q_from_q(cells, q_use)

                # uniform initial cell distribution
                p0 = np.ones(m, dtype=float) / float(m)

                # compute lambda=0 baseline R2 for delta-R2
                r2_0: Optional[float] = None

                for lam in lambdas:
                    T_node = build_T_lambda_node(nbrs, q_use, lam=lam)
                    Tcell = induced_cell_transition(cells, T_node, N=N)
                    p = evolve_cell_chain(Tcell, steps=mix_steps, p0=p0)
                    div = divergence_from_Tcell(Tcell, p)

                    k_hat, se_hat, r2_hat = linreg_kappa(div=div, rho=rho)

                    kkey = _key(charge_name, cell_size, lam)
                    rkey = kkey

                    kappa_series.setdefault(kkey, []).append(float(k_hat))
                    r2_series.setdefault(rkey, []).append(float(r2_hat))

                    if abs(lam) < 1e-15:
                        r2_0 = float(r2_hat)
                        dr2_series.setdefault(_key(charge_name, cell_size, lam), []).append(0.0)
                    else:
                        if r2_0 is None or not np.isfinite(r2_0):
                            dr2 = float("nan")
                        else:
                            dr2 = float(r2_hat - r2_0)
                        dr2_series.setdefault(kkey, []).append(float(dr2))

    kappa_out = {k: np.asarray(v, dtype=float) for k, v in kappa_series.items()}
    r2_out = {k: np.asarray(v, dtype=float) for k, v in r2_series.items()}
    dr2_out = {k: np.asarray(v, dtype=float) for k, v in dr2_series.items()}

    return ChainBOut(
        accept_rate=accepts / max(1, props),
        kappa=kappa_out,
        r2=r2_out,
        dr2=dr2_out,
    )


# -----------------------------
# Multi-chain summarization
# -----------------------------
@dataclass
class Stat1D:
    mean: float
    stderr: float
    ess: float
    n: int


def summarize_1d_series(chains: List[np.ndarray]) -> Stat1D:
    if not chains:
        return Stat1D(mean=float("nan"), stderr=float("nan"), ess=0.0, n=0)

    allv = np.concatenate(chains) if len(chains) > 1 else chains[0]
    allv = np.asarray(allv, dtype=float)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return Stat1D(mean=float("nan"), stderr=float("nan"), ess=0.0, n=0)

    ess_total = float(sum(ess_1d(c[np.isfinite(c)]) for c in chains if np.isfinite(c).any()))
    mean = float(allv.mean())
    var = float(allv.var(ddof=1)) if allv.size > 1 else 0.0
    stderr = math.sqrt(var / max(1.0, ess_total))
    return Stat1D(mean=mean, stderr=stderr, ess=ess_total, n=int(allv.size))


@dataclass
class GridPointBStats:
    acc_mean: float
    # key -> stats
    kappa: Dict[str, Stat1D]
    r2: Dict[str, Stat1D]
    dr2: Dict[str, Stat1D]


def summarize_b_chains(chains: List[ChainBOut]) -> GridPointBStats:
    acc_mean = float(np.mean([c.accept_rate for c in chains])) if chains else float("nan")

    # collect all keys present
    keys = set()
    for c in chains:
        keys |= set(c.kappa.keys())

    kappa_stats: Dict[str, Stat1D] = {}
    r2_stats: Dict[str, Stat1D] = {}
    dr2_stats: Dict[str, Stat1D] = {}

    for k in sorted(keys):
        kappa_stats[k] = summarize_1d_series([c.kappa.get(k, np.array([], dtype=float)) for c in chains])
        r2_stats[k] = summarize_1d_series([c.r2.get(k, np.array([], dtype=float)) for c in chains])
        dr2_stats[k] = summarize_1d_series([c.dr2.get(k, np.array([], dtype=float)) for c in chains])

    return GridPointBStats(acc_mean=acc_mean, kappa=kappa_stats, r2=r2_stats, dr2=dr2_stats)


# -----------------------------
# Plotting
# -----------------------------
def pick_plot_keys(
    lambdas: List[float],
    cell_sizes: List[int],
) -> Tuple[str, str, float, int]:
    """
    Choose one representative setting for compact plots:
      - charge='deg'
      - smallest cell size
      - positive nonzero lambda if present else first nonzero
    """
    charge = "deg"
    cell = int(min(cell_sizes))
    lpos = [l for l in lambdas if l > 0]
    lneg = [l for l in lambdas if l < 0]
    if lpos:
        lam = float(sorted(lpos)[0])
    elif lneg:
        lam = float(sorted(lneg, reverse=True)[0])  # closest to 0 negative
    else:
        lam = 0.0
    key = _key(charge, cell, lam)
    key0 = _key(charge, cell, 0.0)
    return key, key0, lam, cell


def make_plots(
    outdir: str,
    Ns: List[int],
    betas: List[float],
    plot_key: str,
    plot_key0: str,
    lam_used: float,
    cell_used: int,
    # per N arrays over beta:
    kappa_mean: Dict[int, np.ndarray],
    kappa_err: Dict[int, np.ndarray],
    dr2_mean: Dict[int, np.ndarray],
    dr2_err: Dict[int, np.ndarray],
    acc: Dict[int, np.ndarray],
    ess: Dict[int, np.ndarray],
) -> str:
    if plt is None:
        raise RuntimeError("Plotting requires matplotlib. Install it or run with plotting disabled.")
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, "protocolB_transport.png")

    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    for N in Ns:
        ax1.errorbar(betas, kappa_mean[N], yerr=kappa_err[N], marker="o", capsize=3, label=f"N={N}")
    ax1.axhline(0.0, linestyle="--", linewidth=1.2)
    ax1.set_title(f"Protocol B: κ vs β (key={plot_key})")
    ax1.set_xlabel("beta")
    ax1.set_ylabel("kappa")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    for N in Ns:
        ax2.errorbar(betas, dr2_mean[N], yerr=dr2_err[N], marker="o", capsize=3, label=f"N={N}")
    ax2.axhline(0.0, linestyle="--", linewidth=1.2)
    ax2.set_title(f"ΔR² vs β (λ={lam_used:+g}, cell={cell_used}; relative to λ=0)")
    ax2.set_xlabel("beta")
    ax2.set_ylabel("ΔR²")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    for N in Ns:
        ax3.plot(betas, ess[N], marker="o", label=f"N={N}")
    ax3.set_title("ESS(κ) vs β (summed over chains)")
    ax3.set_xlabel("beta")
    ax3.set_ylabel("ESS")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    for N in Ns:
        ax4.plot(betas, acc[N], marker="o", label=f"N={N}")
    ax4.set_title("Acceptance rate vs β")
    ax4.set_xlabel("beta")
    ax4.set_ylabel("accept rate")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(outpng, dpi=250)
    plt.close(fig)
    return outpng


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Protocol B (defect-induced transport validation)")

    # Core geometry / grid
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--Ns", type=str, default="180,250,350")
    ap.add_argument("--betas", type=str, default="0.002,0.0025,0.003,0.0035,0.004")

    # Action
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--alpha", type=str, default="1,-9,16,-8")

    # MCMC
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--thin", type=int, default=40)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--mix_local", type=float, default=0.5)
    ap.add_argument("--local_sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--progress", action="store_true")

    # Density control (same as A/C)
    ap.add_argument("--fix_density", action="store_true")
    ap.add_argument("--Nref", type=int, default=120)
    ap.add_argument("--Tref", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=None)

    # Protocol B parameters
    ap.add_argument("--cell_sizes", type=str, default="25,45", help="Comma-separated cell sizes (time bins)")
    ap.add_argument("--lambdas", type=str, default="0,0.1,-0.1", help="Comma-separated lambda values (must include 0 or it is auto-added)")
    ap.add_argument("--nbr_rank", type=int, default=12, help="Max number of causal future neighbors per node")
    ap.add_argument("--scan_window", type=int, default=80, help="How far forward to scan in time order for neighbors")
    ap.add_argument("--mix_steps", type=int, default=60, help="Steps to evolve cell chain before measuring divergence")

    # Charge parameters
    ap.add_argument("--tau", type=float, default=3.0, help="Robust z-score threshold for defect support")
    ap.add_argument("--qmax", type=float, default=5.0, help="Charge clipping")

    # Controls
    ap.add_argument("--shuffle_charge", action="store_true", help="Shuffle q(x) across elements (control)")
    ap.add_argument("--excise_defects", action="store_true", help="Set q(x)=0 on defect cores (control)")

    # Output
    ap.add_argument("--outdir", type=str, default="protocolB_out")

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

    cell_sizes = parse_csv_ints(args.cell_sizes)
    lambdas = parse_csv_floats(args.lambdas)
    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas

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

    # Display header
    print("\n--- Protocol B (Defect-induced transport validation) ---")
    print(f"DIM={dim} | NS={Ns} | betas=[{betas[0]}, …, {betas[-1]}] ({len(betas)} points)")
    print(f"burn={args.burn} steps={args.steps} thin={args.thin} chains={args.chains}")
    print(f"moves=mixed(p_local={args.mix_local:.2f}, sigma={args.local_sigma})")
    print(f"action: k_max={k_max} | alpha={alpha.tolist()}")
    print(f"density: {density_mode}")
    print(f"cells: {cell_sizes} | lambdas={lambdas}")
    print(f"neighbors: nbr_rank={args.nbr_rank} scan_window={args.scan_window} | mix_steps={args.mix_steps}")
    print(f"charge: tau={args.tau} qmax={args.qmax}")
    print(f"controls: shuffle_charge={bool(args.shuffle_charge)} excise_defects={bool(args.excise_defects)}\n")

    os.makedirs(args.outdir, exist_ok=True)

    # Choose one representative key for compact plots
    plot_key, plot_key0, lam_used, cell_used = pick_plot_keys(lambdas=lambdas, cell_sizes=cell_sizes)

    # Storage for plot series: per N -> arrays over beta
    kappa_mean: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    kappa_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    dr2_mean: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    dr2_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    acc: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    ess_plot: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}

    # CSV rows
    csv_path = os.path.join(args.outdir, "protocolB_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write(
            "N,beta,T_used,charge,cell_size,lambda,"
            "kappa_mean,kappa_stderr,kappa_z,"
            "r2_mean,r2_stderr,dr2_mean,dr2_stderr,"
            "ess_kappa,acc_mean,n_samples\n"
        )

        # Run grid
        for N in Ns:
            T_used = float(args.T) if rho0 is None else T_for_fixed_density(dim, N, rho0)

            for bi, beta in enumerate(betas):
                chains_out: List[ChainBOut] = []
                for _ in range(int(args.chains)):
                    child_seed = int(master_rng.integers(0, 2**32 - 1))
                    rng = np.random.default_rng(child_seed)

                    chains_out.append(
                        run_protocol_b_chain(
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
                            cell_sizes=cell_sizes,
                            lambdas=lambdas,
                            nbr_rank=int(args.nbr_rank),
                            scan_window=int(args.scan_window),
                            mix_steps=int(args.mix_steps),
                            tau=float(args.tau),
                            qmax=float(args.qmax),
                            shuffle_charge=bool(args.shuffle_charge),
                            excise_defects=bool(args.excise_defects),
                            show_progress=bool(args.progress),
                        )
                    )

                stats = summarize_b_chains(chains_out)
                acc_mean_val = stats.acc_mean

                # For compact plots, extract chosen key stats
                ks = stats.kappa.get(plot_key, Stat1D(float("nan"), float("nan"), 0.0, 0))
                ds = stats.dr2.get(plot_key, Stat1D(float("nan"), float("nan"), 0.0, 0))
                kappa_mean[N][bi] = ks.mean
                kappa_err[N][bi] = ks.stderr
                dr2_mean[N][bi] = ds.mean
                dr2_err[N][bi] = ds.stderr
                acc[N][bi] = acc_mean_val
                ess_plot[N][bi] = ks.ess

                # Print one representative line (like C)
                kz = (ks.mean / ks.stderr) if (np.isfinite(ks.mean) and np.isfinite(ks.stderr) and ks.stderr > 0) else float("nan")
                print(
                    f"N={N:4d} beta={beta:7.4f} | T={T_used:.4f} | "
                    f"[plot] κ={ks.mean:+.4e} ± {ks.stderr:.2e} (z={kz:+.2f}, ESS≈{ks.ess:.1f}) | "
                    f"ΔR²={ds.mean:+.3f} ± {ds.stderr:.3f} | acc={acc_mean_val:.3f}"
                )

                # Write full CSV rows for all keys (charge/cell/lambda)
                for k, kstat in stats.kappa.items():
                    # parse key
                    # format: charge|cellXX|lam+...
                    parts = k.split("|")
                    charge = parts[0]
                    cell_part = parts[1]
                    lam_part = parts[2]
                    cell_size = int(cell_part.replace("cell", ""))
                    lam_val = float(lam_part.replace("lam", ""))

                    rstat = stats.r2.get(k, Stat1D(float("nan"), float("nan"), 0.0, 0))
                    dstat = stats.dr2.get(k, Stat1D(float("nan"), float("nan"), 0.0, 0))

                    kz = (kstat.mean / kstat.stderr) if (np.isfinite(kstat.mean) and np.isfinite(kstat.stderr) and kstat.stderr > 0) else float("nan")

                    fcsv.write(
                        f"{N},{beta},{T_used},"
                        f"{charge},{cell_size},{lam_val},"
                        f"{kstat.mean},{kstat.stderr},{kz},"
                        f"{rstat.mean},{rstat.stderr},"
                        f"{dstat.mean},{dstat.stderr},"
                        f"{kstat.ess},{acc_mean_val},{kstat.n}\n"
                    )

    # Plotting (compact)
    if plt is not None:
        outpng = make_plots(
            outdir=args.outdir,
            Ns=Ns,
            betas=betas,
            plot_key=plot_key,
            plot_key0=plot_key0,
            lam_used=lam_used,
            cell_used=cell_used,
            kappa_mean=kappa_mean,
            kappa_err=kappa_err,
            dr2_mean=dr2_mean,
            dr2_err=dr2_err,
            acc=acc,
            ess=ess_plot,
        )
        print(f"\nSaved figure: {outpng}")
    else:
        print("\nmatplotlib not available; skipping plots.")

    print(f"Saved CSV: {csv_path}")

    print("\nProtocol B interpretation guide:")
    print("- Primary success signal (for λ≠0): κ significantly nonzero with correct sign, and ΔR² > 0 vs λ=0.")
    print("- Robustness checks (not automated here): compare across N, both cell sizes, and both charge definitions (deg, ip).")
    print("- Controls: --shuffle_charge should collapse κ and ΔR²; --excise_defects should also collapse them.")


if __name__ == "__main__":
    main()
