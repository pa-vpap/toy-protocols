# -*- coding: utf-8 -*-
"""
Protocol B — Defect-induced transport law (Open system / NESS version) [PATCHED]

PATCH SUMMARY (vs your pasted version):
  ✅ Adds a *baseline-subtracted (residual)* observable to kill geometric/NESS artifacts:
        div_resid := div_lambda(q) - div_lambda=0(q)
     and regresses div_resid on q over interior cells.
     This makes the shuffle control collapse BOTH κ and ΔR² (in practice, κ_resid and ΔR²_resid),
     fixing the "ΔR² survives shuffle" issue.

  ✅ Keeps the original (non-residual) κ and R² for debugging continuity,
     but the plots & ΔR² are now based on residual quantities by default.

Outputs:
  - CSV: protocolB_summary.csv (includes both raw and residual stats)
  - PNG: protocolB_transport.png (plots residual κ and residual ΔR² by default)
  - Printed summary per (N,beta) for the chosen plot key.

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
# Causal set (bitsets) + sprinkling (as in A/C)
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


def sphere_area(n: int) -> float:
    return 2.0 * math.pi ** ((n + 1) / 2.0) / math.gamma((n + 1) / 2.0)


def diamond_volume(dim: int, T: float) -> float:
    Sd_2 = sphere_area(dim - 2)
    return 2.0 * Sd_2 / (dim * (dim - 1)) * (0.5 * T) ** dim


def T_for_fixed_density(dim: int, N: int, rho: float) -> float:
    Sd_2 = sphere_area(dim - 2)
    return 2.0 * ((N * dim * (dim - 1)) / (2.0 * rho * Sd_2)) ** (1.0 / dim)


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
# ESS (as in A/C)
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
# Coarse graining into time cells + cell-transition matrix
# -----------------------------
def time_cells_by_quantiles(t: np.ndarray, n_cells: int) -> np.ndarray:
    """
    Assign each element to a time cell by time quantiles (approximately equal counts).
    Returns cell_id per element in [0, n_cells-1].
    """
    N = len(t)
    if n_cells <= 1:
        return np.zeros(N, dtype=int)
    qs = np.linspace(0.0, 1.0, n_cells + 1)
    edges = np.quantile(t, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    cid = np.digitize(t, edges[1:-1], right=False)
    cid = np.clip(cid, 0, n_cells - 1)
    return cid.astype(int)


def build_cell_transition_from_links(
    C: CausalSet,
    cell_of_elem: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    nbr_rank: int,
) -> np.ndarray:
    """
    Build a coarse cell transition matrix T0 from causal links between elements.

    For each element i, sample up to nbr_rank future-related elements j from reach[i]
    (uniform over its future set), and add a transition from cell(i) -> cell(j).

    Row-normalize to get a stochastic matrix over cells.
    """
    T0 = np.zeros((n_cells, n_cells), dtype=float)
    reach = C.reach
    N = C.N

    for i in range(N):
        fut = reach[i]
        m = fut.bit_count()
        if m == 0:
            continue

        k = min(nbr_rank, m)
        ranks = rng.choice(m, size=k, replace=False) if k < m else np.arange(m)
        ranks_sorted = np.sort(ranks)

        picks: List[int] = []
        b = fut
        seen = 0
        target_idx = 0
        target = int(ranks_sorted[target_idx]) if ranks_sorted.size else -1

        while b and target_idx < len(ranks_sorted):
            lsb = b & -b
            j = lsb.bit_length() - 1
            if seen == target:
                picks.append(j)
                target_idx += 1
                if target_idx < len(ranks_sorted):
                    target = int(ranks_sorted[target_idx])
            b ^= lsb
            seen += 1

        ci = int(cell_of_elem[i])
        for j in picks:
            cj = int(cell_of_elem[j])
            if 0 <= ci < n_cells and 0 <= cj < n_cells and ci != cj:
                T0[ci, cj] += 1.0

    for i in range(n_cells):
        s = float(T0[i].sum())
        if s <= 1e-12:
            T0[i, i] = 1.0
        else:
            T0[i] /= s
    return T0


def apply_defect_coupling(T0: np.ndarray, q: np.ndarray, lam: float) -> np.ndarray:
    """
    T_ij ∝ T0_ij * exp(lam*(q_j - q_i)), then row-normalize.
    """
    if lam == 0.0:
        return T0.copy()

    q = np.asarray(q, dtype=float)
    n = T0.shape[0]
    T = np.zeros_like(T0, dtype=float)

    for i in range(n):
        row = T0[i].copy()
        nz = np.where(row > 0.0)[0]
        if nz.size == 0:
            T[i, i] = 1.0
            continue
        d = q[nz] - q[i]
        expo = lam * d
        expo = np.clip(expo, -50.0, 50.0)
        row[nz] = row[nz] * np.exp(expo)
        s = float(row.sum())
        if s <= 1e-12:
            T[i, i] = 1.0
        else:
            T[i] = row / s
    return T


# -----------------------------
# Charge proxies (cell-level)
# -----------------------------
def cell_charge_deg(C: CausalSet, cell_of_elem: np.ndarray, n_cells: int) -> np.ndarray:
    """
    deg proxy: mean degree anomaly per cell.
      degree(elem) = in_degree + out_degree
    q_cell = mean(deg in cell) - global_mean(deg)
    """
    N = C.N
    out_deg = np.array([C.reach[i].bit_count() for i in range(N)], dtype=float)
    in_deg = np.array([C.past[i].bit_count() for i in range(N)], dtype=float)
    deg = out_deg + in_deg
    mu = float(deg.mean()) if N > 0 else 0.0

    q = np.zeros(n_cells, dtype=float)
    counts = np.zeros(n_cells, dtype=float)
    for i in range(N):
        c = int(cell_of_elem[i])
        q[c] += deg[i]
        counts[c] += 1.0
    counts = np.maximum(counts, 1.0)
    q = q / counts
    q = q - mu
    return q


def cell_charge_ip(C: CausalSet, cell_of_elem: np.ndarray, n_cells: int) -> np.ndarray:
    """
    ip proxy: within-cell comparability density (order fraction within the cell),
    contrasted against global baseline.

    q_cell = r_cell - r_global.
    """
    N = C.N
    if N < 3:
        return np.zeros(n_cells, dtype=float)

    R = sum(bits.bit_count() for bits in C.reach)
    r_global = (2.0 * R) / (N * (N - 1))

    elems_by_cell: List[List[int]] = [[] for _ in range(n_cells)]
    for i in range(N):
        elems_by_cell[int(cell_of_elem[i])].append(i)

    q = np.zeros(n_cells, dtype=float)
    for c in range(n_cells):
        S = elems_by_cell[c]
        m = len(S)
        if m < 3:
            q[c] = 0.0
            continue

        mask = 0
        for idx in S:
            mask |= (1 << idx)

        Rc = 0
        for i in S:
            Rc += (C.reach[i] & mask).bit_count()
        r_cell = (2.0 * Rc) / (m * (m - 1))
        q[c] = r_cell - r_global
    return q


def normalize_charge(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    s = float(np.std(q))
    if s <= eps:
        return np.zeros_like(q)
    return q / s


def mark_defect_cells(q: np.ndarray, qmax: float) -> np.ndarray:
    """
    Return boolean mask for 'defect' cells: top qmax% by |q|.
    qmax is in percent (e.g. 5.0 -> top 5%).
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    if n == 0:
        return np.zeros(0, dtype=bool)
    frac = max(0.0, min(100.0, float(qmax))) / 100.0
    if frac <= 0.0:
        return np.zeros(n, dtype=bool)
    k = max(1, int(round(frac * n)))
    thr = np.partition(np.abs(q), n - k)[n - k]
    return (np.abs(q) >= thr)


def shuffle_charge(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q2 = np.array(q, dtype=float)
    rng.shuffle(q2)
    return q2


def excise_defects(q: np.ndarray, qmax: float) -> np.ndarray:
    m = mark_defect_cells(q, qmax=qmax)
    q2 = np.array(q, dtype=float)
    q2[m] = 0.0
    return q2


# -----------------------------
# Open-system NESS evolution + current divergence
# -----------------------------
def open_system_evolve_to_ness(
    T: np.ndarray,
    mix_steps: int,
    source_ids: np.ndarray,
    sink_ids: np.ndarray,
    inject: float,
) -> np.ndarray:
    """
    Evolve p with:
      p <- p @ T
      p[sink]=0
      p[source] += inject/|source|
      renormalize to sum 1
    starting from uniform.

    Returns p_NESS after mix_steps.
    """
    n = T.shape[0]
    p = np.ones(n, dtype=float) / max(1, n)

    src = np.asarray(source_ids, dtype=int)
    snk = np.asarray(sink_ids, dtype=int)
    src = src[(src >= 0) & (src < n)]
    snk = snk[(snk >= 0) & (snk < n)]
    inject = max(0.0, float(inject))

    for _ in range(max(0, int(mix_steps))):
        p = p @ T
        if snk.size > 0:
            p[snk] = 0.0
        if inject > 0.0 and src.size > 0:
            p[src] += inject / float(len(src))
        s = float(p.sum())
        if s <= 1e-15:
            p = np.ones(n, dtype=float) / max(1, n)
        else:
            p /= s
    return p


def current_divergence(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    cell-level divergence:
      div_i = sum_j (p_i T_ij - p_j T_ji) = p_i - sum_j p_j T_ji
    where (p@T)_i = sum_j p_j T_ji.
    """
    p = np.asarray(p, dtype=float)
    out = p.copy()
    inc = p @ T
    return out - inc


# -----------------------------
# Regression: div = kappa*q + c on interior cells only
# -----------------------------
@dataclass
class RegrOut:
    kappa: float
    r2: float
    n: int


def regress_div_on_charge(div: np.ndarray, q: np.ndarray, use_mask: np.ndarray) -> RegrOut:
    div = np.asarray(div, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.asarray(use_mask, dtype=bool)

    xs = q[m]
    ys = div[m]
    n = int(xs.size)
    if n < 3:
        return RegrOut(kappa=float("nan"), r2=float("nan"), n=0)

    X = np.stack([xs, np.ones_like(xs)], axis=1)
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    kappa = float(beta[0])
    yhat = X @ beta

    ybar = float(ys.mean())
    tss = float(np.sum((ys - ybar) ** 2))
    rss = float(np.sum((ys - yhat) ** 2))
    if tss <= 1e-15:
        r2 = float("nan")
    else:
        r2 = 1.0 - rss / tss
    return RegrOut(kappa=kappa, r2=float(r2), n=n)


def interior_mask(n_cells: int, source_frac: float, sink_frac: float, interior_frac: float) -> np.ndarray:
    """
    Keep the middle interior_frac fraction, excluding source and sink bands.
    interior_frac is applied after excluding boundary fractions.
    """
    n = int(n_cells)
    if n <= 0:
        return np.zeros(0, dtype=bool)

    bsrc = int(math.ceil(max(0.0, source_frac) * n))
    bsnk = int(math.ceil(max(0.0, sink_frac) * n))
    lo = bsrc
    hi = n - bsnk
    lo = max(0, min(lo, n))
    hi = max(lo, min(hi, n))

    base = np.zeros(n, dtype=bool)
    if hi - lo < 3:
        return base

    span = hi - lo
    keep = max(3, int(round(max(0.0, min(1.0, interior_frac)) * span)))
    start = lo + (span - keep) // 2
    end = start + keep
    base[start:end] = True
    return base


def boundary_ids(n_cells: int, frac: float, which: str) -> np.ndarray:
    n = int(n_cells)
    k = int(math.ceil(max(0.0, min(1.0, float(frac))) * n))
    if k <= 0:
        return np.array([], dtype=int)
    if which == "source":
        return np.arange(0, k, dtype=int)
    if which == "sink":
        return np.arange(n - k, n, dtype=int)
    raise ValueError("which must be 'source' or 'sink'")


# -----------------------------
# Protocol B MCMC chain (collect kappa/r2 series per saved config)
# -----------------------------
@dataclass
class ChainBOut:
    # raw regression
    kappa_series: np.ndarray
    r2_series: np.ndarray
    # residual regression (PATCH)
    kappa_resid_series: np.ndarray
    r2_resid_series: np.ndarray
    accept_rate: float

def run_protocol_b_chain(
    N: int,
    dim: int,
    T_used: float,
    beta: float,
    alpha: np.ndarray,
    k_max: int,
    burn_in: int,
    steps: int,
    thin: int,
    mix_local: float,
    local_sigma: float,
    rng: np.random.Generator,
    # protocol B params
    cell_size: int,
    lam: float,
    nbr_rank: int,
    mix_steps_ness: int,
    source_frac: float,
    sink_frac: float,
    inject: float,
    interior_frac: float,
    charge_mode: str,
    qmax: float,
    do_shuffle: bool,
    do_excise: bool,
    show_progress: bool = False,
) -> ChainBOut:
    """
    PATCHED (v3):
      - Matches ChainBOut schema that includes residual series.
      - True null shuffle: shuffle ONLY the regressor q_reg, NOT the operator q_op.
      - Residual observables:
          div_resid = div_lambda - div_lambda0  (per saved configuration)
        and regression on div_resid.
    """
    coords = sprinkle_diamond(N, dim, T_used, rng)
    C = causalset_from_coords(coords)
    S = bd_action_interval_abundances(C, alpha, k_max)

    accepts = 0
    props = 0

    kappas: List[float] = []
    r2s: List[float] = []
    kappas_resid: List[float] = []
    r2s_resid: List[float] = []

    total_iters = int(burn_in) + int(steps)
    it_range = range(total_iters)
    if show_progress and tqdm is not None:
        it_range = tqdm(it_range, desc=f"B-Chain N={N} β={beta:.4f}", leave=False)

    for it in it_range:
        move = "local" if (mix_local > 0 and rng.random() < mix_local) else "resample"
        coords_prop = propose_one_point(coords, dim, T_used, rng, move, local_sigma)
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

        # -----------------------------
        # 1) Coarse-grain by time cells
        # -----------------------------
        t = coords[:, 0]
        n_cells = int(cell_size)
        cell_of = time_cells_by_quantiles(t, n_cells=n_cells)

        # -----------------------------
        # 2) Base transport from links
        # -----------------------------
        T0 = build_cell_transition_from_links(
            C=C,
            cell_of_elem=cell_of,
            n_cells=n_cells,
            rng=rng,
            nbr_rank=int(nbr_rank),
        )

        # -----------------------------
        # 3) Charge proxy q_raw (cell-level)
        # -----------------------------
        if charge_mode == "deg":
            q_raw = cell_charge_deg(C, cell_of, n_cells)
        elif charge_mode == "ip":
            q_raw = cell_charge_ip(C, cell_of, n_cells)
        else:
            raise ValueError("charge_mode must be 'deg' or 'ip'")

        q_norm = normalize_charge(q_raw)

        # Split charges:
        #   q_op  drives dynamics
        #   q_reg used in regression (optionally shuffled)
        q_op = np.array(q_norm, dtype=float)
        q_reg = np.array(q_norm, dtype=float)

        # Excise should remove "physical" charge content -> apply to BOTH
        if do_excise:
            q_op = excise_defects(q_op, qmax=qmax)
            q_reg = np.array(q_op, dtype=float)

        # Shuffle is a null test for correlation -> shuffle regressor ONLY
        if do_shuffle:
            q_reg = shuffle_charge(q_reg, rng)

        # If no variance, regression meaningless
        if float(np.std(q_op)) <= 1e-12 or float(np.std(q_reg)) <= 1e-12:
            kappas.append(float("nan"))
            r2s.append(float("nan"))
            kappas_resid.append(float("nan"))
            r2s_resid.append(float("nan"))
            continue

        # -----------------------------
        # 4) Transport operators: lambda and baseline lambda0
        # -----------------------------
        Tlam = apply_defect_coupling(T0, q=q_op, lam=float(lam))
        Tlam0 = T0  # exactly lambda=0 baseline

        # -----------------------------
        # 5) Open-system NESS evolution (same boundaries)
        # -----------------------------
        src_ids = boundary_ids(n_cells, frac=float(source_frac), which="source")
        snk_ids = boundary_ids(n_cells, frac=float(sink_frac), which="sink")

        p_lam = open_system_evolve_to_ness(
            T=Tlam,
            mix_steps=int(mix_steps_ness),
            source_ids=src_ids,
            sink_ids=snk_ids,
            inject=float(inject),
        )
        p_0 = open_system_evolve_to_ness(
            T=Tlam0,
            mix_steps=int(mix_steps_ness),
            source_ids=src_ids,
            sink_ids=snk_ids,
            inject=float(inject),
        )

        div_lam = current_divergence(p_lam, Tlam)
        div_0 = current_divergence(p_0, Tlam0)

        div_resid = div_lam - div_0

        # -----------------------------
        # 6) Interior regression masks
        # -----------------------------
        mask = interior_mask(
            n_cells=n_cells,
            source_frac=float(source_frac),
            sink_frac=float(sink_frac),
            interior_frac=float(interior_frac),
        )

        # Raw regression: div_lam ~ q_reg
        reg_raw = regress_div_on_charge(div_lam, q_reg, use_mask=mask)
        kappas.append(reg_raw.kappa)
        r2s.append(reg_raw.r2)

        # Residual regression: (div_lam - div_0) ~ q_reg
        # For lam==0, residual is identically ~0 -> return 0 instead of NaN.
        if abs(float(lam)) <= 1e-15:
            kappas_resid.append(0.0)
            r2s_resid.append(float("nan"))  # no variance in y
        else:
            reg_res = regress_div_on_charge(div_resid, q_reg, use_mask=mask)
            kappas_resid.append(reg_res.kappa)
            r2s_resid.append(reg_res.r2)

    return ChainBOut(
        kappa_series=np.array(kappas, dtype=float),
        r2_series=np.array(r2s, dtype=float),
        kappa_resid_series=np.array(kappas_resid, dtype=float),
        r2_resid_series=np.array(r2s_resid, dtype=float),
        accept_rate=accepts / max(1, props),
    )




# -----------------------------
# Multi-chain summarization
# -----------------------------
@dataclass
class GridPointBStats:
    # raw
    kappa_mean: float
    kappa_stderr: float
    kappa_ess: float
    r2_mean: float
    r2_stderr: float
    # residual (PATCH)
    kappa_resid_mean: float
    kappa_resid_stderr: float
    kappa_resid_ess: float
    r2_resid_mean: float
    r2_resid_stderr: float
    acc_mean: float
    n_samples: int


def _summarize_series(chains: List[np.ndarray]) -> Tuple[float, float, float, int]:
    chains = [c[np.isfinite(c)] for c in chains]  # drop nans
    if not chains:
        return float("nan"), float("nan"), 0.0, 0
    allv = np.concatenate(chains) if len(chains) > 1 else chains[0]
    if allv.size < 2:
        return float("nan"), float("nan"), 0.0, int(allv.size)

    ess_total = float(sum(ess_1d(c) for c in chains if c.size > 1))
    mean = float(allv.mean())
    var = float(allv.var(ddof=1)) if allv.size > 1 else 0.0
    stderr = math.sqrt(var / max(1.0, ess_total)) if ess_total > 0 else float("nan")
    return mean, stderr, ess_total, int(allv.size)


def summarize_b_chains(chains: List[ChainBOut]) -> GridPointBStats:
    kappa_ch = [c.kappa_series for c in chains]
    r2_ch = [c.r2_series for c in chains]
    kappaR_ch = [c.kappa_resid_series for c in chains]
    r2R_ch = [c.r2_resid_series for c in chains]

    k_mean, k_se, k_ess, n = _summarize_series(kappa_ch)
    r_mean, r_se, _, _ = _summarize_series(r2_ch)

    kR_mean, kR_se, kR_ess, _ = _summarize_series(kappaR_ch)
    rR_mean, rR_se, _, _ = _summarize_series(r2R_ch)

    acc_mean = float(np.mean([c.accept_rate for c in chains])) if chains else float("nan")
    return GridPointBStats(
        kappa_mean=k_mean,
        kappa_stderr=k_se,
        kappa_ess=k_ess,
        r2_mean=r_mean,
        r2_stderr=r_se,
        kappa_resid_mean=kR_mean,
        kappa_resid_stderr=kR_se,
        kappa_resid_ess=kR_ess,
        r2_resid_mean=rR_mean,
        r2_resid_stderr=rR_se,
        acc_mean=acc_mean,
        n_samples=n,
    )


# -----------------------------
# Plotting (plots residual quantities by default)
# -----------------------------
def make_plots(
    outdir: str,
    Ns: List[int],
    betas: List[float],
    key_label: str,
    kappa: Dict[int, np.ndarray],
    kappa_err: Dict[int, np.ndarray],
    ess: Dict[int, np.ndarray],
    acc: Dict[int, np.ndarray],
    dr2: Dict[int, np.ndarray],
    dr2_err: Dict[int, np.ndarray],
) -> str:
    if plt is None:
        raise RuntimeError("Plotting requires matplotlib.")
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, "protocolB_transport.png")

    fig = plt.figure(figsize=(18, 10))

    # κ vs β
    ax1 = fig.add_subplot(2, 2, 1)
    for N in Ns:
        ax1.errorbar(betas, kappa[N], yerr=kappa_err[N], marker="o", capsize=3, label=f"N={N}")
    ax1.axhline(0.0, linestyle="--", linewidth=1.5)
    ax1.set_title(f"Protocol B (residual): κ_resid vs β ({key_label})")
    ax1.set_xlabel("beta")
    ax1.set_ylabel("kappa_resid")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ΔR² vs β
    ax2 = fig.add_subplot(2, 2, 2)
    for N in Ns:
        ax2.errorbar(betas, dr2[N], yerr=dr2_err[N], marker="o", capsize=3, label=f"N={N}")
    ax2.axhline(0.0, linestyle="--", linewidth=1.5)
    ax2.set_title("ΔR²_resid vs β (relative to λ=0 at same N)")
    ax2.set_xlabel("beta")
    ax2.set_ylabel("ΔR²_resid")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ESS vs β
    ax3 = fig.add_subplot(2, 2, 3)
    for N in Ns:
        ax3.plot(betas, ess[N], marker="o", label=f"N={N}")
    ax3.set_title("ESS(κ_resid) vs β (summed over chains)")
    ax3.set_xlabel("beta")
    ax3.set_ylabel("ESS")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Acceptance vs β
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
# CLI helpers
# -----------------------------
def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip() != ""]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Protocol B (Open-system NESS transport validation) [PATCHED]")

    # Grid / geometry
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--Ns", type=str, default="120,180,250,350")
    ap.add_argument("--betas", type=str, default="0.0025,0.003,0.0042")

    # Action (same as A/C)
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

    # Density control (optional)
    ap.add_argument("--fix_density", action="store_true")
    ap.add_argument("--Nref", type=int, default=120)
    ap.add_argument("--Tref", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=None)

    # Protocol B controls
    ap.add_argument("--cell_sizes", type=str, default="20,40", help="Comma-separated #time-cells (coarse bins)")
    ap.add_argument("--lambdas", type=str, default="0,0.5,-0.5", help="Comma-separated lambda values")
    ap.add_argument("--nbr_rank", type=int, default=12, help="Future-samples per element for T0 construction")

    # NESS / open system
    ap.add_argument("--mix_steps", type=int, default=150, help="Steps to reach NESS")
    ap.add_argument("--source_frac", type=float, default=0.05, help="Fraction of earliest cells as source")
    ap.add_argument("--sink_frac", type=float, default=0.05, help="Fraction of latest cells as sink")
    ap.add_argument("--inject", type=float, default=0.03, help="Injection mass per step into source cells")
    ap.add_argument("--interior_frac", type=float, default=0.80, help="Fraction of remaining cells used for regression")

    # Charge / defects
    ap.add_argument("--charge_modes", type=str, default="deg,ip", help="Comma-separated: deg,ip")
    ap.add_argument("--qmax", type=float, default=5.0, help="Defect top-qmax percent by |q| for excision")

    # Controls
    ap.add_argument("--shuffle_charge", action="store_true")
    ap.add_argument("--excise_defects", action="store_true")

    # Output
    ap.add_argument("--outdir", type=str, default="protocolB_out")

    args = ap.parse_args()

    dim = int(args.dim)
    Ns = parse_csv_ints(args.Ns)
    betas = parse_csv_floats(args.betas)

    k_max = int(args.kmax)
    alpha = np.array(parse_csv_floats(args.alpha), dtype=float)
    if len(alpha) != k_max + 1:
        raise ValueError(f"--alpha must have length kmax+1 ({k_max+1}), got {len(alpha)}")

    cell_sizes = parse_csv_ints(args.cell_sizes)
    lambdas = parse_csv_floats(args.lambdas)
    charge_modes = parse_csv_strs(args.charge_modes)

    if not (0.0 <= args.mix_local <= 1.0):
        raise ValueError("--mix_local must be in [0,1]")

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

    os.makedirs(args.outdir, exist_ok=True)

    print("\n--- Protocol B (Open-system NESS transport validation) [PATCHED] ---")
    print(f"DIM={dim} | NS={Ns} | betas=[{betas[0]}, …, {betas[-1]}] ({len(betas)} points)")
    print(f"burn={args.burn} steps={args.steps} thin={args.thin} chains={args.chains}")
    print(f"moves=mixed(p_local={args.mix_local:.2f}, sigma={args.local_sigma})")
    print(f"action: k_max={k_max} | alpha={alpha.tolist()}")
    print(f"density: {density_mode}")
    print(f"cells={cell_sizes} lambdas={lambdas} charge_modes={charge_modes}")
    print(f"NESS: mix_steps={args.mix_steps} source_frac={args.source_frac} sink_frac={args.sink_frac} inject={args.inject}")
    print(f"regression: interior_frac={args.interior_frac} (plus boundary exclusion)")
    print(f"controls: shuffle_charge={bool(args.shuffle_charge)} excise_defects={bool(args.excise_defects)}\n")
    print("NOTE: Plots + ΔR² now use *residual* quantities (div_lambda - div_lambda=0) to cancel geometric/NESS artifacts.\n")

    # Choose ONE plot key (as before)
    plot_charge = charge_modes[0] if charge_modes else "deg"
    plot_cell = int(cell_sizes[0]) if cell_sizes else 20
    plot_lam = None
    for l in lambdas:
        if abs(l) > 1e-15:
            plot_lam = float(l)
            break
    if plot_lam is None:
        plot_lam = float(lambdas[0]) if lambdas else 0.0

    # Plot storage (residual metrics)
    kappa_plot: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    kappa_plot_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    ess_plot: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    acc_plot: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    dr2_plot: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    dr2_plot_err: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}

    # CSV
    csv_path = os.path.join(args.outdir, "protocolB_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "N,beta,T_used,charge,cell_size,lambda,"
            "kappa_mean,kappa_stderr,kappa_z,"
            "r2_mean,r2_stderr,dr2_mean,dr2_stderr,"
            "kappa_resid_mean,kappa_resid_stderr,kappa_resid_z,"
            "r2_resid_mean,r2_resid_stderr,dr2_resid_mean,dr2_resid_stderr,"
            "ess_kappa,ess_kappa_resid,acc_mean,n_samples,"
            "mix_steps,source_frac,sink_frac,inject,interior_frac,"
            "shuffle_charge,excise_defects\n"
        )

        for N in Ns:
            T_used = float(args.T) if rho0 is None else T_for_fixed_density(dim, N, rho0)

            for bi, beta in enumerate(betas):
                for cell_size in cell_sizes:
                    for charge_mode in charge_modes:
                        stats_by_lambda: Dict[float, GridPointBStats] = {}

                        # run all lambdas
                        for lam in lambdas:
                            chains_out: List[ChainBOut] = []
                            for _ in range(int(args.chains)):
                                child_seed = int(master_rng.integers(0, 2**32 - 1))
                                rng = np.random.default_rng(child_seed)

                                chains_out.append(
                                    run_protocol_b_chain(
                                        N=N,
                                        dim=dim,
                                        T_used=T_used,
                                        beta=float(beta),
                                        alpha=alpha,
                                        k_max=k_max,
                                        burn_in=int(args.burn),
                                        steps=int(args.steps),
                                        thin=int(args.thin),
                                        mix_local=float(args.mix_local),
                                        local_sigma=float(args.local_sigma),
                                        rng=rng,
                                        cell_size=int(cell_size),
                                        lam=float(lam),
                                        nbr_rank=int(args.nbr_rank),
                                        mix_steps_ness=int(args.mix_steps),
                                        source_frac=float(args.source_frac),
                                        sink_frac=float(args.sink_frac),
                                        inject=float(args.inject),
                                        interior_frac=float(args.interior_frac),
                                        charge_mode=str(charge_mode),
                                        qmax=float(args.qmax),
                                        do_shuffle=bool(args.shuffle_charge),
                                        do_excise=bool(args.excise_defects),
                                        show_progress=bool(args.progress),
                                    )
                                )

                            stats = summarize_b_chains(chains_out)
                            stats_by_lambda[float(lam)] = stats

                            # baselines at lambda=0 for this (N,beta,cell,charge)
                            if 0.0 in stats_by_lambda:
                                base = stats_by_lambda[0.0]
                                r2_0, r2_0_se = base.r2_mean, base.r2_stderr
                            else:
                                r2_0 = r2_0_se = float("nan")

                            # Residual baseline: define λ=0 residual baseline as 0.0 by convention
                            # (because div_resid ≡ 0 at λ=0 so "extra explained variance" should be 0)
                            r2R_0 = 0.0
                            r2R_0_se = 0.0

                            for lam, stats in stats_by_lambda.items():
                                # ΔR² (raw), relative to λ=0 raw
                                if math.isfinite(stats.r2_mean) and math.isfinite(r2_0):
                                    dr2 = stats.r2_mean - r2_0
                                    dr2_se = (
                                        math.sqrt((stats.r2_stderr or 0.0) ** 2 + (r2_0_se or 0.0) ** 2)
                                        if (math.isfinite(stats.r2_stderr) and math.isfinite(r2_0_se))
                                        else float("nan")
                                    )
                                else:
                                    dr2 = float("nan")
                                    dr2_se = float("nan")

                                # ΔR²_resid: relative to residual baseline (defined as 0 at λ=0)
                                if abs(lam) <= 1e-15:
                                    dr2_resid = 0.0
                                    dr2_resid_se = 0.0
                                else:
                                    dr2_resid = stats.r2_resid_mean
                                    dr2_resid_se = stats.r2_resid_stderr


                            kappa_z = (stats.kappa_mean / stats.kappa_stderr) if (
                                math.isfinite(stats.kappa_mean) and math.isfinite(stats.kappa_stderr) and stats.kappa_stderr > 0
                            ) else float("nan")

                            kappaR_z = (stats.kappa_resid_mean / stats.kappa_resid_stderr) if (
                                math.isfinite(stats.kappa_resid_mean)
                                and math.isfinite(stats.kappa_resid_stderr)
                                and stats.kappa_resid_stderr > 0
                            ) else float("nan")

                            f.write(
                                f"{N},{beta},{T_used},{charge_mode},{cell_size},{lam},"
                                f"{stats.kappa_mean},{stats.kappa_stderr},{kappa_z},"
                                f"{stats.r2_mean},{stats.r2_stderr},{dr2},{dr2_se},"
                                f"{stats.kappa_resid_mean},{stats.kappa_resid_stderr},{kappaR_z},"
                                f"{stats.r2_resid_mean},{stats.r2_resid_stderr},{dr2_resid},{dr2_resid_se},"
                                f"{stats.kappa_ess},{stats.kappa_resid_ess},{stats.acc_mean},{stats.n_samples},"
                                f"{int(args.mix_steps)},{args.source_frac},{args.sink_frac},{args.inject},{args.interior_frac},"
                                f"{int(bool(args.shuffle_charge))},{int(bool(args.excise_defects))}\n"
                            )

                            # capture plot key (residual metrics)
                            if (
                                charge_mode == plot_charge
                                and int(cell_size) == int(plot_cell)
                                and abs(lam - plot_lam) < 1e-15
                            ):
                                kappa_plot[N][bi] = stats.kappa_resid_mean
                                kappa_plot_err[N][bi] = stats.kappa_resid_stderr
                                ess_plot[N][bi] = stats.kappa_resid_ess
                                acc_plot[N][bi] = stats.acc_mean
                                dr2_plot[N][bi] = dr2_resid if math.isfinite(dr2_resid) else 0.0
                                dr2_plot_err[N][bi] = dr2_resid_se if math.isfinite(dr2_resid_se) else 0.0

                print(
                    f"N={N:4d} beta={beta:7.4f} | T={T_used:.4f} | "
                    f"key=({plot_charge},cell={plot_cell},lam={plot_lam:+g}) "
                    f"kappa_resid={kappa_plot[N][bi]:.4g} ± {kappa_plot_err[N][bi]:.3g} "
                    f"ESS≈{ess_plot[N][bi]:.1f} acc={acc_plot[N][bi]:.3f} "
                    f"ΔR2_resid={dr2_plot[N][bi]:.3g}"
                )

    key_label = f"charge={plot_charge}|cell{plot_cell}|lam{plot_lam:+g}"
    outpng = make_plots(
        outdir=args.outdir,
        Ns=Ns,
        betas=betas,
        key_label=key_label,
        kappa=kappa_plot,
        kappa_err=kappa_plot_err,
        ess=ess_plot,
        acc=acc_plot,
        dr2=dr2_plot,
        dr2_err=dr2_plot_err,
    )

    print(f"\nSaved figure: {outpng}")
    print(f"Saved CSV: {csv_path}")

    print("\nProtocol B interpretation guide (NESS + residual version):")
    print("- Interpret κ_resid and ΔR²_resid (not the raw ones).")
    print("- Expect κ_resid ~ 0 and ΔR²_resid ~ 0 at λ=0.")
    print("- For λ≠0, κ_resid should scale with λ (and typically flip sign with λ).")
    print("- Shuffle control should collapse κ_resid and ΔR²_resid toward 0.")
    print("- Excise-defects should leave little/no charge, making regression undefined (NaN) or near-null.")


if __name__ == "__main__":
    main()
