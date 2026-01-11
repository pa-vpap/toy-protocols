
# -*- coding: utf-8 -*-
"""
Protocol A — Coordinate MCMC over sprinklings (updated, density-consistent)

Key upgrades vs earlier versions:
  1) Optional fixed-density finite-size scan:
     - If --fix_density is set (or --rho given), the Alexandrov interval height T
       is scaled with N so that the sprinkling density rho is constant across N.
     - This makes "<d_MM>(N) plateau" claims meaningful at fixed discreteness scale.

  2) Mixed proposal kernel with correct logging:
     - Use --mix_local in [0,1] to mix local Gaussian moves (better basin mixing)
       and resample moves (global refresh).
     - The header prints the actual move mix.

  3) Multi-chain statistics + ESS estimate:
     - Runs `--chains` independent chains per (N,beta).
     - Reports mean ± stderr (using ESS-adjusted stderr from pooled samples),
       plus ESS and acceptance.

  4) Same CST conventions as before:
     - Sprinkling in a D-dimensional Alexandrov interval (causal diamond).
     - Causality from Minkowski metric.
     - Interval-abundance action: S = Σ_k alpha_k N_k, with k = open interval size.

Usage examples:
  # Quick sanity (beta=0 should give d≈DIM)
  python a_gpt.py --betas 0.0 --Ns 120 --chains 5

  # Your current local kmax=3 selector
  python a_gpt.py --kmax 3 --alpha "1,-9,16,-8" \
    --betas 0.003,0.005,0.007,0.01 --Ns 120,180,250,350 --chains 5 --mix_local 0.5

  # Density-consistent scan: keep rho fixed using a reference (Nref,Tref)
  python a_gpt.py --fix_density --Nref 120 --Tref 1.0 \
    --kmax 3 --alpha "1,-9,16,-8" \
    --betas 0.003,0.005,0.007 --Ns 120,180,250,350 --chains 5

  # Density-consistent scan: specify rho directly
  python a_gpt.py --rho 120.0 \
    --kmax 3 --alpha "1,-9,16,-8" \
    --betas 0.003,0.005,0.007 --Ns 120,180,250,350 --chains 5
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
# Myrheim–Meyer dimension from ordering fraction
# -----------------------------
def _f0_ordering_fraction(d: float) -> float:
    # f0(d) = Γ(d+1) Γ(d/2) / (4 Γ(3d/2))
    return math.exp(
        math.lgamma(d + 1.0)
        + math.lgamma(d / 2.0)
        - math.lgamma(1.5 * d)
        - math.log(4.0)
    )


def _r_of_d(d: float) -> float:
    # Myrheim ordering fraction r(d) = 2 f0(d)
    return 2.0 * _f0_ordering_fraction(d)


def invert_ordering_fraction_to_dimension(
    r_hat: float,
    d_min: float = 1.01,
    d_max: float = 30.0,
    tol: float = 1e-7,
    max_iter: int = 120,
) -> float:
    """
    Invert r(d) = r_hat via bisection.
    r(d) is monotone decreasing in d.
    """
    r_hat = float(r_hat)
    r_hat = min(max(r_hat, 1e-12), 1.0 - 1e-12)

    lo, hi = d_min, d_max
    r_lo, r_hi = _r_of_d(lo), _r_of_d(hi)

    if r_hat > r_lo:
        return lo
    if r_hat < r_hi:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = _r_of_d(mid)
        if abs(r_mid - r_hat) < tol:
            return mid
        if r_mid > r_hat:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -----------------------------
# Causal set bitset representation
# -----------------------------
@dataclass
class CausalSet:
    N: int
    reach: List[int]  # reach[i] bitset of all j with i ≺ j
    past: List[int]   # past[j] bitset of all i with i ≺ j

    def comparable_pairs_count(self) -> int:
        return sum(bits.bit_count() for bits in self.reach)

    def ordering_fraction(self) -> float:
        if self.N < 2:
            return 0.0
        R = self.comparable_pairs_count()
        return 2.0 * R / (self.N * (self.N - 1))

    def myrheim_meyer_dimension(self) -> float:
        return invert_ordering_fraction_to_dimension(self.ordering_fraction())


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
# Alexandrov interval (causal diamond) volume + density scaling
# -----------------------------
def sphere_area(n: int) -> float:
    # Area of unit n-sphere S^n: 2*pi^{(n+1)/2}/Gamma((n+1)/2)
    return 2.0 * math.pi ** ((n + 1) / 2.0) / math.gamma((n + 1) / 2.0)


def diamond_volume(dim: int, T: float) -> float:
    """
    Volume of a D-dimensional Alexandrov interval (causal diamond) of proper time T:
      vol = 2 * S_{d-2} / (d(d-1)) * (T/2)^d
    """
    Sd_2 = sphere_area(dim - 2)
    return 2.0 * Sd_2 / (dim * (dim - 1)) * (0.5 * T) ** dim


def T_for_fixed_density(dim: int, N: int, rho: float) -> float:
    """
    Solve N = rho * vol_d(T) for T, given vol_d(T) above.
    """
    Sd_2 = sphere_area(dim - 2)
    return 2.0 * ((N * dim * (dim - 1)) / (2.0 * rho * Sd_2)) ** (1.0 / dim)


# -----------------------------
# Alexandrov interval sprinkling
# -----------------------------
def inside_alexandrov(pt: np.ndarray, dim: int, T: float) -> bool:
    """
    Diamond defined by p=(-T/2, 0..0), q=(+T/2, 0..0).
    Condition: p ≺ x ≺ q  <=>  (t+T/2)^2 > r^2 AND (T/2 - t)^2 > r^2.
    """
    half = 0.5 * T
    t = float(pt[0])
    r2 = float(np.sum(pt[1:] ** 2))
    return ((t + half) ** 2 > r2) and ((half - t) ** 2 > r2) and (-half < t < half)


def sample_point_in_diamond(dim: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """
    Rejection sampling from bounding box:
      t ∈ [-T/2, T/2], x_i ∈ [-T/2, T/2]
    """
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
    # Sort by time coordinate (speeds causality build; labels are gauge)
    pts = pts[np.argsort(pts[:, 0])]
    return pts


# -----------------------------
# Build causal relations from coordinates (Minkowski)
# -----------------------------
def causalset_from_coords(coords: np.ndarray) -> CausalSet:
    """
    Minkowski causality:
      i ≺ j iff t_j > t_i and (Δt)^2 > ||Δx||^2
    coords must be sorted by time.
    """
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
# BD-style action via interval abundances (open interval size k)
# -----------------------------
def bd_action_interval_abundances(C: CausalSet, alpha: np.ndarray, k_max: int) -> float:
    """
    S = Σ_k α_k N_k, where N_k = Σ_x n_k(x),
    n_k(x) counts y≺x with open interval size |(y,x)| = k.

    |(y,x)| = popcount( reach[y] & past[x] ).
    """
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
# Coordinate proposals (Metropolis moves) + mixed kernel
# -----------------------------
def propose_one_point(
    coords: np.ndarray,
    dim: int,
    T: float,
    rng: np.random.Generator,
    move: str,
    local_sigma: float,
) -> np.ndarray:
    """
    Propose new coordinate set by modifying one randomly chosen point.
    Then re-sort by time coordinate.
    """
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


def propose_coords(
    coords: np.ndarray,
    dim: int,
    T: float,
    rng: np.random.Generator,
    proposal: str,
    local_sigma: float,
) -> np.ndarray:
    """Compatibility wrapper used by the test suite.

    The implementation's native function is `propose_one_point(..., move=...)`.
    """
    return propose_one_point(
        coords=coords,
        dim=dim,
        T=T,
        rng=rng,
        move=proposal,
        local_sigma=local_sigma,
    )


# -----------------------------
# ESS (effective sample size) estimation for 1D series
# -----------------------------
def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """
    Fast autocorrelation using FFT. Returns acf[0..n-1] normalized with acf[0]=1.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.ones(n)
    x = x - x.mean()
    # next power of 2 for speed
    m = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=m)[:n]
    ac /= ac[0] if ac[0] != 0 else 1.0
    return ac


def ess_1d(x: np.ndarray) -> float:
    """
    Conservative ESS estimate:
      ESS = n / tau_int, tau_int = 1 + 2*sum_{t>=1} rho_t
    Truncate sum at first negative rho (common heuristic).
    """
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
# Coordinate MCMC core (single chain)
# -----------------------------
@dataclass
class ChainOut:
    samples: np.ndarray
    accept_rate: float


@dataclass
class MCMCOut:
    """Compatibility return type for the test suite."""

    accept_rate: float
    num_samples: int
    mean_dmm: float
    var_dmm: float
    samples: np.ndarray


def run_coordinate_chain(
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
    show_progress: bool = False,
) -> ChainOut:
    coords = sprinkle_diamond(N, dim, T, rng)
    C = causalset_from_coords(coords)
    S = bd_action_interval_abundances(C, alpha, k_max)

    accepts = 0
    props = 0
    dmm_samples: List[float] = []

    total_iters = burn_in + steps
    it_range = range(total_iters)
    if show_progress and tqdm is not None:
        it_range = tqdm(it_range, desc=f"Chain N={N} β={beta:.4f}", leave=False)

    for it in it_range:
        # mixed kernel
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
            dmm_samples.append(C.myrheim_meyer_dimension())

    return ChainOut(samples=np.array(dmm_samples, dtype=float), accept_rate=accepts / max(1, props))


def run_coordinate_mcmc(
    N: int,
    dim: int,
    T: float,
    beta: float,
    alpha: np.ndarray,
    k_max: int,
    burn_in: int,
    steps: int,
    thin: int,
    proposal: str,
    local_sigma: float,
    rng: np.random.Generator,
    progress: bool = False,
) -> MCMCOut:
    """Compatibility wrapper used by the test suite.

    Maps the test suite's `proposal` to this implementation's `mix_local`:
    - 'resample' -> mix_local=0.0
    - 'local'    -> mix_local=1.0
    """

    if proposal == "resample":
        mix_local = 0.0
    elif proposal == "local":
        mix_local = 1.0
    else:
        raise ValueError("proposal must be 'resample' or 'local'")

    chain = run_coordinate_chain(
        N=N,
        dim=dim,
        T=T,
        beta=beta,
        alpha=alpha,
        k_max=k_max,
        burn_in=burn_in,
        steps=steps,
        thin=thin,
        mix_local=mix_local,
        local_sigma=local_sigma,
        rng=rng,
        show_progress=progress,
    )

    samples = np.asarray(chain.samples, dtype=float)
    mean = float(samples.mean()) if samples.size else float("nan")
    var = float(samples.var()) if samples.size else float("nan")
    return MCMCOut(
        accept_rate=float(chain.accept_rate),
        num_samples=int(samples.size),
        mean_dmm=mean,
        var_dmm=var,
        samples=samples,
    )


# -----------------------------
# Multi-chain aggregation
# -----------------------------
@dataclass
class GridPointStats:
    mean: float
    stderr: float
    ess: float
    acc: float
    n_samples: int


def summarize_chains(chains: List[ChainOut]) -> GridPointStats:
    all_samples = np.concatenate([c.samples for c in chains]) if chains else np.array([], dtype=float)
    if len(all_samples) == 0:
        return GridPointStats(mean=float("nan"), stderr=float("nan"), ess=0.0, acc=float("nan"), n_samples=0)

    # ESS: sum ESS per chain (reasonable if chains independent)
    ess_total = float(sum(ess_1d(c.samples) for c in chains))
    mean = float(all_samples.mean())

    # estimate variance of the underlying draws using sample variance, then stderr via ESS
    var = float(all_samples.var(ddof=1)) if len(all_samples) > 1 else 0.0
    stderr = math.sqrt(var / max(1.0, ess_total))

    acc = float(np.mean([c.accept_rate for c in chains]))
    return GridPointStats(mean=mean, stderr=stderr, ess=ess_total, acc=acc, n_samples=int(len(all_samples)))


# -----------------------------
# Plateau scoring (heuristic but explicit)
# -----------------------------
def plateau_score(means_by_N: Dict[int, float], target_dim: float) -> float:
    Ns = sorted(means_by_N.keys())
    ys = np.array([means_by_N[N] for N in Ns], dtype=float)
    dev = float(np.mean(np.abs(ys - target_dim)))
    x = np.array(Ns, dtype=float)
    b = float(np.polyfit(x, ys, 1)[0]) if len(Ns) >= 2 else 0.0
    slope = abs(b)

    dev_scale = 0.25
    slope_scale = 0.002
    return float(math.exp(-(dev / dev_scale)) * math.exp(-(slope / slope_scale)))


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# -----------------------------
# Plotting
# -----------------------------
def make_plots(
    outdir: str,
    dim: int,
    Ns: List[int],
    betas: List[float],
    mean_d: Dict[int, np.ndarray],
    err_d: Dict[int, np.ndarray],
    acc: Dict[int, np.ndarray],
    ess: Dict[int, np.ndarray],
) -> str:
    if plt is None:
        raise RuntimeError("Plotting requires matplotlib. Install it or run with plotting disabled.")
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, "protocolA_coordinate_mcmc.png")

    fig = plt.figure(figsize=(16, 9))

    # <dMM> vs beta (per N)
    ax1 = fig.add_subplot(2, 2, 1)
    for N in Ns:
        ax1.errorbar(betas, mean_d[N], yerr=err_d[N], marker="o", capsize=3, label=f"N={N}")
    ax1.axhline(float(dim), linestyle="--", linewidth=1.5)
    ax1.set_title("Coordinate MCMC: <d_MM> vs beta")
    ax1.set_xlabel("beta")
    ax1.set_ylabel("<d_MM>")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ESS vs beta
    ax2 = fig.add_subplot(2, 2, 2)
    for N in Ns:
        ax2.plot(betas, ess[N], marker="o", label=f"N={N}")
    ax2.set_title("Effective sample size (ESS) vs beta")
    ax2.set_xlabel("beta")
    ax2.set_ylabel("ESS")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Finite-size scan at each beta
    ax3 = fig.add_subplot(2, 2, 3)
    for bi, beta in enumerate(betas):
        ys = [mean_d[N][bi] for N in Ns]
        ax3.plot(Ns, ys, marker="o", label=f"β={beta:.3f}")
    ax3.axhline(float(dim), linestyle="--", linewidth=1.5)
    ax3.set_title("Finite-size scan: <d_MM>(N) for each beta")
    ax3.set_xlabel("N")
    ax3.set_ylabel("<d_MM>")
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=2, fontsize=8)

    # Acceptance rate vs beta
    ax4 = fig.add_subplot(2, 2, 4)
    for N in Ns:
        ax4.plot(betas, acc[N], marker="o", label=f"N={N}")
    ax4.set_title("Acceptance rate vs beta")
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
def main():
    ap = argparse.ArgumentParser(description="Protocol A (Coordinate MCMC over sprinklings) — updated")
    ap.add_argument("--dim", type=int, default=4, help="Target Minkowski dimension (default 4)")
    ap.add_argument("--T", type=float, default=1.0, help="Diamond height T (used unless density is fixed)")
    ap.add_argument("--Ns", type=str, default="120,180,250,350", help="Comma-separated N list")
    ap.add_argument("--betas", type=str, default="0,0.003,0.005,0.007,0.01",
                    help="Comma-separated beta list")
    ap.add_argument("--kmax", type=int, default=3, help="Interval cutoff k_max (default 3)")
    ap.add_argument("--alpha", type=str, default="1,-9,16,-8",
                    help="Comma-separated alpha_k list length kmax+1")
    ap.add_argument("--burn", type=int, default=1000, help="MCMC burn-in steps")
    ap.add_argument("--steps", type=int, default=16000, help="MCMC sampling steps after burn-in")
    ap.add_argument("--thin", type=int, default=40, help="Record every thin steps")
    ap.add_argument("--chains", type=int, default=5, help="Independent chains per (N,beta)")
    ap.add_argument("--mix_local", type=float, default=0.5,
                    help="Probability of proposing a local Gaussian move (else resample).")
    ap.add_argument("--local_sigma", type=float, default=0.03, help="Sigma for local Gaussian proposal")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed")
    ap.add_argument("--outdir", type=str, default="protocolA_out", help="Output directory for plots")
    ap.add_argument("--progress", action="store_true", help="Show per-chain progress bars")

    # Density control
    ap.add_argument("--fix_density", action="store_true",
                    help="Keep sprinkling density fixed across N by scaling T(N) using Nref,Tref.")
    ap.add_argument("--Nref", type=int, default=120, help="Reference N for density calibration")
    ap.add_argument("--Tref", type=float, default=1.0, help="Reference T for density calibration")
    ap.add_argument("--rho", type=float, default=None,
                    help="If set, keep density fixed at rho across N by scaling T(N). Overrides --fix_density.")

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

    print("\n--- Protocol A (Coordinate MCMC over sprinklings) ---")
    print(f"DIM={dim} | NS={Ns} | betas=[{betas[0]}, …, {betas[-1]}] ({len(betas)} points)")
    print(f"burn={args.burn} steps={args.steps} thin={args.thin} chains={args.chains}")
    print(f"moves=mixed(p_local={args.mix_local:.2f}, sigma={args.local_sigma})")
    print(f"k_max={k_max} | alpha={alpha.tolist()}")
    print(f"density: {density_mode}")
    print("NOTE: Replace alpha with your true BD/BDG coefficients (incl. smearing if used) for a meaningful physics test.\n")

    mean_d: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    err_d: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    ess_d: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}
    acc: Dict[int, np.ndarray] = {N: np.zeros(len(betas), dtype=float) for N in Ns}

    # Run grid
    for N in Ns:
        # choose T for this N
        if rho0 is None:
            T_used = float(args.T)
        else:
            T_used = T_for_fixed_density(dim, N, rho0)

        for bi, beta in enumerate(betas):
            # spawn independent chain rngs
            chains_out: List[ChainOut] = []
            for ci in range(int(args.chains)):
                # deterministic child seed stream
                child_seed = int(master_rng.integers(0, 2**32 - 1))
                rng = np.random.default_rng(child_seed)

                chains_out.append(
                    run_coordinate_chain(
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
                        show_progress=bool(args.progress),
                    )
                )

            stats = summarize_chains(chains_out)
            mean_d[N][bi] = stats.mean
            err_d[N][bi] = stats.stderr
            ess_d[N][bi] = stats.ess
            acc[N][bi] = stats.acc

            print(
                f"N={N:4d} beta={beta:7.4f} | "
                f"T={T_used:.4f} | "
                f"d={stats.mean:.3f} ± {stats.stderr:.3f} "
                f"ESS≈{stats.ess:.1f} acc={stats.acc:.3f}"
            )

    # Plateau scoring per beta (only meaningful if multiple Ns)
    if len(Ns) >= 2:
        print("\n--- Plateau score (closer to 1 is better) ---")
        for bi, beta in enumerate(betas):
            means_by_N = {N: float(mean_d[N][bi]) for N in Ns}
            score = plateau_score(means_by_N, target_dim=float(dim))
            dev = float(np.mean([abs(means_by_N[N] - dim) for N in Ns]))
            print(f"beta={beta:7.4f} | score={score:.4f} | mean|d-{dim}|={dev:.4f}")

    outpng = make_plots(args.outdir, dim, Ns, betas, mean_d, err_d, acc, ess_d)
    print(f"\nSaved figure: {outpng}")

    print("\nProtocol A interpretation guide:")
    print("- Sanity check: at beta=0, <d_MM> should be near DIM for all N.")
    print("- Protocol A success: exists beta* where <d_MM>(beta*,N) ~ 4 and becomes N-stable (plateau),")
    print("  with variance generally decreasing as N increases in that regime.")
    print("- If acceptance collapses (near 0) or ESS is tiny, increase steps/burn or adjust proposal mix.")
    print("- For finite-size claims, use --fix_density or --rho so discreteness scale stays fixed across N.")


if __name__ == "__main__":
    main()
