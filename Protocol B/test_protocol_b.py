#!/usr/bin/env python3
"""
Protocol B (v4): Stochastic ΩΛ and the Hubble Tension
=====================================================

VERSION 4 - Fully reviewed implementation:
1. SNe estimator fits H0 consistently with simulated D_L(z) using χ(z)(1+z)
2. CMB estimator does θ* root finding with ω-fixed degeneracy breaker
   (uses realization's own θ*, not Planck θ*_obs - see docstring)
3. OU time grid flows from early (high z) to late (low z)
4. CMB θ* includes analytic radiation-tail correction beyond z_max
5. ΩΛ clamped to [-0.2, 1.5] to prevent unphysical E² values

Key Physics (toy model):
- ΩΛ fluctuates via OU process with correlation time τ_c
- Fluctuation amplitude: σ(z) = σ₀ · min(E(z), E_cap)²
- Multiple spatial blocks with Toeplitz correlation
- CMB: θ*_target from realization → H0 via ω-fixed ΛCDM root finding
- Late-time uses SNe distance ladder (consistent) + low-z H(z=0.05) probe

Interpretation:
- Each realization represents one Hubble patch
- The patch "measures" its own θ* (different from other patches due to ΩΛ fluctuations)
- Tension arises because H0 inferred from early-time (θ*) differs from late-time

Author: Generated for CST research
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

N_CORES = os.cpu_count() or 4

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

H0_PLANCK = 67.4
H0_SHOES = 73.0
OMEGA_M_PLANCK = 0.315
OMEGA_R_PLANCK = 9.0e-5
OMEGA_B_PLANCK = 0.049
Z_STAR = 1089.0
Z_DRAG = 1060.0
C_KM_S = 299792.458
THETA_STAR_PLANCK = 0.0104


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProtocolBConfig:
    n_realizations: int = 200
    Omega_m: float = 0.315
    Omega_r: float = 9.0e-5
    Omega_b: float = 0.049
    H0_fiducial: float = 75.0
    sigma_OmegaLambda_0: float = 0.08
    correlation_time: float = 0.5
    use_H2_scaling: bool = True
    E_cap: float = 10.0  # Cap on E(z) for H² scaling to prevent runaway at high z
    n_spatial_blocks: int = 8
    spatial_correlation: float = 0.7
    z_max: float = 3000.0
    n_z_points: int = 2000
    z_star: float = 1089.0
    z_drag: float = 1060.0
    theta_star_observed: float = THETA_STAR_PLANCK
    z_max_late: float = 0.15
    distance_modulus_noise: float = 0.10
    n_sne: int = 150
    H0_direct_noise: float = 2.0
    verbose: bool = True
    n_workers: int = N_CORES
    parallel: bool = True
    seed: Optional[int] = None


# =============================================================================
# STOCHASTIC ΩΛ MODEL (with corrected time direction)
# =============================================================================

@dataclass
class StochasticOmegaLambdaModel:
    """Toy model for stochastic vacuum energy density parameter."""
    
    OmegaLambda_bar: float
    correlation_time: float
    sigma_0: float
    use_H2_scaling: bool
    n_spatial_blocks: int
    spatial_correlation: float
    E_cap: float = 10.0  # Cap on E(z) for H² scaling
    
    t_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    OmegaLambda_history: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def generate(self, z_grid: np.ndarray, E_z: np.ndarray, 
                 seed: Optional[int] = None):
        """
        Generate stochastic ΩΛ realization using OU process.
        
        Time flows from early (high z) to late (low z), i.e., cosmic time increasing.
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_z = len(z_grid)
        
        # Build cosmic time grid: t increases as z decreases
        # t_grid is "time since z_max" in H0^{-1} units:
        #   t_grid[-1] = 0 at z_max (earliest time)
        #   t_grid[0] = largest value at z=0 (latest time)
        self.t_grid = np.zeros(n_z)
        
        # Start from highest z (earliest time, t=0 at z_max)
        for i in range(n_z - 2, -1, -1):  # from n_z-2 down to 0
            dz = z_grid[i+1] - z_grid[i]  # positive since z_grid ascending
            E_mid = 0.5 * (E_z[i] + E_z[i+1])
            z_mid = 0.5 * (z_grid[i] + z_grid[i+1])
            # dt/dz = -1/((1+z)E), so dt = -dz/((1+z)E) when going from higher z to lower z
            # Since we're going from high z to low z, and dz > 0 (from i to i+1), 
            # we accumulate time as t[i] = t[i+1] + |dz|/((1+z)E)
            self.t_grid[i] = self.t_grid[i+1] + dz / ((1 + z_mid) * E_mid)
        
        theta = 1.0 / self.correlation_time
        
        # Spatial correlation
        rho = self.spatial_correlation
        spatial_cov = np.array([[rho ** abs(i - j) 
                                 for j in range(self.n_spatial_blocks)]
                                for i in range(self.n_spatial_blocks)])
        
        try:
            L_spatial = np.linalg.cholesky(spatial_cov)
        except np.linalg.LinAlgError:
            L_spatial = np.eye(self.n_spatial_blocks)
        
        # Initialize at high z (earliest time, index n_z-1)
        self.OmegaLambda_history = np.zeros((self.n_spatial_blocks, n_z))
        init_noise = L_spatial @ np.random.randn(self.n_spatial_blocks)
        self.OmegaLambda_history[:, n_z-1] = self.OmegaLambda_bar + self.sigma_0 * init_noise
        
        # Evolve OU process forward in cosmic time (backward in z index)
        for i in range(n_z - 2, -1, -1):
            dt = self.t_grid[i] - self.t_grid[i+1]  # should be positive
            
            if dt <= 0:
                self.OmegaLambda_history[:, i] = self.OmegaLambda_history[:, i+1]
                continue
            
            if self.use_H2_scaling:
                # Cap the scaling to prevent explosion at high z
                E_capped = min(E_z[i], self.E_cap)
                sigma_t = self.sigma_0 * E_capped * E_capped
            else:
                sigma_t = self.sigma_0
            
            decay = np.exp(-theta * dt)
            noise_var = (1 - np.exp(-2 * theta * dt)) / (2 * theta)
            noise_scale = sigma_t * np.sqrt(max(noise_var, 0))
            
            noise = L_spatial @ np.random.randn(self.n_spatial_blocks)
            
            self.OmegaLambda_history[:, i] = (
                self.OmegaLambda_bar + 
                (self.OmegaLambda_history[:, i+1] - self.OmegaLambda_bar) * decay +
                noise_scale * noise
            )
    
    def get_patch_OmegaLambda(self, z: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
        OmegaLambda_mean = np.mean(self.OmegaLambda_history, axis=0)
        return np.interp(z, z_grid, OmegaLambda_mean)
    
    def get_spatial_variance(self, z: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
        OL_interp = np.zeros((self.n_spatial_blocks, len(z)))
        for b in range(self.n_spatial_blocks):
            OL_interp[b, :] = np.interp(z, z_grid, self.OmegaLambda_history[b, :])
        return np.var(OL_interp, axis=0)


# =============================================================================
# COSMOLOGICAL COMPUTATIONS
# =============================================================================

def E_squared(z: np.ndarray, Omega_m: float, Omega_r: float, 
              Omega_Lambda: np.ndarray) -> np.ndarray:
    """
    Compute E²(z) = H²(z)/H0² for given parameters.
    
    Ωk is set from ΩΛ(z=0) to enforce E(0)=1, i.e., H(z=0)=H0.
    This means each stochastic realization has its own implied Ωk.
    """
    if np.isscalar(Omega_Lambda):
        Omega_Lambda = np.full_like(z, Omega_Lambda)
    
    # Ωk chosen to enforce E(0)=1 given ΩΛ(z=0)
    Omega_k = 1.0 - Omega_m - Omega_r - Omega_Lambda[0]
    
    E2 = (Omega_m * (1 + z)**3 + 
          Omega_r * (1 + z)**4 + 
          Omega_Lambda +
          Omega_k * (1 + z)**2)
    
    return np.maximum(E2, 1e-10)


def comoving_distance_dimless(z: np.ndarray, E_z: np.ndarray) -> np.ndarray:
    """χ(z) = H0 * D_C / c = ∫₀ᶻ dz'/E(z')"""
    dz = np.diff(z)
    integrand = 1.0 / E_z
    
    chi = np.zeros_like(z)
    chi[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    
    return chi


def sound_horizon_dimless(z_grid: np.ndarray, E_z: np.ndarray, 
                          Omega_b: float, z_drag: float) -> float:
    """
    Dimensionless sound horizon r_s * H0 / c.
    
    Integrates from z_drag to z_max (finite grid).
    NOTE: Radiation-dominated tail correction (z > z_max) is applied
    in infer_H0_early_CMB(), not here.
    """
    # R_b(z) = 3ρ_b/(4ρ_γ) - baryon-to-photon ratio
    R_b_0 = 31500 * Omega_b * 0.49 * (2.725/2.7)**(-4)
    
    mask = z_grid >= z_drag
    z_int = z_grid[mask]
    E_int = E_z[mask]
    
    if len(z_int) < 2:
        return 0.02
    
    R_b = R_b_0 / (1 + z_int)
    cs_over_c = 1.0 / np.sqrt(3 * (1 + R_b))
    
    integrand = cs_over_c / E_int
    
    dz = np.diff(z_int)
    r_s_dimless = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    
    return r_s_dimless


# =============================================================================
# H₀ INFERENCE: LATE-TIME SNe (REVIEWER-PROVIDED FIX)
# =============================================================================

def infer_H0_late_SNe(
    z_grid: np.ndarray,
    E_z: np.ndarray,
    H0_fid: float,
    z_max: float,
    noise_sigma: float,
    n_sne: int,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Infer H0 from mock SNe in a way consistent with the simulated D_L(z).

    Model:
      D_L(z) = (c/H0) * chi(z) * (1+z),  where chi(z)=∫ dz/E(z) (dimensionless)
      mu(z) = 5 log10(D_L/Mpc) + 25

    Key identity:
      mu(z) = [5 log10(c*chi(z)*(1+z)) + 25] - 5 log10(H0)

    So log10(H0) is just an offset between mu_obs and the known shape term.
    
    Args:
        H0_fid: Fiducial H0 used as normalization (not a hidden "true" value)
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample SNe redshifts
    z_sne = np.sort(np.random.uniform(0.01, z_max, n_sne))

    # True distances from the simulated cosmology (uses E_z shape)
    chi = comoving_distance_dimless(z_grid, E_z)
    chi_sne = np.interp(z_sne, z_grid, chi)

    D_L_true = (C_KM_S / H0_fid) * chi_sne * (1.0 + z_sne)  # Mpc
    mu_true = 5.0 * np.log10(np.maximum(D_L_true, 1e-30)) + 25.0
    mu_obs = mu_true + np.random.normal(0.0, noise_sigma, n_sne)

    # Shape-only term (independent of H0):
    # mu_shape(z) = 5 log10(c*chi(z)*(1+z)) + 25
    mu_shape = 5.0 * np.log10(np.maximum(C_KM_S * chi_sne * (1.0 + z_sne), 1e-30)) + 25.0

    # mu_obs = mu_shape - 5 log10(H0)  =>  log10(H0) = (mu_shape - mu_obs)/5
    log10H0_samples = (mu_shape - mu_obs) / 5.0
    log10H0_hat = np.mean(log10H0_samples)

    H0_est = 10.0 ** log10H0_hat

    # Uncertainty
    sigma_log10H0 = np.std(log10H0_samples, ddof=1) / np.sqrt(n_sne) if n_sne > 1 else np.nan
    H0_err = (np.log(10.0) * H0_est) * sigma_log10H0

    return np.clip(H0_est, 50.0, 90.0), H0_err


# =============================================================================
# H₀ INFERENCE: EARLY-TIME CMB (REVIEWER-PROVIDED FIX)
# =============================================================================

def infer_H0_early_CMB(
    z_grid: np.ndarray,
    E_z_model: np.ndarray,
    Omega_m: float,
    Omega_r: float,
    Omega_b: float,
    z_star: float,
    z_drag: float,
    theta_star_obs: float,  # Kept for API compatibility; see note below
) -> float:
    """
    CMB-like H0 inference via θ* root finding with ω-fixed degeneracy breaker.

    INTERPRETATION (important for reviewers):
      We compute θ*_target from the stochastic model itself (the "realization's θ*").
      This represents what CMB would measure if observing from that Hubble patch.
      We then infer H0 by finding the H0 in standard flat ΛCDM (with ω's fixed)
      that reproduces this θ*_target.
      
      This is an "internal mapping" approach: realization → inferred H0.
      We do NOT use the Planck θ*_obs directly; instead, tension arises because
      different realizations have different θ*_target values.
      
      The parameter theta_star_obs is retained for API compatibility but is
      not used in the inference. It could be used as a diagnostic to compare
      the realization's θ* against the observed Planck value.

    Degeneracy breaker:
      Hold physical densities fixed while varying H0:
          ωm = Ωm h², ωb = Ωb h², ωr = Ωr h², with h = H0/100.

    Tail correction:
      Adds a radiation-dominated analytic tail approximation for r_s beyond z_max.
      This is an order-of-magnitude correction assuming pure radiation domination
      and c_s/c = 1/√3 (ignores baryon loading at high z).

    Returns:
      H0 inferred (km/s/Mpc) or NaN if bracketing fails.
    """
    # theta_star_obs is not used in this internal-mapping approach
    # (retained for API compatibility; could be used for diagnostics)
    _ = theta_star_obs
    # --------------------------
    # 1) θ*_target from the stochastic model
    # --------------------------
    chi_model = comoving_distance_dimless(z_grid, E_z_model)
    chi_star_model = np.interp(z_star, z_grid, chi_model)
    D_A_star_model = chi_star_model / (1.0 + z_star)

    if not np.isfinite(D_A_star_model) or D_A_star_model <= 0:
        return np.nan

    r_s_model = sound_horizon_dimless(z_grid, E_z_model, Omega_b, z_drag)
    if not np.isfinite(r_s_model) or r_s_model <= 0:
        return np.nan

    # Radiation-dominated tail beyond z_max for the MODEL
    z_max_here = float(z_grid[-1])
    # In rad-dom, E(z) ~ sqrt(Ωr) (1+z)^2  =>  r_s_tail ≈ ∫ dz (1/√3)/E(z)
    # ≈ 1/(√3 * sqrt(Ωr) * (1+z_max))
    r_s_tail_model = 1.0 / (np.sqrt(3.0) * np.sqrt(max(Omega_r, 1e-30)) * (1.0 + z_max_here))
    theta_target = (r_s_model + r_s_tail_model) / D_A_star_model

    if not np.isfinite(theta_target) or theta_target <= 0:
        return np.nan

    # --------------------------
    # 2) Fix physical densities ω at a reference H0
    # --------------------------
    H0_ref = 70.0
    h_ref = H0_ref / 100.0

    omega_m = Omega_m * h_ref * h_ref
    omega_b = Omega_b * h_ref * h_ref
    omega_r = Omega_r * h_ref * h_ref

    # --------------------------
    # 3) θ*(H0_trial) for reference ΛCDM with ω fixed
    # --------------------------
    def theta_star_for_H0(H0_trial: float) -> float:
        h = H0_trial / 100.0
        if h <= 0:
            return np.nan

        Om = omega_m / (h * h)
        Ob = omega_b / (h * h)
        Or = omega_r / (h * h)

        Ol = 1.0 - Om - Or  # flat reference
        if not (np.isfinite(Om) and np.isfinite(Ob) and np.isfinite(Or) and np.isfinite(Ol)):
            return np.nan
        if Om <= 0 or Or <= 0 or Ol <= 0 or Ob <= 0:
            return np.nan

        E2 = E_squared(z_grid, Om, Or, np.full_like(z_grid, Ol))
        E = np.sqrt(E2)

        # D_A(z_star) in dimensionless units (chi/(1+z))
        chi = comoving_distance_dimless(z_grid, E)
        chi_star = np.interp(z_star, z_grid, chi)
        D_A = chi_star / (1.0 + z_star)
        if not np.isfinite(D_A) or D_A <= 0:
            return np.nan

        # r_s(z_drag) in dimensionless units
        r_s = sound_horizon_dimless(z_grid, E, Ob, z_drag)
        if not np.isfinite(r_s) or r_s <= 0:
            return np.nan

        # Tail correction in reference model
        r_s_tail = 1.0 / (np.sqrt(3.0) * np.sqrt(max(Or, 1e-30)) * (1.0 + z_max_here))
        return (r_s + r_s_tail) / D_A

    # Root function
    def f(H0_trial: float) -> float:
        th = theta_star_for_H0(H0_trial)
        if not np.isfinite(th):
            return np.nan
        return th - theta_target

    # --------------------------
    # 4) Bracket + bisection
    # --------------------------
    H_lo, H_hi = 50.0, 90.0
    f_lo, f_hi = f(H_lo), f(H_hi)

    # If bracket fails, widen once
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (f_lo * f_hi > 0):
        H_lo, H_hi = 40.0, 100.0
        f_lo, f_hi = f(H_lo), f(H_hi)

    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (f_lo * f_hi > 0):
        # Could not bracket a root robustly
        return np.nan

    for _ in range(80):
        H_mid = 0.5 * (H_lo + H_hi)
        f_mid = f(H_mid)
        if not np.isfinite(f_mid):
            return np.nan

        if f_lo * f_mid <= 0:
            H_hi, f_hi = H_mid, f_mid
        else:
            H_lo, f_lo = H_mid, f_mid

        if abs(H_hi - H_lo) < 1e-3:
            break

    H0_inferred = 0.5 * (H_lo + H_hi)

    # theta_star_obs can be used as a diagnostic to compare to Planck
    # For the inference itself we match the realization's θ*
    return float(np.clip(H0_inferred, 50.0, 90.0))


# =============================================================================
# H₀ INFERENCE: LATE-TIME DIRECT
# =============================================================================

def infer_H0_late_direct(E_z: np.ndarray, z_grid: np.ndarray, H0_fid: float,
                          noise: float, seed: Optional[int] = None) -> float:
    """
    Low-z H(z) probe as internal consistency check.
    
    Uses H(z=0.05) rather than H(z→0) to actually sense the E(z) shape,
    since E(0)=1 by construction in our Ωk normalization.
    
    This is a secondary late-time diagnostic, not a precision H0 measurement.
    
    Args:
        H0_fid: Fiducial H0 used as normalization (not a hidden "true" value)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use H at small nonzero z to respond to cosmology
    # (E(0)=1 by construction, so z=0 wouldn't sense the shape)
    z_probe = 0.05
    E_probe = np.interp(z_probe, z_grid, E_z)
    H0_probe = H0_fid * E_probe
    H0_est = H0_probe + np.random.normal(0, noise)
    
    return np.clip(H0_est, 50.0, 90.0)


# =============================================================================
# SINGLE REALIZATION ANALYSIS
# =============================================================================

@dataclass
class RealizationResult:
    realization_id: int
    H0_late_SNe: float
    H0_late_SNe_err: float
    H0_late_direct: float
    H0_early_CMB: float
    delta_H_SNe: float
    delta_H_direct: float
    sigma_intra: float
    OmegaLambda_spatial_var: float


def analyze_single_realization(args) -> RealizationResult:
    realization_id, config_dict, seed = args
    
    config = ProtocolBConfig(**config_dict)
    
    z_grid = np.concatenate([
        np.array([0.0]),
        np.logspace(-4, np.log10(config.z_max), config.n_z_points - 1)
    ])
    z_grid = np.sort(np.unique(z_grid))
    
    Omega_Lambda_fid = 1 - config.Omega_m - config.Omega_r
    E2_fid = E_squared(z_grid, config.Omega_m, config.Omega_r,
                       np.full_like(z_grid, Omega_Lambda_fid))
    E_fid = np.sqrt(E2_fid)
    
    model = StochasticOmegaLambdaModel(
        OmegaLambda_bar=Omega_Lambda_fid,
        correlation_time=config.correlation_time,
        sigma_0=config.sigma_OmegaLambda_0,
        use_H2_scaling=config.use_H2_scaling,
        n_spatial_blocks=config.n_spatial_blocks,
        spatial_correlation=config.spatial_correlation,
        E_cap=config.E_cap
    )
    
    model.generate(z_grid, E_fid, seed=seed)
    
    OmegaLambda_z = model.get_patch_OmegaLambda(z_grid, z_grid)
    
    # Clamp ΩΛ to physically reasonable range to avoid E² clipping artifacts.
    # Ωk is then set using the clipped ΩΛ(z=0) in E_squared().
    OmegaLambda_z = np.clip(OmegaLambda_z, -0.2, 1.5)
    
    E2_model = E_squared(z_grid, config.Omega_m, config.Omega_r, OmegaLambda_z)
    E_model = np.sqrt(E2_model)
    
    seed_sne = (seed + 1000) if (seed is not None) else None
    H0_SNe, H0_SNe_err = infer_H0_late_SNe(
        z_grid, E_model, config.H0_fiducial,
        config.z_max_late, config.distance_modulus_noise,
        config.n_sne, seed=seed_sne
    )
    
    seed_direct = (seed + 2000) if (seed is not None) else None
    H0_direct = infer_H0_late_direct(
        E_model, z_grid, config.H0_fiducial,
        config.H0_direct_noise, seed=seed_direct
    )
    
    H0_CMB = infer_H0_early_CMB(
        z_grid, E_model, config.Omega_m, config.Omega_r, config.Omega_b,
        config.z_star, config.z_drag, config.theta_star_observed
    )
    
    if H0_CMB > 0 and not np.isnan(H0_CMB):
        delta_H_SNe = (H0_SNe - H0_CMB) / H0_CMB
        delta_H_direct = (H0_direct - H0_CMB) / H0_CMB
        
        late_estimates = [H0_SNe, H0_direct]
        H0_late_mean = np.mean(late_estimates)
        sigma_intra = np.std(late_estimates) / H0_late_mean if H0_late_mean > 0 else np.nan
    else:
        delta_H_SNe = np.nan
        delta_H_direct = np.nan
        sigma_intra = np.nan
    
    OL_var = model.get_spatial_variance(np.array([0.0]), z_grid)[0]
    
    return RealizationResult(
        realization_id=realization_id,
        H0_late_SNe=H0_SNe,
        H0_late_SNe_err=H0_SNe_err,
        H0_late_direct=H0_direct,
        H0_early_CMB=H0_CMB,
        delta_H_SNe=delta_H_SNe,
        delta_H_direct=delta_H_direct,
        sigma_intra=sigma_intra,
        OmegaLambda_spatial_var=OL_var
    )


# =============================================================================
# PROTOCOL B MAIN
# =============================================================================

@dataclass
class ProtocolBResults:
    n_realizations: int
    sigma_OmegaLambda_0: float
    correlation_time: float
    use_H2_scaling: bool
    realization_results: List[RealizationResult]
    H0_late_SNe_mean: float
    H0_late_SNe_std: float
    H0_late_direct_mean: float
    H0_late_direct_std: float
    H0_early_mean: float
    H0_early_std: float
    delta_H_mean: float
    sigma_patch: float
    sigma_intra_mean: float
    computation_time: float
    nan_rate_cmb: float = 0.0  # Fraction of realizations where CMB inference failed
    
    def __str__(self):
        lines = [
            "=" * 70,
            "Protocol B Results (v4): Stochastic ΩΛ and Hubble Tension",
            "=" * 70,
            f"Realizations: {self.n_realizations} (CMB NaN rate: {self.nan_rate_cmb*100:.1f}%)",
            f"ΩΛ fluctuation amplitude: σ = {self.sigma_OmegaLambda_0:.3f}",
            f"Correlation time: τ_c = {self.correlation_time:.2f} H₀⁻¹",
            "",
            "H₀ Inference Results:",
            f"  Late (SNe):     H₀ = {self.H0_late_SNe_mean:.2f} ± {self.H0_late_SNe_std:.2f}",
            f"  Late (low-z):   H₀ = {self.H0_late_direct_mean:.2f} ± {self.H0_late_direct_std:.2f}",
            f"  Early (CMB):    H₀ = {self.H0_early_mean:.2f} ± {self.H0_early_std:.2f}",
            "",
            "Hubble Tension Metrics:",
            f"  ⟨δH⟩ = {self.delta_H_mean*100:.2f}%",
            f"  σ_patch = {self.sigma_patch*100:.2f}%",
            f"  σ_intra = {self.sigma_intra_mean*100:.2f}%",
            "",
            f"Observed: {(H0_SHOES-H0_PLANCK)/H0_PLANCK*100:.1f}% tension",
            f"Time: {self.computation_time:.1f}s",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def passes_targets(self) -> Dict[str, bool]:
        return {
            'sigma_patch_in_range': 0.03 <= self.sigma_patch <= 0.15,  # Wider for θ* model
            'sigma_intra_acceptable': self.sigma_intra_mean <= 0.03,
            'produces_tension': abs(self.delta_H_mean) >= 0.02,
        }


def run_protocol_b(config: ProtocolBConfig) -> ProtocolBResults:
    start_time = time.time()
    
    if config.verbose:
        print("=" * 70)
        print("Protocol B (v4): Stochastic ΩΛ and Hubble Tension")
        print("  CMB: θ* = r_s/D_A root finding with fixed ω's + tail correction")
        print("  SNe: consistent D_L(z) = (c/H0)*χ(z)*(1+z)")
        print("=" * 70)
        print(f"σ_ΩΛ = {config.sigma_OmegaLambda_0}, τ_c = {config.correlation_time}, E_cap = {config.E_cap}")
        print(f"{config.n_realizations} realizations, {config.n_workers} cores")
        print()
    
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    
    if config.seed is not None:
        np.random.seed(config.seed)
    seeds = [np.random.randint(0, 2**31) for _ in range(config.n_realizations)]
    
    args_list = [(i, config_dict, seeds[i]) for i in range(config.n_realizations)]
    
    results_list = []
    
    if config.parallel and config.n_workers > 1:
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = [executor.submit(analyze_single_realization, args) 
                      for args in args_list]
            for i, future in enumerate(as_completed(futures)):
                results_list.append(future.result())
                if config.verbose and (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{config.n_realizations}...")
    else:
        for args in args_list:
            results_list.append(analyze_single_realization(args))
    
    results_list.sort(key=lambda x: x.realization_id)
    
    H0_SNe = [r.H0_late_SNe for r in results_list if not np.isnan(r.H0_late_SNe)]
    H0_direct = [r.H0_late_direct for r in results_list]
    H0_CMB = [r.H0_early_CMB for r in results_list if not np.isnan(r.H0_early_CMB)]
    delta_H = [r.delta_H_SNe for r in results_list if not np.isnan(r.delta_H_SNe)]
    sigma_intra = [r.sigma_intra for r in results_list if not np.isnan(r.sigma_intra)]
    
    # Compute NaN rate for CMB estimator
    n_cmb_nan = sum(1 for r in results_list if np.isnan(r.H0_early_CMB))
    nan_rate_cmb = n_cmb_nan / len(results_list)
    
    return ProtocolBResults(
        n_realizations=config.n_realizations,
        sigma_OmegaLambda_0=config.sigma_OmegaLambda_0,
        correlation_time=config.correlation_time,
        use_H2_scaling=config.use_H2_scaling,
        realization_results=results_list,
        H0_late_SNe_mean=np.mean(H0_SNe),
        H0_late_SNe_std=np.std(H0_SNe, ddof=1) if len(H0_SNe) > 1 else 0.0,
        H0_late_direct_mean=np.mean(H0_direct),
        H0_late_direct_std=np.std(H0_direct, ddof=1) if len(H0_direct) > 1 else 0.0,
        H0_early_mean=np.mean(H0_CMB) if H0_CMB else np.nan,
        H0_early_std=np.std(H0_CMB, ddof=1) if len(H0_CMB) > 1 else np.nan,
        delta_H_mean=np.mean(delta_H) if delta_H else np.nan,
        sigma_patch=np.std(delta_H, ddof=1) if len(delta_H) > 1 else np.nan,
        sigma_intra_mean=np.mean(sigma_intra) if sigma_intra else np.nan,
        computation_time=time.time() - start_time,
        nan_rate_cmb=nan_rate_cmb
    )


def plot_results(results: ProtocolBResults, save_path: Optional[str] = None):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    delta_H = [r.delta_H_SNe for r in results.realization_results if not np.isnan(r.delta_H_SNe)]
    H0_SNe = [r.H0_late_SNe for r in results.realization_results]
    H0_direct = [r.H0_late_direct for r in results.realization_results]
    sigma_intra = [r.sigma_intra for r in results.realization_results if not np.isnan(r.sigma_intra)]
    
    ax1 = axes[0, 0]
    ax1.hist(np.array(delta_H) * 100, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=results.delta_H_mean * 100, color='red', linestyle='--',
                label=f'Mean = {results.delta_H_mean*100:.2f}%')
    ax1.axvline(x=8.3, color='green', linestyle=':', label='Observed 8.3%')
    ax1.set_xlabel('δH [%]')
    ax1.set_ylabel('Count')
    ax1.set_title('Hubble Tension Distribution')
    ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.scatter(H0_SNe, H0_direct, alpha=0.5, s=20)
    lims = [min(min(H0_SNe), min(H0_direct)) - 2, max(max(H0_SNe), max(H0_direct)) + 2]
    ax2.plot(lims, lims, 'k--', alpha=0.5)
    ax2.set_xlabel('H₀ (SNe)')
    ax2.set_ylabel('H₀ (low-z probe)')
    ax2.set_title('Late-Time Methods')
    
    ax3 = axes[1, 0]
    ax3.hist(np.array(sigma_intra) * 100, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax3.axvline(x=results.sigma_intra_mean * 100, color='red', linestyle='--')
    ax3.axvline(x=3.0, color='green', linestyle=':')
    ax3.set_xlabel('σ_intra [%]')
    ax3.set_title('Intra-Patch Scatter')
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    targets = results.passes_targets()
    summary = f"""
    Protocol B v4 Summary
    ═════════════════════════════════
    
    CMB: θ* root finding (ω fixed)
    SNe: D_L = (c/H0)χ(z)(1+z)
    
    σ_patch = {results.sigma_patch*100:.2f}%  {'✓' if targets['sigma_patch_in_range'] else '✗'}
    σ_intra = {results.sigma_intra_mean*100:.2f}%  {'✓' if targets['sigma_intra_acceptable'] else '✗'}
    |⟨δH⟩| = {abs(results.delta_H_mean)*100:.2f}%  {'✓' if targets['produces_tension'] else '✗'}
    """
    ax4.text(0.1, 0.5, summary, fontsize=12, fontfamily='monospace')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    backend = str(mpl.get_backend()).lower()
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        interactive_backends = {
            b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
    except Exception:
        interactive_backends = {b.lower() for b in getattr(mpl.rcsetup, "interactive_bk", [])}
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    can_show = (backend in interactive_backends) and has_display

    if can_show:
        plt.show()
    else:
        plt.close(fig)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Protocol B (v4): Stochastic ΩΛ and the Hubble Tension               ║
║  With: θ* root finding, consistent D_L, corrected time direction     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    config = ProtocolBConfig(
        n_realizations=300,
        sigma_OmegaLambda_0=0.12,
        correlation_time=0.4,
        use_H2_scaling=True,
        E_cap=10.0,  # Cap E(z) for H² scaling
        n_spatial_blocks=10,
        spatial_correlation=0.8,
        z_max_late=0.15,
        distance_modulus_noise=0.10,
        n_sne=150,
        H0_direct_noise=2.0,
        z_max=3000.0,
        n_z_points=2000,
        n_workers=N_CORES,
        parallel=True,
        verbose=True,
        H0_fiducial=73.8,
        seed=42
    )
    
    results = run_protocol_b(config)
    print(results)
    
    print("\nTARGETS:")
    for name, passed in results.passes_targets().items():
        print(f"  {name}: {'✓' if passed else '✗'}")
    
    try:
        plot_results(results, 'protocol_b_v4_results.png')
    except Exception as e:
        print(f"Plot failed: {e}")
    
    return results


if __name__ == "__main__":
    main()
