````md
# Protocol B (v4): Stochastic ΩΛ and the Hubble Tension

This repository contains the reference implementation of **Protocol B** from the paper (Protocol section: **“PROTOCOL B – Stochastic ΩΛ and the Hubble Tension”**).

Protocol B is a **toy cosmology pipeline** designed to demonstrate how **patch-to-patch stochastic fluctuations in the effective vacuum energy density** (modeled as a correlated OU process in redshift-time) can generate an **early–late H₀ tension**—even when each patch is internally consistent.

The code produces:
- a distribution of **Hubble tension** values across many “Hubble patches” (realizations),
- the **late-time H₀** from a distance-ladder-like SNe estimator,
- the **early-time H₀** from a CMB-like **θ\*** inference mapping,
- sanity diagnostics (NaN rate, intra-patch scatter, patch-to-patch variance),
- and publication-ready plots.

---

## Conceptual mapping to the paper (Protocol B)

In the paper’s narrative, **each realization corresponds to a single Hubble patch** drawn from an ensemble of stochastic geometries / effective cosmologies. The patch contains:

1) A **stochastic ΩΛ(z)** history (vacuum-energy fluctuations),
2) A resulting **expansion history** \( E(z) = H(z)/H_0 \),
3) An **early-time “CMB” observable** \( \theta_* \equiv r_s/D_A \),
4) A **late-time “SNe” distance ladder** inference of \( H_0 \).

Protocol B creates a tension because:

- **Early-time inference** uses a **θ\*-to-H₀ mapping** in a *reference ΛCDM family with fixed physical densities* (ω-fixed degeneracy breaker).
- **Late-time inference** uses a *distance-ladder-like normalization* in the same patch, via a self-consistent luminosity distance shape.

Even though each patch is internally coherent, **the inferred H₀ from early-time (θ\*) differs from the inferred H₀ from late-time (SNe)** when ΩΛ fluctuates across cosmic time.

---

## What this code implements

### 1) Stochastic ΩΛ model (OU process in cosmic time)
- ΩΛ is generated as an **Ornstein–Uhlenbeck (OU)** process.
- Time flows from **early (high z)** → **late (low z)**.
- Fluctuations are correlated across **spatial blocks** using a Toeplitz covariance \( \rho^{|i-j|} \).
- Fluctuation amplitude scales as:

\[
\sigma(z) = \sigma_0 \cdot \min(E(z), E_{\text{cap}})^2
\]

This is a toy “H²-scaling” proxy to increase fluctuation amplitude at early epochs while remaining numerically stable.

### 2) Expansion history
Given Ωm, Ωr, and stochastic ΩΛ(z), the code computes:

\[
E^2(z) = \Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_\Lambda(z) + \Omega_k(1+z)^2
\]

**Important normalization choice:**
- \( \Omega_k \) is chosen **per realization** so that **E(0)=1**, i.e. the patch has a well-defined H₀ normalization.

ΩΛ(z) is clamped for stability:
- \( \Omega_\Lambda \in [-0.2, 1.5] \)

### 3) Late-time H₀ inference (SNe distance ladder)
The SNe estimator uses a self-consistent luminosity distance:

\[
D_L(z) = \frac{c}{H_0}\,\chi(z)\,(1+z),\quad
\chi(z) = \int_0^z \frac{dz'}{E(z')}
\]

The key identity used in the estimator is:

\[
\mu(z) = \big[5\log_{10}(c\chi(1+z)) + 25\big] - 5\log_{10}(H_0)
\]

So **H₀ is an offset** between the measured distance moduli and the known shape term.

**Note on `H0_fiducial`:**
- In this toy model, `H0_fiducial` is the *late-time calibration scale* (distance-ladder anchor), not a hidden “true” cosmological parameter.

### 4) Early-time H₀ inference (CMB-like θ\*)
Protocol B does *not* compare directly to Planck θ\*.
Instead it implements the paper’s “internal mapping” idea:

1. Compute the patch’s own:

\[
\theta_{*,\text{target}} = \frac{r_s(z_{\rm drag})}{D_A(z_*)}
\]

2. Infer H₀ by finding the H₀ in **reference flat ΛCDM** (with **ω fixed**) such that:

\[
\theta_*(H_0) = \theta_{*,\text{target}}
\]

**Degeneracy breaker (ω-fixed):**
- physical densities are held fixed while varying H₀:

\[
\omega_m = \Omega_m h^2,\quad
\omega_b = \Omega_b h^2,\quad
\omega_r = \Omega_r h^2,\quad h\equiv H_0/100
\]

**Radiation tail correction:**
Because the integral only runs to `z_max`, an analytic radiation-dominated tail is added to the sound horizon:

\[
r_{s,\text{tail}} \approx \frac{1}{\sqrt{3}\sqrt{\Omega_r}(1+z_{\max})}
\]

This is a **controlled approximation** used for stability and speed.

---

## Outputs and interpretation

The run produces:
- Mean and scatter of late-time and early-time H₀
- Mean tension:

\[
\delta H \equiv \frac{H_{0,\text{late}} - H_{0,\text{early}}}{H_{0,\text{early}}}
\]

- Patch-to-patch variance σ_patch = std(δH)
- Intra-patch consistency σ_intra (agreement between SNe and low-z probe)
- NaN rate for the CMB root solve

The included plot (`protocol_b_v4_results.png`) contains:
1. Histogram of δH across patches (tension distribution)
2. SNe H₀ vs low-z probe H₀ scatter (late-time consistency diagnostic)
3. Intra-patch scatter histogram
4. Summary panel with pass/fail targets

---

## Quickstart

### Requirements
- Python 3.9+
- `numpy`
- (optional) `matplotlib` for plots

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
````

Run:

```bash
python test_protocol_b.py
```

This will print the summary table and (by default) write:

* `protocol_b_v4_results.png`

---

## Recommended “paper-matching” configuration

The following configuration matches the paper-style Protocol B behavior where the mean tension is near the observed ~8.3%:

```python
config = ProtocolBConfig(
    n_realizations=300,
    sigma_OmegaLambda_0=0.12,
    correlation_time=0.4,
    use_H2_scaling=True,
    E_cap=10.0,
    n_spatial_blocks=10,
    spatial_correlation=0.8,
    z_max_late=0.15,
    distance_modulus_noise=0.10,
    n_sne=150,
    H0_direct_noise=2.0,
    z_max=3000.0,
    n_z_points=2000,
    parallel=True,
    verbose=True,
    H0_fiducial=73.8,
    seed=42
)
```

---

## Parameter guide (what to tune and why)

### Primary knobs (affect the tension distribution)

* `H0_fiducial`
  Sets the **late-time calibration scale** (distance ladder). Increasing it shifts late-time H₀ up and increases mean δH.

* `sigma_OmegaLambda_0`
  Controls the **amplitude** of ΩΛ fluctuations (baseline). Increasing it typically increases **σ_patch** and can broaden/skew δH.

* `correlation_time` (τ_c in units of H₀⁻¹)
  Controls how quickly ΩΛ wanders in cosmic time. Smaller τ_c → “faster” fluctuations; larger τ_c → smoother histories.

* `E_cap`
  Caps the H² scaling at high z, preventing runaway variance. Lowering it reduces early-time volatility and may reduce CMB scatter/NaNs.

### Spatial correlation knobs

* `n_spatial_blocks` and `spatial_correlation`
  Control block-to-block covariance and how “coherent” ΩΛ is spatially within a patch ensemble.

### Numerical / estimator knobs

* `n_z_points`, `z_max`
  Integration fidelity. Increasing these improves accuracy but costs time.

* `z_max_late`, `n_sne`, `distance_modulus_noise`
  Control the SNe mock survey.

* `H0_direct_noise`
  Sets the width of the low-z probe diagnostic (not meant to be a precision estimator).

---

## Notes on limitations (toy-model disclaimers)

This implementation is **not** a full cosmological parameter inference pipeline and does not attempt to fit real data. It is a **mechanism demo** aligned with Protocol B’s purpose in the paper:

* ΩΛ(z) fluctuations are phenomenological and not derived from first principles here.
* Radiation-tail correction is approximate (pure radiation domination, constant sound speed).
* CMB “observation” is internal: θ* is generated by the realization and then mapped to an inferred H₀.

These choices are deliberate to keep Protocol B:

* fast,
* transparent,
* and focused on the early–late inference mismatch mechanism.

---

## Reproducibility

* Set `seed` for reproducible ensembles.
* The code parallelizes realizations via `ProcessPoolExecutor`.
* For deterministic debugging, set `parallel=False` and `n_workers=1`.

---

## License

MIT (as stated in the header).

```
```
