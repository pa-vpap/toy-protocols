```md
# Protocol B (v4) — Results & Plot Guide (Paper: “Protocol B”)

This README explains how to interpret the **Protocol B** run output and the generated figure (`protocol_b_v4_results.png`) in the context of the paper’s **Protocol B: Stochastic ΩΛ → early/late H₀ tension** mechanism.

Protocol B is a **toy mechanism demo**: each Monte Carlo realization represents a distinct **Hubble patch** with a stochastic vacuum-energy history ΩΛ(z). The patch then “measures”:
- an **early-time** CMB-like observable (θ\*), and infers an **early-time H₀** via ω-fixed ΛCDM mapping;
- a **late-time** H₀ via a **self-consistent SNe distance ladder** estimator;
- plus a secondary late-time diagnostic from a low-z H(z) probe.

The goal (as described in the paper) is to show that **patch-to-patch stochasticity** can generate an apparent **Hubble tension** even when each patch is internally self-consistent.

---

## Run configuration (the “paper-matching” tension case)

This result corresponds to the configuration tuned to reproduce an ~8.3% mean tension:

- `n_realizations = 300`
- `sigma_OmegaLambda_0 = 0.12`
- `correlation_time = 0.4 (H0^-1)`
- `use_H2_scaling = True`, `E_cap = 10`
- `n_spatial_blocks = 10`, `spatial_correlation = 0.8`
- `z_max = 3000`, `n_z_points = 2000`
- **late-time calibration:** `H0_fiducial = 73.8`
- `seed = 42`

**Important interpretation of `H0_fiducial`:**  
In Protocol B it acts like the **distance-ladder calibration scale** (a “late anchor”), not a hidden true cosmological H₀. Raising/lowering it shifts the *late-time inferred H₀* and therefore shifts the mean δH.

---

## Summary table — how to read the printed results

```

Realizations: 300 (CMB NaN rate: 1.3%)
ΩΛ fluctuation amplitude: σ = 0.120
Correlation time: τ_c = 0.40 H₀⁻¹

H₀ Inference Results:
Late (SNe):     H₀ = 73.77 ± 0.27
Late (low-z):   H₀ = 75.49 ± 2.07
Early (CMB):    H₀ = 69.11 ± 8.04

Hubble Tension Metrics:
⟨δH⟩ = 8.28%
σ_patch = 13.57%
σ_intra = 1.46%

Observed: 8.3% tension

```

### What each line means (Protocol B mapping)

#### **CMB NaN rate: 1.3%**
A small fraction of realizations fail the θ\* root bracketing (early-time inference returns NaN).  
This is expected when stochastic ΩΛ pushes the realization into a shape where θ\*(H₀) can’t be robustly bracketed in the search window.

#### **Late (SNe): H₀ = 73.77 ± 0.27**
This is the **late-time inferred H₀** from the mock distance ladder:
- It uses the realized expansion history E(z) for the **shape** of D_L(z),
- and extracts **H₀ as the normalization** offset in the distance modulus relation.

The small scatter (±0.27) is mostly driven by the chosen SNe noise and sample size.

#### **Late (low-z): H₀ = 75.49 ± 2.07**
A **secondary diagnostic**, not a precision estimator.  
It evaluates a low-z probe at z=0.05:

- It is intentionally noisier,
- and provides an **intra-patch consistency check** against the SNe estimate.

#### **Early (CMB): H₀ = 69.11 ± 8.04**
This is the **early-time inferred H₀** from the patch’s **own** θ\* (internal mapping):
1) compute θ\* = r_s / D_A using the realization’s E(z),  
2) then solve for the H₀ in a *reference flat ΛCDM* with **ω fixed** that reproduces this θ\*.

The larger scatter (±8.04) is the key Protocol B feature:  
stochastic ΩΛ histories induce significant patch-to-patch variation in θ\*, hence in inferred H₀.

---

## The key tension quantities (the “Protocol B” observables)

### δH (per realization)
Protocol B defines the fractional tension as:

\[
\delta H \equiv \frac{H_{0,\text{late}} - H_{0,\text{early}}}{H_{0,\text{early}}}
\]

In the summary:
- **⟨δH⟩ = 8.28%** (mean over realizations with valid CMB inference)

This matches the “observed” benchmark shown in the printout (~8.3%).

### σ_patch (patch-to-patch scatter)
- **σ_patch = 13.57%** is the standard deviation of δH across patches.

Interpretation (paper context):  
the ensemble of Hubble patches does not share a single inferred H₀—there is a broad distribution of early/late mismatch due to stochastic ΩΛ.

### σ_intra (intra-patch scatter)
- **σ_intra = 1.46%** measures how consistent late-time methods are *within the same patch* (SNe vs low-z probe).

Interpretation (paper context):  
late-time measurements are mutually consistent inside each patch, while the early-time mapping can disagree—this is the desired “tension-like” structure.

---

## Figure: `protocol_b_v4_results.png` — panel-by-panel explanation

The plot is designed to visually match the paper’s Protocol B narrative: **one mechanism → three diagnostics**.

### (Top-left) Hubble Tension Distribution (histogram of δH)
- **x-axis:** δH [%] across realizations  
- **red dashed line:** mean ⟨δH⟩ = 8.28%  
- **green dotted line:** reference “Observed 8.3%”

**What it shows:**  
Protocol B produces a broad δH distribution whose mean can be tuned to match the observed Hubble tension level. The width is controlled mainly by `sigma_OmegaLambda_0`, `correlation_time`, and the H² scaling cap (`E_cap`).

### (Top-right) Late-Time Methods (scatter: H₀(SNe) vs H₀(low-z))
- each point is one realization (patch)
- the dashed diagonal is y=x

**What it shows:**  
Points cluster in a near-vertical band around **H₀(SNe) ≈ 73–74**, with **moderate scatter in low-z probe**.  
This supports the Protocol B requirement that *late-time estimators remain broadly consistent*.

### (Bottom-left) Intra-Patch Scatter (histogram of σ_intra)
- **red dashed:** mean σ_intra (~1.46%)
- **green dotted:** nominal acceptance line (3%)

**What it shows:**  
Most patches have small intra-patch disagreement—i.e., “late-time cosmology” is coherent inside a patch. This is important: Protocol B attributes the tension primarily to **early vs late inference mismatch**, not to inconsistent late probes.

### (Bottom-right) Summary panel
A text summary mirroring the paper’s “Protocol B checklist”:
- CMB inference method (θ\*, ω fixed)
- SNe distance ladder formula
- σ_patch, σ_intra, |⟨δH⟩| and pass/fail markers

---

## Why this run “works” (paper perspective)

This configuration achieves the Protocol B behavior:

- **Mean tension matches observation:**  
  ⟨δH⟩ ≈ 8.28% ~ 8.3%

- **Patch-to-patch scatter is sizeable:**  
  σ_patch ≈ 13.6% (broad distribution across patches)

- **Intra-patch late-time consistency remains small:**  
  σ_intra ≈ 1.46% (below the 3% target)

Mechanistically, this corresponds to the paper’s thesis for Protocol B:
> stochastic ΩΛ histories shift the effective early-time ruler/angle mapping (θ\* → H₀), while the late-time ladder remains anchored to its own calibration scale, producing an apparent tension.

---

## Notes / caveats (toy-model disclaimers)

Protocol B is not a full data-fitting pipeline:
- θ\* is *internally generated* per patch (not directly compared to Planck in the inference step).
- The radiation tail correction is approximate (radiation-dominated analytic tail).
- Ωk is chosen per realization to enforce E(0)=1, which is a normalization device for patch-level H₀.

These are deliberate simplifications to keep Protocol B a fast, transparent mechanism test consistent with the paper’s intended role for “Protocol B”.

---
```
