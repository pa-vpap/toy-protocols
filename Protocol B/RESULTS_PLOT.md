```md
# Protocol B (v4) — Plot Notes

![Protocol B v4 Results](Protocol%20B/protocol_b_v4_results.png)

This document explains the **four-panel diagnostic plot** produced by **Protocol B (v4)**, as used in the paper section **“PROTOCOL B – Stochastic ΩΛ and the Hubble Tension.”**

The figure summarizes how **stochastic vacuum energy histories** lead to an apparent **early–late H₀ tension** across an ensemble of Hubble patches, while preserving internal late-time consistency.

---

## Run context (for this plot)

**Configuration (representative):**
- Realizations: **300** (CMB NaN rate ≈ **1.3%**)
- σΩΛ = **0.12**
- Correlation time τc = **0.40 H₀⁻¹**
- H² scaling enabled, E_cap = **10**
- Spatial blocks = **10**, correlation = **0.8**
- Late-time calibration: **H0_fiducial = 73.8 km/s/Mpc**

**Headline results:**
- Late (SNe): **H₀ = 73.77 ± 0.27**
- Late (low-z probe): **H₀ = 75.49 ± 2.07**
- Early (CMB θ\*): **H₀ = 69.11 ± 8.04**
- Mean tension: **⟨δH⟩ = 8.28%**
- Patch scatter: **σ_patch = 13.57%**
- Intra-patch scatter: **σ_intra = 1.46%**

with
\[
\delta H \equiv \frac{H_{0,\text{late}} - H_{0,\text{early}}}{H_{0,\text{early}}}.
\]

---

## Panel A — *Hubble Tension Distribution* (top-left)

**What is shown:**  
A histogram of **δH [%]** across all realizations.

**Annotations:**
- **Red dashed line:** simulated mean ⟨δH⟩ = **8.28%**
- **Green dotted line:** reference observed tension (**8.3%**)

**Interpretation (Protocol B):**
- Each bar corresponds to a distinct **Hubble patch** with its own stochastic ΩΛ(z).
- The width of the distribution reflects **patch-to-patch cosmic variance** induced by vacuum-energy fluctuations.
- Agreement between the red and green lines shows that the mechanism can reproduce the *magnitude* of the observed Hubble tension within this toy framework.

---

## Panel B — *Late-Time Methods* (top-right)

**Axes:**
- x-axis: **H₀ (SNe distance ladder)**
- y-axis: **H₀ (low-z probe at z = 0.05)**

**Dashed diagonal:** y = x (perfect agreement).

**Interpretation:**
- The vertical clustering indicates that **SNe-based H₀** is tightly constrained.
- The broader vertical spread reflects the intentionally noisier **low-z probe**.
- This panel demonstrates that **late-time estimators are mutually consistent within each patch**, which is essential for Protocol B: the tension arises *between early and late inference*, not from internal late-time disagreement.

---

## Panel C — *Intra-Patch Scatter* (bottom-left)

**Quantity plotted:**  
Histogram of
\[
\sigma_{\text{intra}} = \frac{\mathrm{std}(H_{0,\text{SNe}}, H_{0,\text{low-z}})}{\mathrm{mean}(H_{0,\text{SNe}}, H_{0,\text{low-z}})}.
\]

**Annotations:**
- **Red dashed line:** mean σ_intra ≈ **1.46%**
- **Green dotted line:** reference acceptance threshold (**3%**)

**Interpretation:**
- Most realizations lie well below the 3% threshold.
- Late-time measurements inside a single patch are **self-consistent**, reinforcing the claim that Protocol B’s tension is not an artifact of inconsistent late-time probes.

---

## Panel D — *Protocol B v4 Summary* (bottom-right)

This text panel restates the core ingredients of Protocol B:

- **CMB:** θ\* root finding with **ω-fixed degeneracy breaking**
- **SNe:** \( D_L = (c/H_0)\,\chi(z)(1+z) \)

and reports the three key diagnostics used in the paper:
- **σ_patch = 13.57%** ✓ (patch-to-patch variance)
- **σ_intra = 1.46%** ✓ (within-patch consistency)
- **|⟨δH⟩| = 8.28%** ✓ (produces the observed-scale tension)

---

## Takeaway (paper-level interpretation)

This figure visually demonstrates the **Protocol B mechanism**:

1. **Stochastic ΩΛ(z)** histories modify early-time geometry and hence θ\*.
2. Mapping θ\* → H₀ in an ω-fixed ΛCDM reference produces a **broad distribution of early-time inferred H₀**.
3. Late-time H₀ remains well anchored by the distance ladder.
4. The mismatch yields a realistic **Hubble-tension-like signal** with large patch scatter but small intra-patch inconsistency.

In short, the plot is the compact empirical summary of **Protocol B’s claim**:  
> *A stochastic vacuum energy sector can generate an apparent early–late H₀ tension without breaking late-time self-consistency.*

---
```
