
## 1) Protocol C specs

**Common geometry / kinematics**

* Target dimension: **DIM = 4**
* Alexandrov interval height: **T_used = 1.0** (fixed; i.e., density varies with N)
* System sizes: **N = 180, 250, 350**

**Selector / action**

* Interval-abundance (BD-style) action built from open interval sizes:

  * `kmax = 3`
  * `alpha = [1, -9, 16, -8]`

**MCMC**

* `chains = 5`
* `burn = 1000`
* `steps = 16000`
* `thin = 80`
* Mixed proposal:

  * `mix_local = 0.5`
  * `local_sigma = 0.03`
* Euclidean diagnostic weighting: `exp(-β S)` with scan:

  * β ∈ {0.001, 0.002, 0.003, 0.0035, 0.004, 0.0042, 0.0044, 0.005, 0.006, 0.007, 0.01}

**Interval-statistics sampling & metrics**

* Comparable-pair sampling per measurement: `sample_pairs = 12000`
* Histogram truncation cap: `k_cap = 220`
* Rebinning enabled: `rebin = yes`
* Reported metrics per (N,β):

  * `JS4` and `JS4_stderr` (Jensen–Shannon divergence to 4D reference)
  * `JSKR` and `JSKR_stderr` (Jensen–Shannon divergence to KR baseline)
  * `p20` and `p20_stderr` (mass in “small” intervals k ≤ 20)
  * `acc` (acceptance rate)

**Baselines / references**

* 4D reference: `ref_sets = 80`, `ref_pairs = 12000`
* KR baseline: enabled

  * `kr_sets = 60`, `kr_p = 0.5`

---

## 2) Brief description of Protocol C implementation

Protocol C evaluates whether the BD-weighted coordinate-MCMC ensemble suppresses **non-manifold** microstructure, using **interval statistics**.

For each MCMC sample causal set (C):

1. Sample `sample_pairs` comparable pairs ((y \prec x)).
2. For each pair compute open-interval size:
   [
   k = |(y,x)| = \text{popcount}(\mathrm{reach}[y] ,&, \mathrm{past}[x]).
   ]
3. Build a capped histogram up to `k_cap` (with tail rebinning).
4. Compare this histogram to two references:

   * **4D sprinkling reference** → compute `JS4`
   * **KR (non-manifold) reference** → compute `JSKR`
5. Also compute `p20 = P(k \le 20)` from the same histogram.
6. Aggregate across recorded samples/chains and report ESS-adjusted standard errors and acceptance.

Interpretation:

* **Success**: `JS4` is minimized (or stays low) in the same β window where Protocol A yields (d_{MM}\approx 4), and `JSKR` stays **much larger** than `JS4` (strong separation from non-manifold baselines).
* **Over-constraining**: at large β, `JS4` tends to rise and acceptance falls.

---

## 3) Commands used to produce the results

You ran **one job per β** (one core per β) using GNU Parallel, for each N.

### N=350

```zsh
BETAS=(0.001 0.002 0.003 0.0035 0.004 0.0042 0.0044 0.005 0.006 0.007 0.01)

parallel -j 10 --lb \
  "python protocol_c.py \
    --Ns 350 --betas {1} \
    --chains 5 --burn 1000 --steps 16000 --thin 80 \
    --sample_pairs 12000 --k_cap 220 \
    --ref_sets 80 --ref_pairs 12000 \
    --with_kr --kr_sets 60 --kr_p 0.5 \
    --mix_local 0.5 --local_sigma 0.03 \
    --kmax 3 --alpha '1,-9,16,-8' \
    --outdir protocolC_N350_prod/b{1}" \
  ::: $BETAS
```

### N=250

```zsh
BETAS=(0.001 0.002 0.003 0.0035 0.004 0.0042 0.0044 0.005 0.006 0.007 0.01)

parallel -j 10 --lb \
  "python protocol_c.py \
    --Ns 250 --betas {1} \
    --chains 5 --burn 1000 --steps 16000 --thin 80 \
    --sample_pairs 12000 --k_cap 220 \
    --ref_sets 80 --ref_pairs 12000 \
    --with_kr --kr_sets 60 --kr_p 0.5 \
    --mix_local 0.5 --local_sigma 0.03 \
    --kmax 3 --alpha '1,-9,16,-8' \
    --outdir protocolC_N250_prod/b{1}" \
  ::: $BETAS
```

### N=180

```zsh
BETAS=(0.001 0.002 0.003 0.0035 0.004 0.0042 0.0044 0.005 0.006 0.007 0.01)

parallel -j 10 --lb \
  "python protocol_c.py \
    --Ns 180 --betas {1} \
    --chains 5 --burn 1000 --steps 16000 --thin 80 \
    --sample_pairs 12000 --k_cap 220 \
    --ref_sets 80 --ref_pairs 12000 \
    --with_kr --kr_sets 60 --kr_p 0.5 \
    --mix_local 0.5 --local_sigma 0.03 \
    --kmax 3 --alpha '1,-9,16,-8' \
    --outdir protocolC_N180_prod/b{1}" \
  ::: $BETAS
```


