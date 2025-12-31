Here’s a **clear, bias-check–oriented explanation** of the `--action-type` options in **Protocol C (Toy Model)**, written so it’s obvious what each one is testing and *why it matters*.

---

## What `--action-type` Controls

`--action-type` selects the **action functional**
[
S(C)
]
used in the Boltzmann weight
[
w(C;\beta) \propto e^{-\beta S(C)}.
]

This is the *only* place where “geometry preference” can enter.
Everything else (sampling, TwoNN, prevalence) is diagnostic.

If Protocol C is not biased, **changing the action must change the outcome** in the expected way.

---

## 1. `knn_graph`  (geometry-sensitive / positive control)

```bash
python gpt_protocol_c_v2.py --action-type knn_graph
```

**Definition**

* Normalize cloud to unit RMS radius
* Build full distance matrix
* For each point, find its `k` nearest neighbors
* Return the **mean kNN edge length**

[
S_{\text{kNN}} = \langle d_{k\text{NN}} \rangle
]

**Why it matters**

* Low-dimensional manifolds pack points more locally
* Entropic Gaussian clouds spread neighbors farther apart
* This action *should* distinguish “sprinkled-like” from entropic

**Expected behavior (and observed)**

* `S(sprinkled) < S(entropic)`
* Strong negative correlation `Corr(S, label) ≈ −1`
* As β increases:

  * `p_sprinkled(β)` → 1
  * `mean_id(β)` → intrinsic dimension (~5)
* **This is the intended success case**

**Interpretation**

> Protocol C *can* select manifold-like structure **if** the action encodes local geometric order.

---

## 2. `pair_dist`  (negative control / bias detector)

```bash
python gpt_protocol_c_v2.py --action-type pair_dist --sanity-twonn
```

**Definition (v2)**

* Normalize cloud to unit RMS radius
* Randomly sample pairs `(i,j)`
* Compute **mean squared distance**

[
S_{\text{pair}} = \langle |x_i - x_j|^2 \rangle
]

**Why squared distance matters**

* For centered, RMS-normalized clouds:
  [
  \mathbb{E}|x_i - x_j|^2 \approx 2
  ]
  independent of geometry or dimension.

This makes it a **theoretically flat action**.

**Expected behavior (and observed)**

* `S(entropic) ≈ S(sprinkled)`
* `Corr(S, label) ≈ 0`
* `p_sprinkled(β) ≈ 0.5` for all β
* `mean_id(β)` stays in Ω-mixture regime (~15–17)
* No drift, no selection

**Interpretation**

> Confirms **no hidden bias** in Boltzmann sampling, TwoNN, or normalization.

If Protocol C were cheating, this would still drift.
It does not.

---

## 3. `mst`  (global connectivity / optional test)

```bash
python gpt_protocol_c_v2.py --action-type mst
```

**Definition**

* Normalize cloud
* Compute full distance matrix
* Build Minimum Spanning Tree (Prim’s algorithm)
* Return **mean MST edge length**

[
S_{\text{MST}} = \langle \ell_{\text{MST}} \rangle
]

**Why it matters**

* Captures *global* connectivity rather than local packing
* Less sensitive than kNN to small-scale clustering
* Still geometry-aware, but weaker

**Expected behavior**

* Moderate separation between ensembles
* Slower or weaker drift with β
* Useful as an intermediate check

**Interpretation**

> Shows Protocol C responds smoothly to *strength* of geometric signal.

---

## Summary Table

| Action      | Geometry-aware | Expected Drift | Purpose                       |
| ----------- | -------------- | -------------- | ----------------------------- |
| `knn_graph` | ✅ Strong       | Yes            | Positive control              |
| `pair_dist` | ❌ None         | No             | Negative control / bias check |
| `mst`       | ⚠️ Moderate    | Weak           | Global-structure probe        |

---

## Bottom Line

Your results demonstrate:

* **Protocol C is not biased**
* Selection occurs **iff** the action encodes geometry
* TwoNN drift tracks ensemble change, not hard-coded labels
* Negative control (`pair_dist`) behaves exactly as theory predicts

This is exactly what a *sound* toy Protocol C should do.
