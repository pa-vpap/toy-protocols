Below are **copy/paste-ready commands** that use **10 CPU cores** to reproduce your Protocol A results. I’m assuming your updated `a_gpt.py` supports:

* `--fix_density --Nref --Tref`
* `--chains`
* `--mix_local`
* `--steps --burn --thin`
* `--seed`
* comma-separated `--Ns` and `--betas`

If your script has a `--jobs` / `--nproc` flag, use that too; if not, we parallelize externally (recommended).

---

# A) Reproduce the key “fixed density” scans (β window)

## 1) Baseline sanity check (β=0, all N) — 10 cores not needed but included

```bash
python a_gpt.py \
  --kmax 3 --alpha "1,-9,16,-8" \
  --betas 0.0 \
  --Ns 120,180,250,350 \
  --chains 5 \
  --mix_local 0.5 \
  --steps 16000 --burn 1000 --thin 40 \
  --fix_density --Nref 120 --Tref 1.0 \
  --seed 1234
```

---

## 2) Main “plateau window” scan (β = 0.001…0.010) at fixed density

This is the clean reproduction scan that shows “green starts at 0.001”.

```bash
python a_gpt.py \
  --kmax 3 --alpha "1,-9,16,-8" \
  --betas 0.001,0.002,0.003,0.0035,0.004,0.005,0.006,0.007,0.01 \
  --Ns 120,180,250,350 \
  --chains 5 \
  --mix_local 0.5 \
  --steps 16000 --burn 1000 --thin 40 \
  --fix_density --Nref 120 --Tref 1.0 \
  --seed 2222
```

This will take a while because it’s a full grid.

---

# B) **True 10-core parallel** reproduction (fast + robust)

This runs **each β in parallel** (10 concurrent jobs).
It’s the most reliable way to use 10 cores even if `a_gpt.py` is single-process.

### 0) Create an output folder

```bash
mkdir -p runs
```

### 1) Run the β grid in parallel (10 cores)

macOS has `xargs -P 10` built in:

```bash
printf "%s\n" 0.001 0.002 0.003 0.0035 0.004 0.005 0.006 0.007 0.01 | \
xargs -I{} -P 10 bash -lc '
  python a_gpt.py \
    --kmax 3 --alpha "1,-9,16,-8" \
    --betas {} \
    --Ns 120,180,250,350 \
    --chains 5 \
    --mix_local 0.5 \
    --steps 16000 --burn 1000 --thin 40 \
    --fix_density --Nref 120 --Tref 1.0 \
    --seed 2222 \
  | tee runs/beta_{}.log
'
```

This gives you one log per β in `runs/`.

---

# C) Reproduce your “high-N tighter chain” checks (N=350, longer chain)

These are the runs you used to pinpoint where the drop begins.

### β=0.003, 0.005, 0.007 at N=350 (longer chains)

```bash
python a_gpt.py \
  --kmax 3 --alpha "1,-9,16,-8" \
  --betas 0.003,0.005,0.007 \
  --Ns 350 \
  --chains 5 \
  --mix_local 0.7 \
  --steps 32000 --burn 2000 --thin 40 \
  --fix_density --Nref 120 --Tref 1.0 \
  --seed 3333
```

### Extra points around the transition

```bash
python a_gpt.py \
  --kmax 3 --alpha "1,-9,16,-8" \
  --betas 0.004,0.005,0.006,0.007,0.01 \
  --Ns 350 \
  --chains 5 \
  --mix_local 0.7 \
  --steps 32000 --burn 2000 --thin 40 \
  --fix_density --Nref 120 --Tref 1.0 \
  --seed 1111
```

---

# D) Parallelize the long N=350 transition scan (10 cores)

```bash
printf "%s\n" 0.003 0.0035 0.004 0.005 0.006 0.007 0.01 | \
xargs -I{} -P 10 bash -lc '
  python a_gpt.py \
    --kmax 3 --alpha "1,-9,16,-8" \
    --betas {} \
    --Ns 350 \
    --chains 5 \
    --mix_local 0.7 \
    --steps 32000 --burn 2000 --thin 40 \
    --fix_density --Nref 120 --Tref 1.0 \
    --seed 1111 \
  | tee runs/N350_beta_{}.log
'
```

---

