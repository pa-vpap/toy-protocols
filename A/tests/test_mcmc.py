import numpy as np
import pytest
import a_gpt as M


def test_mcmc_acceptance_in_bounds():
    rng = np.random.default_rng(123)
    out = M.run_coordinate_mcmc(
        N=80,
        dim=4,
        T=1.0,
        beta=0.003,
        alpha=np.array([1.0, -9.0, 16.0, -8.0]),
        k_max=3,
        burn_in=200,
        steps=800,
        thin=20,
        proposal="resample",
        local_sigma=0.03,
        rng=rng,
        progress=False,
    )
    assert 0.0 <= out.accept_rate <= 1.0
    assert out.num_samples > 0
    assert np.isfinite(out.mean_dmm)
    assert np.isfinite(out.var_dmm)


def test_beta_zero_accepts_all():
    rng = np.random.default_rng(123)
    out = M.run_coordinate_mcmc(
        N=60,
        dim=4,
        T=1.0,
        beta=0.0,
        alpha=np.array([1.0, -9.0, 16.0, -8.0]),
        k_max=3,
        burn_in=200,
        steps=600,
        thin=20,
        proposal="resample",
        local_sigma=0.03,
        rng=rng,
        progress=False,
    )
    # For beta=0, Metropolis accepts always
    assert abs(out.accept_rate - 1.0) < 1e-12


def test_reproducibility_same_seed():
    # Same seed + same args => identical (or extremely close) aggregate outputs
    # Because RNG drives everything deterministically.
    alpha = np.array([1.0, -9.0, 16.0, -8.0])
    out1 = M.run_coordinate_mcmc(
        N=60, dim=4, T=1.0, beta=0.003,
        alpha=alpha, k_max=3,
        burn_in=200, steps=800, thin=20,
        proposal="resample", local_sigma=0.03,
        rng=np.random.default_rng(777),
        progress=False,
    )
    out2 = M.run_coordinate_mcmc(
        N=60, dim=4, T=1.0, beta=0.003,
        alpha=alpha, k_max=3,
        burn_in=200, steps=800, thin=20,
        proposal="resample", local_sigma=0.03,
        rng=np.random.default_rng(777),
        progress=False,
    )
    assert out1.mean_dmm == out2.mean_dmm
    assert out1.var_dmm == out2.var_dmm
    assert out1.accept_rate == out2.accept_rate
    assert out1.num_samples == out2.num_samples
