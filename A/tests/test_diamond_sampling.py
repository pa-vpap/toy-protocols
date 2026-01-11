import numpy as np
import pytest
import a_gpt as M


@pytest.mark.parametrize("dim", [2, 3, 4, 6])
def test_inside_alexandrov_center_is_inside(dim):
    T = 1.0
    pt = np.zeros(dim)
    assert M.inside_alexandrov(pt, dim, T)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_inside_alexandrov_outside_time_is_outside(dim):
    T = 1.0
    half = T / 2
    pt = np.zeros(dim)
    pt[0] = half + 1e-6
    assert not M.inside_alexandrov(pt, dim, T)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_sample_point_in_diamond_returns_inside(dim):
    rng = np.random.default_rng(123)
    T = 1.0
    for _ in range(200):
        pt = M.sample_point_in_diamond(dim, T, rng)
        assert M.inside_alexandrov(pt, dim, T)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_sprinkle_sorted_by_time(dim):
    rng = np.random.default_rng(123)
    T = 1.0
    N = 200
    pts = M.sprinkle_diamond(N, dim, T, rng)
    assert np.all(np.diff(pts[:, 0]) >= 0)
    # all inside
    assert all(M.inside_alexandrov(pts[i], dim, T) for i in range(N))
