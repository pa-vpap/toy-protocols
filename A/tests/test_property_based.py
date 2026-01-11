import numpy as np
import pytest

import a_gpt as M

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st


@given(
    dim=st.integers(min_value=2, max_value=6),
    T=st.floats(min_value=0.2, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_inside_alexandrov_origin_always_inside(dim, T):
    pt = np.zeros(dim)
    assert M.inside_alexandrov(pt, dim, float(T))


@given(
    N=st.integers(min_value=20, max_value=120),
    dim=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=25)
def test_sprinkle_all_inside(N, dim):
    rng = np.random.default_rng(123)
    T = 1.0
    pts = M.sprinkle_diamond(int(N), int(dim), T, rng)
    assert all(M.inside_alexandrov(pts[i], int(dim), T) for i in range(int(N)))
