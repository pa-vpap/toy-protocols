import math
import numpy as np
import pytest

# import from your module
import a_gpt as M


@pytest.mark.parametrize("d", [1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0])
def test_r_of_d_monotone_decreasing(d):
    # r(d) should decrease with d
    r = M._r_of_d(d)
    r2 = M._r_of_d(d + 0.5)
    assert r2 < r


@pytest.mark.parametrize("d", [1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
def test_invert_ordering_fraction_roundtrip(d):
    r = M._r_of_d(d)
    d_hat = M.invert_ordering_fraction_to_dimension(r, d_min=1.01, d_max=30.0)
    assert abs(d_hat - d) < 1e-4


def test_invert_ordering_fraction_clamps():
    # extreme values should clamp safely without throwing
    d1 = M.invert_ordering_fraction_to_dimension(0.0)
    d2 = M.invert_ordering_fraction_to_dimension(1.0)
    assert math.isfinite(d1)
    assert math.isfinite(d2)
    assert d1 >= 1.01
    assert d2 <= 30.0
