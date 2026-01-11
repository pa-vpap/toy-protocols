import math
import numpy as np
import pytest

import a_gpt as M


def test_T_scaling_fixed_density_ratio():
    """
    For fixed density rho, Volume ~ T^d, so N ~ rho T^d => T ~ (N/rho)^(1/d).
    Ratios should satisfy:
      T(N2)/T(N1) = (N2/N1)^(1/d)
    """
    dim = 4
    N1, N2 = 120, 350
    T1 = 1.0

    # The CLI code uses Nref/Tref to infer rho; emulate that.
    # Use the module's fixed-density helper so any volume prefactor cancels.
    if not hasattr(M, "diamond_volume") or not hasattr(M, "T_for_fixed_density"):
      pytest.skip("Expose diamond_volume(dim,T) and T_for_fixed_density(dim,N,rho) to test density scaling.")

    rho = N1 / M.diamond_volume(dim, T1)
    T2 = M.T_for_fixed_density(dim, N2, rho)

    expected_ratio = (N2 / N1) ** (1.0 / dim)
    assert abs((T2 / T1) - expected_ratio) < 1e-10
