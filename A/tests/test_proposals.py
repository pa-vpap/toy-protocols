import numpy as np
import pytest
import a_gpt as M


@pytest.mark.parametrize("proposal", ["resample", "local"])
def test_propose_coords_preserves_shape_and_sort(proposal):
    rng = np.random.default_rng(123)
    N, dim, T = 200, 4, 1.0
    coords = M.sprinkle_diamond(N, dim, T, rng)

    out = M.propose_coords(
        coords=coords,
        dim=dim,
        T=T,
        rng=rng,
        proposal=proposal,
        local_sigma=0.03,
    )

    assert out.shape == coords.shape
    assert np.all(np.diff(out[:, 0]) >= 0)
    # all points should remain inside diamond
    assert all(M.inside_alexandrov(out[i], dim, T) for i in range(N))
