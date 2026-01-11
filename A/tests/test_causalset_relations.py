import numpy as np
import pytest
import a_gpt as M


def bitset_iter(bits: int):
    while bits:
        lsb = bits & -bits
        j = lsb.bit_length() - 1
        yield j
        bits ^= lsb


def test_build_past_from_reach_consistency_small():
    # Make a tiny hand-crafted reach relation: 0≺1≺2, 0≺2
    N = 3
    reach = [0, 0, 0]
    reach[0] = (1 << 1) | (1 << 2)
    reach[1] = (1 << 2)
    past = M.build_past_from_reach(N, reach)

    # past[2] should contain 0 and 1
    assert (past[2] >> 0) & 1
    assert (past[2] >> 1) & 1
    assert not ((past[0] >> 1) & 1)


def test_causalset_from_coords_time_respects_order():
    rng = np.random.default_rng(123)
    pts = M.sprinkle_diamond(150, dim=4, T=1.0, rng=rng)
    C = M.causalset_from_coords(pts)

    # If i ≺ j then i < j because we time-sorted coordinates
    for i in range(C.N):
        for j in bitset_iter(C.reach[i]):
            assert i < j


def test_reach_past_are_mutual():
    rng = np.random.default_rng(123)
    pts = M.sprinkle_diamond(120, dim=4, T=1.0, rng=rng)
    C = M.causalset_from_coords(pts)

    for i in range(C.N):
        for j in bitset_iter(C.reach[i]):
            assert (C.past[j] >> i) & 1 == 1


def test_ordering_fraction_bounds():
    rng = np.random.default_rng(123)
    pts = M.sprinkle_diamond(80, dim=4, T=1.0, rng=rng)
    C = M.causalset_from_coords(pts)
    r = C.ordering_fraction()
    assert 0.0 <= r <= 1.0
