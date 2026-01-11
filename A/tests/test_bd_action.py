import numpy as np
import pytest
import a_gpt as M


def test_bd_action_on_total_chain_known_counts():
    # Total chain of length 4: 0≺1≺2≺3, with transitive closure
    N = 4
    reach = [0] * N
    # reach[0] = {1,2,3}, reach[1]={2,3}, reach[2]={3}
    reach[0] = (1 << 1) | (1 << 2) | (1 << 3)
    reach[1] = (1 << 2) | (1 << 3)
    reach[2] = (1 << 3)
    reach[3] = 0
    past = M.build_past_from_reach(N, reach)
    C = M.CausalSet(N=N, reach=reach, past=past)

    # Count open intervals:
    # Pairs (y,x) with y≺x:
    # (0,1): k=0
    # (0,2): k=1 (just {1})
    # (0,3): k=2 ({1,2})
    # (1,2): k=0
    # (1,3): k=1 ({2})
    # (2,3): k=0
    # So N0=3, N1=2, N2=1, others 0.
    alpha = np.array([10.0, 100.0, 1000.0], dtype=float)
    S = M.bd_action_interval_abundances(C, alpha=alpha, k_max=2)
    expected = 10.0 * 3 + 100.0 * 2 + 1000.0 * 1
    assert abs(S - expected) < 1e-9


def test_bd_action_kmax_truncates():
    # Same chain as above, but k_max=0 only counts k=0
    N = 4
    reach = [0] * N
    reach[0] = (1 << 1) | (1 << 2) | (1 << 3)
    reach[1] = (1 << 2) | (1 << 3)
    reach[2] = (1 << 3)
    past = M.build_past_from_reach(N, reach)
    C = M.CausalSet(N=N, reach=reach, past=past)

    alpha = np.array([1.0], dtype=float)
    S = M.bd_action_interval_abundances(C, alpha=alpha, k_max=0)
    # N0 should be 3
    assert abs(S - 3.0) < 1e-9
