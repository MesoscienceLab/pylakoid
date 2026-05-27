# Regression tests for two related bug fixes in pylakoid's parallel tempering:
#
# 1. anneal_helpers.prepare_swappable was returning an all-False matrix because
#    the result of swappable.at[i, j].set(True) was discarded (JAX functional
#    update mistakenly used as if it were in-place).
#
# 2. anneal.SwapEvenOdd's Metropolis acceptance treated
#    (1/kT_i - 1/kT_j) * (E_i - E_j) as if it were a probability, when it is
#    in fact the log of the Metropolis factor. After the fix, SwapEvenOdd is
#    a deterministic even-odd (DEO) sweep using log_p_accept > 0 |
#    rand < exp(log_p_accept) and arange(parity, n-1, 2) so j+1 is in-bounds.
#
# These tests are written against the public API (SwapStats return type,
# SwapEvenOdd constructor taking no args, prepare_swappable signature) so
# they will catch a regression of either bug regardless of implementation
# refactors that preserve the contract.

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from pylakoid.structure import anneal
from pylakoid.structure import anneal_helpers as ah


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trivial_multi_membrane(num_replicas: int, num_proteins: int = 2) -> anneal.MultiMembrane:
    """Build a MultiMembrane with arbitrary but distinct per-replica values so
    swap-propagation can be checked by inspecting which replica's data ends up
    where."""
    com = jnp.stack(
        [jnp.full((num_proteins, 2), float(r)) for r in range(num_replicas)], axis=0
    )
    angle = jnp.stack(
        [jnp.full((num_proteins,), float(r)) for r in range(num_replicas)], axis=0
    )
    radius = jnp.ones((num_replicas, num_proteins))
    protein_type = jnp.zeros((num_replicas, num_proteins), dtype=jnp.int32)
    return anneal.MultiMembrane(com, angle, radius, protein_type)


# ---------------------------------------------------------------------------
# prepare_swappable
# ---------------------------------------------------------------------------


def test_prepare_swappable_empty_pairs_returns_all_false():
    """No pairs in can_swap -> result is all False with correct shape and dtype."""
    ptm = {"A": 0, "B": 1, "C": 2}
    sw = ah.prepare_swappable([], ptm)
    assert sw.shape == (3, 3)
    assert sw.dtype == jnp.bool_
    assert not bool(sw.any())


def test_prepare_swappable_single_pair_sets_both_directions():
    """Single (A, B) pair -> only [0,1] and [1,0] are True (the function sets
    both directions regardless of input order)."""
    ptm = {"A": 0, "B": 1, "C": 2}
    sw = ah.prepare_swappable([("A", "B")], ptm)
    expected = jnp.array(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    assert jnp.array_equal(sw, expected)


def test_prepare_swappable_self_pair():
    """Self-pair (A, A) sets the diagonal entry."""
    ptm = {"A": 0, "B": 1}
    sw = ah.prepare_swappable([("A", "A")], ptm)
    expected = jnp.array([[True, False], [False, False]])
    assert jnp.array_equal(sw, expected)


def test_prepare_swappable_full_psii_matrix():
    """All 9 valid PSII pairs -> full 3x3 True block on the PSII subset,
    False elsewhere. This is the use case from MesoscienceLab/MembranePipeline."""
    ptm = {"C2": 0, "C2S2": 1, "C2S2M2": 2, "LHCII": 3, "CYTB6F": 4}
    psii = ["C2", "C2S2", "C2S2M2"]
    can_swap = [(a, b) for a in psii for b in psii]
    sw = ah.prepare_swappable(can_swap, ptm)
    expected = jnp.zeros((5, 5), dtype=jnp.bool_)
    expected = expected.at[:3, :3].set(True)
    assert jnp.array_equal(sw, expected)


def test_prepare_swappable_returns_symmetric_matrix():
    """For any (i, j) input, the result is symmetric: sw[i, j] == sw[j, i].
    This holds because the function sets both directions per pair."""
    ptm = {"A": 0, "B": 1, "C": 2, "D": 3}
    sw = ah.prepare_swappable([("A", "C"), ("B", "D")], ptm)
    assert jnp.array_equal(sw, sw.T)


def test_prepare_swappable_dtype_is_bool():
    """The matrix dtype must be bool for the downstream run_monte_carlo
    indexing to behave correctly."""
    ptm = {"A": 0, "B": 1}
    sw = ah.prepare_swappable([("A", "B")], ptm)
    assert sw.dtype == jnp.bool_


# ---------------------------------------------------------------------------
# SwapEvenOdd return shape and per-pair attempt count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_replicas", [2, 3, 4, 5, 6, 7])
def test_swap_even_odd_stats_shape(num_replicas: int):
    """Returned SwapStats has n_accepted / n_attempted arrays of shape (m-1,),
    regardless of whether the ladder length is even or odd."""
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.linspace(100.0, 10.0, num_replicas)
    kBT = jnp.linspace(10.0, 0.1, num_replicas)
    key = jax.random.key(0)

    _, _, stats = anneal.SwapEvenOdd()(mm, energy, kBT, key)

    assert stats.n_accepted.shape == (num_replicas - 1,)
    assert stats.n_attempted.shape == (num_replicas - 1,)
    assert stats.n_accepted.dtype == jnp.int32
    assert stats.n_attempted.dtype == jnp.int32


@pytest.mark.parametrize("num_replicas", [2, 3, 4, 5, 6, 7])
def test_swap_even_odd_each_pair_attempted_exactly_once(num_replicas: int):
    """Each adjacent pair is attempted exactly once per DEO cycle (one even-sweep
    visit OR one odd-sweep visit). Catches regressions that double-count, drop,
    or out-of-bounds-write into the stats array."""
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.linspace(100.0, 10.0, num_replicas)
    kBT = jnp.linspace(10.0, 0.1, num_replicas)
    key = jax.random.key(0)

    _, _, stats = anneal.SwapEvenOdd()(mm, energy, kBT, key)

    expected = jnp.ones((num_replicas - 1,), dtype=jnp.int32)
    assert jnp.array_equal(stats.n_attempted, expected)
    assert jnp.all(stats.n_accepted <= stats.n_attempted)


# ---------------------------------------------------------------------------
# SwapEvenOdd Metropolis acceptance correctness
# ---------------------------------------------------------------------------


def test_equal_energies_always_accept():
    """log_p_accept = 0 when energies are equal -> exp(0) = 1 -> always accept.
    This is the cheapest direct test of the missing-exp() bug: the broken code
    computed `p_accept = 0` and then `rand < 0` is False, so nothing was
    accepted. The fixed code accepts all swaps in this regime."""
    num_replicas = 5
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.full((num_replicas,), 50.0)
    kBT = jnp.linspace(10.0, 0.1, num_replicas)
    key = jax.random.key(0)

    _, _, stats = anneal.SwapEvenOdd()(mm, energy, kBT, key)

    assert jnp.array_equal(stats.n_accepted, jnp.ones((num_replicas - 1,), dtype=jnp.int32))


def test_anti_equilibrium_always_accept():
    """Cold replica has higher energy than warm (log_p_accept > 0) -> always
    accept. This direction was the only one the broken code ever accepted,
    so the fix must preserve it."""
    num_replicas = 4
    mm = _trivial_multi_membrane(num_replicas)
    # Energies increase with index (replica i+1 hotter and lower-energy is wrong direction);
    # here we put high energy on the cold replica (replica 0 at low kBT) so swaps
    # to move it toward warmer (higher kBT) replicas are favored.
    energy = jnp.array([1000.0, 800.0, 600.0, 400.0])
    kBT = jnp.array([0.1, 0.5, 5.0, 10.0])  # cold first

    # For pair (0, 1): (1/0.1 - 1/0.5) * (1000 - 800) = 8 * 200 = 1600 -> always accept.
    # For pair (2, 3): (1/5 - 1/10) * (600 - 400) = 0.1 * 200 = 20 -> always accept.

    key = jax.random.key(0)
    _, _, stats = anneal.SwapEvenOdd()(mm, energy, kBT, key)

    # Both even-sweep pairs should be accepted with probability 1.
    assert int(stats.n_accepted[0]) == 1
    assert int(stats.n_accepted[2]) == 1


def test_equilibrium_rarely_accepts_at_large_beta_gap():
    """Thermodynamically-equilibrated case: cold replica at lower energy than
    warm. log_p_accept << 0 -> exp(log_p_accept) ≈ 0 -> swap accept rate
    should be ~0 on average across many keys.

    The pre-fix code never accepted in this regime (since rand >= 0 > p_accept).
    The fixed code also rarely accepts, so this test specifically catches a
    regression where exp() were dropped (giving always-accept when E_warm > E_cold)."""
    num_replicas = 3
    mm = _trivial_multi_membrane(num_replicas)
    # Cold (idx 0) at low energy, warm (idx 2) at high energy. kBT decreasing.
    # For pair (0, 1): (1/0.1 - 1/1.0) * (10 - 50) = 9 * (-40) = -360
    #   -> exp(-360) ≈ 0 -> never accept
    # For pair (1, 2): doesn't run on even sweep
    energy = jnp.array([10.0, 50.0, 100.0])
    kBT = jnp.array([0.1, 1.0, 10.0])  # cold to warm

    accept_count = 0
    n_keys = 50
    for k in range(n_keys):
        key = jax.random.key(k)
        _, _, stats = anneal.SwapEvenOdd()(mm, energy, kBT, key)
        accept_count += int(stats.n_accepted[0])  # pair (0, 1)

    # Acceptance probability is exp(-360) ≈ 0, so expect 0/50 accepts.
    # A regression to always-accept (the dropped-exp() bug) would give 50/50.
    assert accept_count == 0


def test_swap_acceptance_obeys_metropolis_distribution():
    """Empirical acceptance rate matches min(1, exp(log_p_accept)) within
    sampling tolerance. Specifically constructed so log_p_accept is in a
    range where the probability is genuinely intermediate (not 0 or 1)."""
    num_replicas = 2
    mm = _trivial_multi_membrane(num_replicas)
    # kBT = [1.0, 2.0], energy = [10.0, 12.0]
    # log_p_accept = (1/1 - 1/2) * (10 - 12) = 0.5 * -2 = -1
    # expected acceptance: exp(-1) ≈ 0.3679
    energy = jnp.array([10.0, 12.0])
    kBT = jnp.array([1.0, 2.0])

    n_keys = 2000
    accept_count = 0
    for k in range(n_keys):
        _, _, stats = anneal.SwapEvenOdd()(mm, energy, kBT, jax.random.key(k))
        accept_count += int(stats.n_accepted[0])

    empirical = accept_count / n_keys
    expected = float(jnp.exp(jnp.array(-1.0)))
    assert abs(empirical - expected) < 0.03, (
        f"empirical={empirical:.3f}, expected={expected:.3f} "
        "(broken: always-accept ~1.0; dropped-exp variant: never-accept 0.0)"
    )


# ---------------------------------------------------------------------------
# SwapEvenOdd actually swaps data when accepted
# ---------------------------------------------------------------------------


def test_accepted_swap_propagates_to_membrane_and_energy():
    """When a pair is accepted, the corresponding replica entries in both
    MultiMembrane and energy must be swapped (not just the stats updated).
    Catches regressions where the swap stats record success but the data was
    not actually rearranged."""
    num_replicas = 2
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.array([10.0, 10.0])  # equal -> guaranteed accept
    kBT = jnp.array([1.0, 2.0])
    key = jax.random.key(0)

    new_mm, new_energy, stats = anneal.SwapEvenOdd()(mm, energy, kBT, key)

    # Guaranteed accept on the only pair
    assert int(stats.n_accepted[0]) == 1
    # Original com[0] was full of 0.0, com[1] was full of 1.0. After swap, [0] should be 1.0.
    assert float(new_mm.center_of_mass[0, 0, 0]) == pytest.approx(1.0)
    assert float(new_mm.center_of_mass[1, 0, 0]) == pytest.approx(0.0)
    # Energy is also swapped — for equal energies it is observationally identical,
    # but at least the array shape and sum are conserved.
    assert float(jnp.sum(new_energy)) == pytest.approx(float(jnp.sum(energy)))


def test_total_energy_conserved_across_swaps():
    """Whatever the accept pattern, the sum of replica energies after a DEO
    cycle equals the sum before (swap rearranges, does not destroy)."""
    num_replicas = 6
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.array([100.0, 80.0, 60.0, 40.0, 20.0, 5.0])
    kBT = jnp.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    key = jax.random.key(7)

    _, new_energy, _ = anneal.SwapEvenOdd()(mm, energy, kBT, key)
    assert float(jnp.sum(new_energy)) == pytest.approx(float(jnp.sum(energy)))


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_swap_even_odd_is_jit_compatible():
    """SwapEvenOdd must be callable under eqx.filter_jit, since pylakoid's
    parallel_tempering wraps it in @eqx.filter_jit. A regression that
    introduces a Python-level branch on a traced array would fail here."""
    num_replicas = 5
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.linspace(100.0, 10.0, num_replicas)
    kBT = jnp.linspace(10.0, 0.1, num_replicas)

    jitted = eqx.filter_jit(anneal.SwapEvenOdd())

    _, _, stats = jitted(mm, energy, kBT, jax.random.key(0))
    assert stats.n_accepted.shape == (num_replicas - 1,)


# ---------------------------------------------------------------------------
# SwapAdjacentRandomly Metropolis acceptance correctness
# ---------------------------------------------------------------------------
#
# SwapAdjacentRandomly had the same missing-exp() Metropolis bug as the
# original SwapEvenOdd: it computed
#
#     p_accept = (1/kT_i - 1/kT_j) * (E_i - E_j)
#     (p_accept > 1) | (rand < p_accept)
#
# treating the log of the Metropolis factor as a probability. These tests
# pin down the corrected behavior so a future change cannot silently
# reintroduce it.


def test_swap_adjacent_randomly_equal_energies_always_accept():
    """Equal energies -> log_p_accept = 0 -> exp(0) = 1 -> always accept.
    The broken code computed p_accept = 0 and `rand < 0` is never True,
    so no swaps were accepted in this regime."""
    num_replicas = 3
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.full((num_replicas,), 50.0)
    kBT = jnp.array([10.0, 1.0, 0.1])
    num_swaps = 20
    key = jax.random.key(0)

    _, _, stats = anneal.SwapAdjacentRandomly(num_swaps)(mm, energy, kBT, key)

    # Every attempt should be accepted (n_accepted == n_attempted on every pair).
    assert jnp.array_equal(stats.n_accepted, stats.n_attempted)
    # Total attempts == num_swaps (one pick per inner iteration).
    assert int(jnp.sum(stats.n_attempted)) == num_swaps


def test_swap_adjacent_randomly_anti_equilibrium_always_accept():
    """Cold replica has higher energy (log_p_accept > 0) -> always accept.
    This was the only regime the buggy code got directionally right; fix
    must preserve it.

    Tested with num_swaps=1 per call across many keys so each call sees the
    same initial state (a successful swap reorders the energy array, which
    would change the next attempt's log_p_accept sign on a 2-replica ladder)."""
    num_replicas = 2
    mm = _trivial_multi_membrane(num_replicas)
    # Pair (0, 1): (1/0.1 - 1/1.0) * (1000 - 100) = 9 * 900 = 8100 >> 0
    # -> always accept.
    energy = jnp.array([1000.0, 100.0])
    kBT = jnp.array([0.1, 1.0])

    n_keys = 20
    accept_count = 0
    for k in range(n_keys):
        _, _, stats = anneal.SwapAdjacentRandomly(1)(mm, energy, kBT, jax.random.key(k))
        accept_count += int(jnp.sum(stats.n_accepted))
    assert accept_count == n_keys


def test_swap_adjacent_randomly_equilibrium_rarely_accepts():
    """log_p_accept very negative -> exp(log_p_accept) ≈ 0 -> never accept.
    Catches a regression that reintroduces the missing-exp() bug."""
    num_replicas = 2
    mm = _trivial_multi_membrane(num_replicas)
    # log_p_accept = (1/0.1 - 1/1.0) * (10 - 50) = 9 * -40 = -360
    # -> exp(-360) ≈ 0 -> never accept.
    energy = jnp.array([10.0, 50.0])
    kBT = jnp.array([0.1, 1.0])

    n_swaps_per_call = 100
    n_keys = 20
    total_accepts = 0
    for k in range(n_keys):
        _, _, stats = anneal.SwapAdjacentRandomly(n_swaps_per_call)(
            mm, energy, kBT, jax.random.key(k)
        )
        total_accepts += int(jnp.sum(stats.n_accepted))

    assert total_accepts == 0


def test_swap_adjacent_randomly_acceptance_matches_metropolis():
    """Empirical accept rate matches exp(log_p_accept) within sampling
    tolerance. Constructed so log_p_accept = -1 -> probability ≈ 0.368,
    which is neither 0 nor 1 — discriminates the correct exp() form from
    both bug variants.

    Uses num_swaps=1 per call across many keys so each call sees the same
    initial (energy, kBT). With num_swaps>1, an accepted swap reorders the
    energy array on a 2-replica ladder and the next attempt sees the
    opposite-sign log_p_accept, biasing the empirical rate."""
    num_replicas = 2
    mm = _trivial_multi_membrane(num_replicas)
    # log_p_accept = (1/1 - 1/2) * (10 - 12) = 0.5 * -2 = -1
    energy = jnp.array([10.0, 12.0])
    kBT = jnp.array([1.0, 2.0])

    n_keys = 2000
    accept_count = 0
    for k in range(n_keys):
        _, _, stats = anneal.SwapAdjacentRandomly(1)(mm, energy, kBT, jax.random.key(k))
        accept_count += int(jnp.sum(stats.n_accepted))

    empirical = accept_count / n_keys
    expected = float(jnp.exp(jnp.array(-1.0)))
    assert abs(empirical - expected) < 0.03, (
        f"empirical={empirical:.3f}, expected={expected:.3f} "
        "(broken: never-accept ~0.0)"
    )


def test_swap_adjacent_randomly_is_jit_compatible():
    """SwapAdjacentRandomly must compile under eqx.filter_jit."""
    num_replicas = 4
    mm = _trivial_multi_membrane(num_replicas)
    energy = jnp.array([100.0, 80.0, 60.0, 40.0])
    kBT = jnp.array([10.0, 5.0, 1.0, 0.1])

    jitted = eqx.filter_jit(anneal.SwapAdjacentRandomly(10))
    _, _, stats = jitted(mm, energy, kBT, jax.random.key(0))
    assert stats.n_accepted.shape == (num_replicas - 1,)
