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

import inspect

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


# ---------------------------------------------------------------------------
# sample_swap respects swap eligibility
# ---------------------------------------------------------------------------
#
# Bug: sample_swap built swap_eligible = jnp.where(swappable[pt1, types],
# size=N)[0] but then never used it: index2 was sampled in [0, n_eligible)
# and used DIRECTLY as a position index. This only happens to be correct
# when the eligible positions are contiguous starting at 0 (e.g.
# MembranePipeline puts all PSII proteins first, so for a PSII-pt1 the
# eligible PSII positions are exactly 0..N_PSII-1). Any other layout
# silently lets sample_swap pick ineligible positions, producing an
# illegal type swap that the validity checker (which only verifies
# boundary / position constraints, not type compatibility) lets through.
#
# Fix: index2 = swap_eligible[k].


def _always_accept_checker(com, angle, pt, index, out_sharding=None):
    """Stub Checker that approves every move (any 2-arg signature consistent
    with the Checker callable contract is fine)."""
    return jnp.array(True)


def _zero_force_field(com1, angle1, pt1, com2, angle2, pt2):
    """Stub ForceField returning zero so dE is always 0 and acceptance
    only depends on swap-eligibility logic, not Metropolis."""
    return jnp.array(0.0)


def test_sample_swap_only_picks_eligible_positions():
    """4-protein membrane alternating PSII (type 0) and LHCII (type 1).
    swappable[0, 0] = True only — so the only legal swap is PSII<->PSII,
    which is a no-op on protein_type (identical types). After many
    sample_swap iterations, protein_type at every position must be unchanged.

    The broken code samples index2 in [0, n_eligible) and uses it as a
    direct position index. With eligible positions [0, 2], n_eligible=2 ->
    index2 ∈ {0, 1}. Picking index2=1 dereferences the LHCII at position 1,
    so the broken code performs an illegal PSII<->LHCII swap that the
    validity check (no type compatibility test) lets through. This test
    catches that by asserting type layout is preserved.
    """
    num_proteins = 4
    com = jnp.zeros((num_proteins, 2))
    angle = jnp.zeros(num_proteins)
    radius = jnp.ones(num_proteins)
    original_protein_type = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    membrane = anneal.Membrane(com, angle, radius, original_protein_type)

    # swappable: only PSII (type 0) can swap with itself.
    swappable = (
        jnp.zeros((2, 2), dtype=jnp.bool_)
        .at[0, 0].set(True)
    )

    kBT = jnp.array(1.0)

    state = membrane
    n_iter = 100
    for k in range(n_iter):
        state, _ = anneal.sample_swap(
            state,
            swappable,
            _zero_force_field,
            _always_accept_checker,
            kBT,
            jax.random.key(k),
            initial_energy=jnp.array(0.0),
        )

    assert jnp.array_equal(state.protein_type, original_protein_type), (
        f"Types changed: started {original_protein_type.tolist()}, "
        f"ended {state.protein_type.tolist()}. An illegal type swap "
        "occurred — sample_swap is using the raw randint(k) as a position "
        "instead of swap_eligible[k]."
    )


# ---------------------------------------------------------------------------
#
# Bug: sample_swap gated validity with `jnp.any(swap_eligible)`. swap_eligible
# is built by jnp.where(..., size=N), which pads unused slots with 0. When the
# ONLY eligible swap partner for the chosen index1 sits at position 0,
# swap_eligible is all zeros, so jnp.any(...) is False and a legitimate swap is
# rejected. This silently biases the sampler against swaps whose partner is
# particle 0.
#
# Fix: track n_eligible = jnp.sum(swappable[pt1, types]) and gate validity on
# (n_eligible > 0); also keep the randint maxval safe via maximum(n_eligible, 1).


def test_sample_swap_accepts_swap_when_only_partner_is_index_0():
    """2-protein membrane, types [0, 1], with only the cross pair 0<->1
    swappable. Every step has dE == 0 (zero force field) and an always-accept
    checker, so a correct sample_swap accepts a swap on EVERY iteration,
    whichever protein is chosen as index1 (the two positions always hold
    different types, so every accepted swap is observable in protein_type).

    For whichever index1 is the protein whose sole eligible partner sits at
    position 0, swap_eligible == [0, 0]; the buggy `jnp.any(swap_eligible)`
    gate evaluates False and that swap is wrongly rejected. The buggy code
    therefore swaps on only ~half the steps; the fixed code swaps on all of
    them. Asserting a swap on every step isolates the gate bug.
    """
    num_proteins = 2
    com = jnp.zeros((num_proteins, 2))
    angle = jnp.zeros(num_proteins)
    radius = jnp.ones(num_proteins)
    protein_type = jnp.array([0, 1], dtype=jnp.int32)
    membrane = anneal.Membrane(com, angle, radius, protein_type)

    # Only the cross pair is swappable (no self-swaps), so for the type whose
    # partner is the type at position 0, the sole eligible position is 0.
    swappable = (
        jnp.zeros((2, 2), dtype=jnp.bool_)
        .at[0, 1].set(True)
        .at[1, 0].set(True)
    )
    kBT = jnp.array(1.0)

    n_iter = 50
    state = membrane
    swaps_observed = 0
    for k in range(n_iter):
        new_state, _ = anneal.sample_swap(
            state,
            swappable,
            _zero_force_field,
            _always_accept_checker,
            kBT,
            jax.random.key(k),
            initial_energy=jnp.array(0.0),
        )
        if not jnp.array_equal(new_state.protein_type, state.protein_type):
            swaps_observed += 1
        state = new_state

    assert swaps_observed == n_iter, (
        f"Expected a swap on every one of {n_iter} steps (dE=0, always-accept "
        f"checker), but only {swaps_observed} occurred. sample_swap is gating "
        "validity on jnp.any(swap_eligible), which is False when the sole "
        "eligible partner is position 0 (swap_eligible padded with zeros)."
    )


def _signature_param_names(cls) -> list[str]:
    """Return the (non-self) parameter names of cls.__init__ in declaration order."""
    return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]


def _annotation_field_names(cls) -> list[str]:
    """Return the names of cls's annotated class-level attributes in declaration
    order (the order they appear in the class body, which Python preserves in
    __annotations__ as of 3.7+)."""
    return list(cls.__annotations__.keys())


# ---------------------------------------------------------------------------
# Membrane / MultiMembrane attribute order matches __init__ order
# ---------------------------------------------------------------------------
#
# Both Membrane and MultiMembrane previously declared attributes as
#   (center_of_mass, angle, protein_type, radius)
# while __init__ took
#   (center_of_mass, angle, radius, protein_type)
#
# Equinox modules tie the class-level attribute order to the pytree
# flatten order, so a mismatch is a footgun for any code that
# reconstructs a Membrane via positional fields (e.g. eqx.tree_at by
# index, jax.tree.unflatten with reordered leaves, or dataclasses-style
# iteration). All existing callers use the __init__ positional order
# and so were unaffected, but the inconsistency was a latent hazard.


def test_membrane_attribute_order_matches_init_order():
    """Membrane: declared attribute order == __init__ parameter order."""
    init_order = _signature_param_names(anneal.Membrane)
    decl_order = _annotation_field_names(anneal.Membrane)
    assert init_order == decl_order, (
        f"__init__ params {init_order} disagree with declared attributes "
        f"{decl_order} — eqx pytree leaf order will not match the constructor."
    )


def test_multi_membrane_attribute_order_matches_init_order():
    """MultiMembrane: declared attribute order == __init__ parameter order."""
    init_order = _signature_param_names(anneal.MultiMembrane)
    decl_order = _annotation_field_names(anneal.MultiMembrane)
    assert init_order == decl_order, (
        f"__init__ params {init_order} disagree with declared attributes "
        f"{decl_order} — eqx pytree leaf order will not match the constructor."
    )


def test_membrane_pytree_round_trip_preserves_fields():
    """A Membrane that is flattened and unflattened via jax.tree.flatten must
    survive with every field intact. If the eqx-declared attribute order
    differs from __init__ order, the pytree leaves come out in declared
    order while unflatten passes them positionally to __init__ — silently
    swapping radius and protein_type. Catches the bug at runtime."""
    p = 4
    com = jnp.arange(2 * p, dtype=jnp.float32).reshape(p, 2)
    angle = jnp.full((p,), 0.5, dtype=jnp.float32)
    radius = jnp.full((p,), 42.0, dtype=jnp.float32)
    protein_type = jnp.array([0, 1, 2, 0], dtype=jnp.int32)

    m = anneal.Membrane(com, angle, radius, protein_type)
    leaves, treedef = jax.tree.flatten(m)
    m2 = jax.tree.unflatten(treedef, leaves)

    assert jnp.array_equal(m2.center_of_mass, com)
    assert jnp.array_equal(m2.angle, angle)
    assert jnp.array_equal(m2.radius, radius)
    assert jnp.array_equal(m2.protein_type, protein_type)
    # And specifically that radius and protein_type weren't transposed:
    assert m2.radius.dtype == jnp.float32
    assert m2.protein_type.dtype == jnp.int32


def test_multi_membrane_pytree_round_trip_preserves_fields():
    """Same pytree round-trip check for MultiMembrane."""
    m, p = 3, 4
    com = jnp.arange(m * p * 2, dtype=jnp.float32).reshape(m, p, 2)
    angle = jnp.full((m, p), 0.5, dtype=jnp.float32)
    radius = jnp.full((m, p), 42.0, dtype=jnp.float32)
    protein_type = jnp.zeros((m, p), dtype=jnp.int32)

    mm = anneal.MultiMembrane(com, angle, radius, protein_type)
    leaves, treedef = jax.tree.flatten(mm)
    mm2 = jax.tree.unflatten(treedef, leaves)

    assert jnp.array_equal(mm2.center_of_mass, com)
    assert jnp.array_equal(mm2.angle, angle)
    assert jnp.array_equal(mm2.radius, radius)
    assert jnp.array_equal(mm2.protein_type, protein_type)
    assert mm2.radius.dtype == jnp.float32
    assert mm2.protein_type.dtype == jnp.int32


def test_sample_swap_preserves_type_counts():
    """Even when types differ, the count of each type in the membrane must
    be exactly preserved by sample_swap (a swap is a permutation of types).
    Tests on a 6-protein membrane with three types, where only two of them
    are swappable, and the swappable positions are non-contiguous."""
    num_proteins = 6
    com = jnp.zeros((num_proteins, 2))
    angle = jnp.zeros(num_proteins)
    radius = jnp.ones(num_proteins)
    # Types: A=0, B=1, C=2. Layout: [A, B, A, C, B, A]. Counts: A=3, B=2, C=1.
    # Only A<->A is swappable -> no-op on counts even if real swap mechanics fire.
    original_protein_type = jnp.array([0, 1, 0, 2, 1, 0], dtype=jnp.int32)
    membrane = anneal.Membrane(com, angle, radius, original_protein_type)

    swappable = (
        jnp.zeros((3, 3), dtype=jnp.bool_)
        .at[0, 0].set(True)
    )

    state = membrane
    for k in range(50):
        state, _ = anneal.sample_swap(
            state,
            swappable,
            _zero_force_field,
            _always_accept_checker,
            jnp.array(1.0),
            jax.random.key(k),
            initial_energy=jnp.array(0.0),
        )

    # Type counts preserved.
    for t in range(3):
        assert int(jnp.sum(state.protein_type == t)) == int(
            jnp.sum(original_protein_type == t)
        ), f"Count of type {t} changed."
    # And position layout preserved (since only no-op swaps are legal).
    assert jnp.array_equal(state.protein_type, original_protein_type)
