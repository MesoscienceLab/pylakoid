# ABOUTME: Tests for DEO (Deterministic Even-Odd) swapper.
# ABOUTME: Verifies SwapDEO achieves O(m) round-trip via systematic even/odd sweeps.

import jax
import jax.numpy as jnp

from pylakoid.structure import anneal


def test_deo_stats_shape_and_attempt_count():
    """DEO stats has shape (m-1, 2) with exactly one attempt per pair."""
    num_replicas = 5  # 4 pairs: 0-1, 1-2, 2-3, 3-4

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.array([100.0, 80.0, 60.0, 40.0, 20.0])
    kBT = jnp.array([1000.0, 800.0, 600.0, 400.0, 200.0])

    key = jax.random.key(42)
    swapper = anneal.SwapDEOWithStats()

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # Shape should be (num_replicas - 1, 2)
    assert swap_stats.shape == (num_replicas - 1, 2)

    # Each pair should have exactly 1 attempt (DEO does one full cycle)
    assert jnp.all(swap_stats[:, 0] == 1)

    # Successes should be 0 or 1 for each pair
    assert jnp.all(swap_stats[:, 1] <= swap_stats[:, 0])


def test_deo_all_swaps_accepted_equal_energies():
    """With equal energies, all swaps should be accepted."""
    num_replicas = 5

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    # Equal energies -> log_p_accept = 0 -> p_accept = 1 -> always accept
    energy = jnp.array([50.0, 50.0, 50.0, 50.0, 50.0])
    kBT = jnp.array([1000.0, 800.0, 600.0, 400.0, 200.0])

    key = jax.random.key(123)
    swapper = anneal.SwapDEOWithStats()

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # All pairs should have 1 attempt and 1 success
    assert jnp.all(swap_stats[:, 0] == 1)
    assert jnp.all(swap_stats[:, 1] == 1)


def test_deo_no_swaps_accepted_unfavorable():
    """With unfavorable energy ordering and low temp, no swaps accepted."""
    num_replicas = 5

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    # Ascending energy with ascending temperature = very unfavorable
    # p_accept = exp((1/kBT[j] - 1/kBT[j+1]) * (E[j] - E[j+1]))
    # With ascending kBT: 1/kBT[j] > 1/kBT[j+1], so difference is positive
    # With ascending E: E[j] < E[j+1], so difference is negative
    # Product is negative -> p_accept << 1 -> swaps rejected
    energy = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    kBT = jnp.array([0.001, 0.002, 0.003, 0.004, 0.005])

    key = jax.random.key(456)
    swapper = anneal.SwapDEOWithStats()

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # All pairs should have 1 attempt and 0 successes
    assert jnp.all(swap_stats[:, 0] == 1)
    assert jnp.all(swap_stats[:, 1] == 0)


def test_deo_swap_correctness():
    """Verify membrane data actually swaps positions when accepted."""
    num_replicas = 4

    center_of_mass = jnp.array([
        [[0.0, 0.0], [0.0, 0.0]],  # replica 0
        [[1.0, 1.0], [1.0, 1.0]],  # replica 1
        [[2.0, 2.0], [2.0, 2.0]],  # replica 2
        [[3.0, 3.0], [3.0, 3.0]],  # replica 3
    ])
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.array([50.0, 50.0, 50.0, 50.0])
    kBT = jnp.array([1000.0, 800.0, 600.0, 400.0])

    key = jax.random.key(789)
    swapper = anneal.SwapDEOWithStats()

    new_membrane, new_energy, _ = swapper(multi_membrane, energy, kBT, key)

    # Check that membranes have been permuted (not in original order)
    assert not jnp.allclose(new_membrane.center_of_mass[0], center_of_mass[0])


def test_deo_parity_coverage():
    """Both even and odd pairs are attempted in a single DEO call."""
    num_replicas = 6

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.ones(num_replicas) * 50.0
    kBT = jnp.linspace(1000.0, 200.0, num_replicas)

    key = jax.random.key(999)
    swapper = anneal.SwapDEOWithStats()

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    assert swap_stats.shape == (5, 2)
    assert jnp.all(swap_stats[:, 0] == 1)

    even_attempts = swap_stats[0, 0] + swap_stats[2, 0] + swap_stats[4, 0]
    odd_attempts = swap_stats[1, 0] + swap_stats[3, 0]
    assert even_attempts == 3
    assert odd_attempts == 2


def test_swap_deo_returns_two_tuple():
    """SwapDEO (without stats) returns 2-tuple like SwapAdjacentRandomly."""
    num_replicas = 4

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.array([100.0, 80.0, 60.0, 40.0])
    kBT = jnp.array([1000.0, 800.0, 600.0, 400.0])

    key = jax.random.key(42)
    swapper = anneal.SwapDEO()

    result = swapper(multi_membrane, energy, kBT, key)

    assert len(result) == 2
    new_membrane, new_energy = result
    assert isinstance(new_membrane, anneal.MultiMembrane)
    assert new_energy.shape == (num_replicas,)
