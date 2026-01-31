# ABOUTME: Tests for parallel tempering swap statistics tracking.
# ABOUTME: Verifies SwapAdjacentRandomlyWithStats returns correct stats shape and counts.

import jax
import jax.numpy as jnp

from pylakoid.structure import anneal


def test_swap_stats_shape_and_attempt_count():
    """Stats array has shape (m-1, 2) and counts all attempts."""
    # 4 replicas, so 3 adjacent pairs
    num_replicas = 4
    num_swaps = 10

    # Create minimal MultiMembrane with 4 replicas, 2 proteins each
    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.array([100.0, 80.0, 60.0, 40.0])  # Descending energy
    kBT = jnp.array([1.0, 2.0, 3.0, 4.0])  # Ascending temperature

    key = jax.random.key(42)
    swapper = anneal.SwapAdjacentRandomlyWithStats(num_swaps)

    new_membrane, new_energy, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # Shape should be (num_replicas - 1, 2)
    assert swap_stats.shape == (num_replicas - 1, 2)

    # Total attempts should equal num_swaps
    assert swap_stats[:, 0].sum() == num_swaps

    # Successes should be <= attempts for each pair
    assert jnp.all(swap_stats[:, 1] <= swap_stats[:, 0])


def test_some_swaps_accepted_with_favorable_initial_ordering():
    """With favorable initial ordering, some swaps should be accepted."""
    num_replicas = 4
    num_swaps = 20

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    # Descending energy with ascending temperature = favorable for first swap
    # After each swap, energies are exchanged, so subsequent swaps of the same
    # pair have unfavorable ordering until swapped back
    energy = jnp.array([1e6, 1e5, 1e4, 1e3])
    kBT = jnp.array([0.001, 0.002, 0.003, 0.004])

    key = jax.random.key(123)
    swapper = anneal.SwapAdjacentRandomlyWithStats(num_swaps)

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # Total attempts should equal num_swaps
    assert swap_stats[:, 0].sum() == num_swaps

    # Some swaps should succeed (at least when ordering is favorable)
    # With these extreme energy differences, swaps occur only when favorable
    total_successes = swap_stats[:, 1].sum()
    assert total_successes > 0  # At least some swaps accepted
    assert total_successes <= num_swaps  # Can't have more successes than attempts


def test_no_swaps_accepted_unfavorable_conditions():
    """When energy ordering is unfavorable and temperature is very low, no swaps accepted."""
    num_replicas = 4
    num_swaps = 20

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    # Ascending energy (lower temp has higher energy = unfavorable)
    # With very low temperatures, p_accept will be essentially 0
    energy = jnp.array([10.0, 20.0, 30.0, 40.0])
    kBT = jnp.array([0.001, 0.002, 0.003, 0.004])  # Very low temperatures

    key = jax.random.key(456)
    swapper = anneal.SwapAdjacentRandomlyWithStats(num_swaps)

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # All attempts should fail (or nearly all - check for 0)
    assert swap_stats[:, 0].sum() == num_swaps  # Total attempts
    assert swap_stats[:, 1].sum() == 0  # No successes


def test_swap_adjacent_randomly_unchanged():
    """Original SwapAdjacentRandomly still returns 2-tuple and works correctly."""
    num_replicas = 4
    num_swaps = 10

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.array([100.0, 80.0, 60.0, 40.0])
    kBT = jnp.array([1.0, 2.0, 3.0, 4.0])

    key = jax.random.key(789)
    swapper = anneal.SwapAdjacentRandomly(num_swaps)

    result = swapper(multi_membrane, energy, kBT, key)

    # Should return exactly 2 elements
    assert len(result) == 2
    new_membrane, new_energy = result
    assert isinstance(new_membrane, anneal.MultiMembrane)
    assert new_energy.shape == (num_replicas,)


def test_swapper_results_match():
    """SwapAdjacentRandomly and SwapAdjacentRandomlyWithStats produce same membrane/energy."""
    num_replicas = 4
    num_swaps = 10

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    energy = jnp.array([100.0, 80.0, 60.0, 40.0])
    kBT = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Use same key for both
    key = jax.random.key(999)

    swapper_old = anneal.SwapAdjacentRandomly(num_swaps)
    swapper_new = anneal.SwapAdjacentRandomlyWithStats(num_swaps)

    membrane_old, energy_old = swapper_old(multi_membrane, energy, kBT, key)
    membrane_new, energy_new, _ = swapper_new(multi_membrane, energy, kBT, key)

    # Results should be identical
    assert jnp.allclose(energy_old, energy_new)
    assert jnp.allclose(membrane_old.center_of_mass, membrane_new.center_of_mass)
    assert jnp.allclose(membrane_old.angle, membrane_new.angle)


def test_parallel_tempering_with_stats_returns_stats():
    """parallel_tempering_with_stats returns swap statistics."""
    from pylakoid.structure.checker import AlwaysAcceptChecker
    from pylakoid.structure.force_field import ZeroForceField

    num_replicas = 3
    num_proteins = 2

    center_of_mass = jnp.zeros((num_replicas, num_proteins, 2))
    angle = jnp.zeros((num_replicas, num_proteins))
    radius = jnp.ones((num_replicas, num_proteins))
    protein_type = jnp.zeros((num_replicas, num_proteins), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    translatable = jnp.array([0, 1], dtype=jnp.int32)
    swappable = jnp.array([[True]], dtype=jnp.bool_)
    rotatable = jnp.array([0, 1], dtype=jnp.int32)

    r_max_values = jnp.array([0.1, 0.1, 0.1])
    phi_max_values = jnp.array([0.1, 0.1, 0.1])
    kBT_values = jnp.array([1.0, 2.0, 3.0])
    initial_energy = jnp.zeros(num_replicas)

    force_field = ZeroForceField()
    checker = AlwaysAcceptChecker()
    swapper = anneal.SwapAdjacentRandomlyWithStats(num_replicas)

    key = jax.random.key(42)

    # Set up mesh context for shard_map (use Explicit axis type)
    mesh = jax.make_mesh((1,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))

    def make_partition_spec(x):
        return jax.sharding.PartitionSpec("x", *([None] * (x.ndim - 1)))

    with jax.set_mesh(mesh):
        # Shard the data
        multi_membrane = jax.device_put(
            multi_membrane, jax.tree.map(make_partition_spec, multi_membrane)
        )
        r_max_values = jax.device_put(r_max_values, make_partition_spec(r_max_values))
        phi_max_values = jax.device_put(phi_max_values, make_partition_spec(phi_max_values))
        kBT_values = jax.device_put(kBT_values, make_partition_spec(kBT_values))
        initial_energy = jax.device_put(initial_energy, make_partition_spec(initial_energy))

        result = anneal.parallel_tempering_with_stats(
            multi_membrane,
            translatable,
            swappable,
            rotatable,
            force_field,
            checker,
            swapper,
            r_max_values,
            phi_max_values,
            kBT_values,
            initial_energy,
            n_steps_between_swaps=2,
            key=key,
        )

    assert len(result) == 3
    new_membrane, new_energy, swap_stats = result
    assert isinstance(new_membrane, anneal.MultiMembrane)
    assert new_energy.shape == (num_replicas,)
    assert swap_stats.shape == (num_replicas - 1, 2)
