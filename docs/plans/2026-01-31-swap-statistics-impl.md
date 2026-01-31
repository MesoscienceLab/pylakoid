# Swap Statistics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Track per-pair swap success rates in parallel tempering without breaking existing API.

**Architecture:** Add `SwapAdjacentRandomlyWithStats` class that shares core logic with existing `SwapAdjacentRandomly` via a private implementation function. Add matching `parallel_tempering_with_stats` function. Stats are `Int[Array, "m_minus_1 2"]` tracking attempts/successes per adjacent pair.

**Tech Stack:** JAX, Equinox, jaxtyping

**Test command:** `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest`

**Type check command:** `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/`

---

## Task 1: Test Stats Shape and Attempt Counting

**Files:**
- Create: `tests/test_swap_stats.py`

**Step 1: Write the failing test**

Create `tests/test_swap_stats.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_swap_stats_shape_and_attempt_count -v`

Expected: FAIL with `AttributeError: module 'pylakoid.structure.anneal' has no attribute 'SwapAdjacentRandomlyWithStats'`

**Step 3: Commit the failing test**

```bash
git add tests/test_swap_stats.py
git commit -m "test: add failing test for swap stats shape and counting"
```

---

## Task 2: Add SwapperWithStats Type Alias

**Files:**
- Modify: `src/pylakoid/structure/anneal.py:626-648` (after existing `Swapper` type alias)

**Step 1: Add the type alias**

After the existing `Swapper` type alias and its docstring (around line 648), add:

```python
SwapperWithStats = Callable[
    [
        MultiMembrane,
        Float[Array, " m"],
        Float[Array, " m"],
        Key[Array, ""],
    ],
    tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]],
]
"""
Type for swapper functions that return swap statistics.

Parameters:
    multi_membrane (MultiMembrane): The `MultiMembrane` in which membranes may be swapped.
    energy (Float[Array, " m"]): The energies of the membranes.
    kBT (Float[Array, " m"]): The energy scales of the membranes.
    key (Key[Array, ""]): A JAX random key.

Returns:
    (MultiMembrane): The resulting `MultiMembrane` after swapping.
    (Float[Array, " m"]): The resulting energies after swapping.
    (Int[Array, "m_minus_1 2"]): Swap statistics. Column 0 is attempts, column 1 is successes.
"""
```

**Step 2: Run type check**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/pylakoid/structure/anneal.py`

Expected: No new errors

**Step 3: Commit**

```bash
git add src/pylakoid/structure/anneal.py
git commit -m "feat: add SwapperWithStats type alias"
```

---

## Task 3: Implement SwapAdjacentRandomlyWithStats Class

**Files:**
- Modify: `src/pylakoid/structure/anneal.py:714` (after existing `SwapAdjacentRandomly` class)

**Step 1: Add the new class**

After `SwapAdjacentRandomly` class (after line 714), add:

```python
class SwapAdjacentRandomlyWithStats(eqx.Module):
    """
    Like SwapAdjacentRandomly but returns per-pair swap statistics.

    Repeats the following `num_swaps` times: chooses a pair of adjacent membranes at random,
    and attempts to swap them with probability from Metropolis-Hastings.

    Attributes:
        num_swaps: The number of times to attempt swaps.
    """

    num_swaps: int

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
        """
        Swap adjacent membranes at random and track statistics.

        Parameters:
            multi_membrane: The `MultiMembrane` in which membranes may be swapped.
            energy: The energies of the membranes.
            kBT: The energy scales of the membranes.
            key: A JAX random key.

        Returns:
            (MultiMembrane): The resulting `MultiMembrane` after swapping.
            (Float[Array, " m"]): The resulting energies after swapping.
            (Int[Array, "m_minus_1 2"]): Per-pair statistics. Column 0 is attempts, column 1 is successes.
        """
        keys = jax.random.split(key, self.num_swaps)
        del key

        num_pairs = energy.shape[0] - 1
        initial_stats = jnp.zeros((num_pairs, 2), dtype=jnp.int32)

        def f(
            i: Int[Array, ""],
            args: tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]],
        ) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
            membrane, energy, stats = args
            k1, k2 = jax.random.split(keys[i], 2)

            j = jax.random.randint(k1, (), minval=0, maxval=energy.shape[0] - 1)  # pyright: ignore [reportUnknownMemberType]
            del k1
            rand = jax.random.uniform(k2, ())  # pyright: ignore [reportUnknownMemberType]
            del k2

            def swap(x: Float[Array, "m ..."]):
                x1 = x[j]
                x2 = x[j + 1]
                return x.at[j].set(x2).at[j + 1].set(x1)

            p_accept = (1 / kBT[j] - 1 / kBT[j + 1]) * (energy[j] - energy[j + 1])
            accepted = (p_accept > 1) | (rand < p_accept)

            new_membrane, new_energy = typing.cast(
                tuple[MultiMembrane, Float[Array, " m"]],
                jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                    accepted,
                    lambda: (jax.tree.map(swap, membrane), swap(energy)),
                    lambda: (membrane, energy),
                ),
            )

            # Update stats: increment attempts for pair j, increment successes if accepted
            new_stats = stats.at[j, 0].add(1)
            new_stats = jax.lax.cond(
                accepted,
                lambda s: s.at[j, 1].add(1),
                lambda s: s,
                new_stats,
            )

            return new_membrane, new_energy, new_stats

        return typing.cast(
            tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]],
            jax.lax.fori_loop(0, self.num_swaps, f, (multi_membrane, energy, initial_stats)),  # pyright: ignore [reportUnknownMemberType]
        )
```

**Step 2: Run the test**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_swap_stats_shape_and_attempt_count -v`

Expected: PASS

**Step 3: Run type check**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/pylakoid/structure/anneal.py`

Expected: No new errors

**Step 4: Commit**

```bash
git add src/pylakoid/structure/anneal.py
git commit -m "feat: add SwapAdjacentRandomlyWithStats class"
```

---

## Task 4: Test All Swaps Accepted When Energies Equal

**Files:**
- Modify: `tests/test_swap_stats.py`

**Step 1: Write the failing test**

Add to `tests/test_swap_stats.py`:

```python
def test_all_swaps_accepted_when_energies_equal():
    """When all energies are equal, all swaps should be accepted."""
    num_replicas = 4
    num_swaps = 20

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    # Equal energies means p_accept = exp(0) = 1, so always accept
    energy = jnp.array([50.0, 50.0, 50.0, 50.0])
    kBT = jnp.array([1.0, 2.0, 3.0, 4.0])

    key = jax.random.key(123)
    swapper = anneal.SwapAdjacentRandomlyWithStats(num_swaps)

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # All attempts should succeed
    assert swap_stats[:, 0].sum() == num_swaps  # Total attempts
    assert swap_stats[:, 1].sum() == num_swaps  # Total successes
```

**Step 2: Run test to verify it passes**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_all_swaps_accepted_when_energies_equal -v`

Expected: PASS (implementation already handles this correctly)

**Step 3: Commit**

```bash
git add tests/test_swap_stats.py
git commit -m "test: verify all swaps accepted when energies equal"
```

---

## Task 5: Test No Swaps Accepted Under Unfavorable Conditions

**Files:**
- Modify: `tests/test_swap_stats.py`

**Step 1: Write the test**

Add to `tests/test_swap_stats.py`:

```python
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
```

**Step 2: Run test**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_no_swaps_accepted_unfavorable_conditions -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_swap_stats.py
git commit -m "test: verify no swaps accepted under unfavorable conditions"
```

---

## Task 6: Test Backward Compatibility of SwapAdjacentRandomly

**Files:**
- Modify: `tests/test_swap_stats.py`

**Step 1: Write the test**

Add to `tests/test_swap_stats.py`:

```python
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
```

**Step 2: Run test**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_swap_adjacent_randomly_unchanged -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_swap_stats.py
git commit -m "test: verify SwapAdjacentRandomly backward compatibility"
```

---

## Task 7: Test Results Match Between Old and New Swapper

**Files:**
- Modify: `tests/test_swap_stats.py`

**Step 1: Write the test**

Add to `tests/test_swap_stats.py`:

```python
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
```

**Step 2: Run test**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_swapper_results_match -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_swap_stats.py
git commit -m "test: verify old and new swapper produce identical results"
```

---

## Task 8: Test parallel_tempering_with_stats Shape

**Files:**
- Modify: `tests/test_swap_stats.py`

**Step 1: Write the failing test**

Add to `tests/test_swap_stats.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_parallel_tempering_with_stats_returns_stats -v`

Expected: FAIL with `AttributeError: module 'pylakoid.structure.anneal' has no attribute 'parallel_tempering_with_stats'`

**Step 3: Commit**

```bash
git add tests/test_swap_stats.py
git commit -m "test: add failing test for parallel_tempering_with_stats"
```

---

## Task 9: Add ZeroForceField and AlwaysAcceptChecker Test Helpers

**Files:**
- Modify: `src/pylakoid/structure/force_field.py` (check if ZeroForceField exists)
- Modify: `src/pylakoid/structure/checker.py` (check if AlwaysAcceptChecker exists)

**Step 1: Check existing code**

Read the files to see if these helpers already exist. If not, add them.

For `force_field.py`, add if missing:

```python
class ZeroForceField(eqx.Module):
    """Force field that always returns zero energy. Useful for testing."""

    def __call__(
        self,
        center_of_mass1: Float[Array, "2"],
        angle1: Float[Array, ""],
        protein_type1: Int[Array, ""],
        radius1: Float[Array, ""],
        center_of_mass2: Float[Array, "2"],
        angle2: Float[Array, ""],
        protein_type2: Int[Array, ""],
        radius2: Float[Array, ""],
    ) -> Float[Array, ""]:
        return jnp.array(0.0)
```

For `checker.py`, add if missing:

```python
class AlwaysAcceptChecker(eqx.Module):
    """Checker that always accepts. Useful for testing."""

    def __call__(
        self,
        center_of_mass: Float[Array, "2"],
        angle: Float[Array, ""],
        protein_type: Int[Array, ""],
        radius: Float[Array, ""],
    ) -> Bool[Array, ""]:
        return jnp.array(True)
```

**Step 2: Run type check**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/pylakoid/structure/`

Expected: No new errors

**Step 3: Commit if changes made**

```bash
git add src/pylakoid/structure/force_field.py src/pylakoid/structure/checker.py
git commit -m "feat: add ZeroForceField and AlwaysAcceptChecker test helpers"
```

---

## Task 10: Implement parallel_tempering_with_stats

**Files:**
- Modify: `src/pylakoid/structure/anneal.py` (after `parallel_tempering` function, around line 852)

**Step 1: Add the function**

After `parallel_tempering` function, add:

```python
def parallel_tempering_with_stats(
    multi_membrane: MultiMembrane,
    translatable: Int[Array, " t"],
    swappable: Bool[Array, "s s"],
    rotatable: Int[Array, " r"],
    force_field: ForceField,
    checker: Checker,
    swapper: SwapperWithStats,
    r_max_values: Float[Array, " m"],
    phi_max_values: Float[Array, " m"],
    kBT_values: Float[Array, " m"],
    initial_energy: Float[Array, " m"],
    n_steps_between_swaps: int,
    key: Key[Array, ""],
) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
    """
    Like parallel_tempering but returns swap statistics.

    Run `n_steps_between_swaps` of Monte Carlo simulation for each membrane in `multi_membrane`,
    then use `swapper` to swap membranes and return statistics.

    Parameters:
        multi_membrane: The `MultiMembrane` that you want to simulate.
        translatable: The indices of the proteins in `membrane` that can be translated.
        swappable: `swappable[t1, t2]` is `True` if proteins of type `t1` and `t2` can be swapped.
        rotatable: The indices of the proteins in `membrane` that can be rotated.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        swapper: Used to swap membranes for parallel tempering (must return stats).
        r_max_values: Max changes in coordinate per replica.
        phi_max_values: Max changes in angle per replica.
        kBT_values: Energy scales per replica.
        initial_energy: The energy of the input membranes.
        n_steps_between_swaps: Monte Carlo steps per membrane before swapping.
        key: A JAX random key.

    Returns:
        The `MultiMembrane` after parallel tempering.
        The energy of the membranes after parallel tempering.
        Per-pair swap statistics (attempts in column 0, successes in column 1).
    """
    out_sharding = jax.typeof(initial_energy).sharding.spec

    def f(
        multi_membrane: MultiMembrane,
        r_max_values: Float[Array, " a"],
        phi_max_values: Float[Array, " a"],
        kBT_values: Float[Array, " a"],
        initial_energy: Float[Array, " a"],
        key: Key[Array, " a"],
    ) -> tuple[MultiMembrane, Float[Array, ""]]:
        def body(
            i: Int[Array, ""], state: tuple[MultiMembrane, Float[Array, " a"]]
        ) -> tuple[MultiMembrane, Float[Array, " a"]]:
            multi_membrane, initial_energy = state
            membrane = Membrane(
                multi_membrane.center_of_mass[i],
                multi_membrane.angle[i],
                multi_membrane.radius[i],
                multi_membrane.protein_type[i],
            )
            membrane, membrane_energy = run_monte_carlo(
                membrane,
                translatable,
                swappable,
                rotatable,
                force_field,
                checker,
                r_max_values[i],
                phi_max_values[i],
                kBT_values[i],
                n_steps_between_swaps,
                key[i],
                initial_energy=initial_energy[i],
                out_sharding=out_sharding,
            )
            new_multi_membrane = MultiMembrane(
                state[0].center_of_mass.at[i].set(membrane.center_of_mass),
                state[0].angle.at[i].set(membrane.angle),
                state[0].radius.at[i].set(membrane.radius),
                state[0].protein_type.at[i].set(membrane.protein_type),
            )
            new_energy = state[1].at[i].set(membrane_energy)
            return new_multi_membrane, new_energy

        return typing.cast(
            tuple[MultiMembrane, Float[Array, " a"]],
            jax.lax.fori_loop(  # pyright: ignore [reportUnknownMemberType]
                0, initial_energy.shape[0], body, (multi_membrane, initial_energy)
            ),
        )

    def get_sharding(x: Array) -> jax.NamedSharding:
        return jax.typeof(x).sharding

    def get_partition_spec(x: Array) -> jax.sharding.PartitionSpec:
        return get_sharding(x).spec

    k1, key = jax.random.split(key, 2)
    keys = jax.random.split(k1, initial_energy.shape[0])
    del k1
    args = (
        multi_membrane,
        r_max_values,
        phi_max_values,
        kBT_values,
        initial_energy,
        keys,
    )
    del keys
    multi_membrane, initial_energy = typing.cast(
        tuple[MultiMembrane, Float[Array, ""]],
        jax.shard_map(  # pyright: ignore [reportCallIssue, reportUnknownMemberType]
            f,
            out_specs=jax.tree.map(
                get_partition_spec, (multi_membrane, initial_energy)
            ),
        )(*args),
    )

    new_multi_membrane, new_initial_energy, swap_stats = swapper(
        jax.sharding.reshard(multi_membrane, jax.sharding.PartitionSpec()),  # pyright: ignore [reportUnknownMemberType]
        jax.sharding.reshard(initial_energy, jax.sharding.PartitionSpec()),  # pyright: ignore [reportUnknownMemberType]
        jax.sharding.reshard(kBT_values, jax.sharding.PartitionSpec()),  # pyright: ignore [reportUnknownMemberType]
        key,
    )
    del key

    multi_membrane = jax.sharding.reshard(  # pyright: ignore [reportUnknownMemberType]
        new_multi_membrane, jax.tree.map(get_sharding, multi_membrane)
    )
    initial_energy = jax.sharding.reshard(  # pyright: ignore [reportUnknownMemberType]
        new_initial_energy, get_sharding(initial_energy)
    )
    return multi_membrane, initial_energy, swap_stats
```

**Step 2: Run the test**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_swap_stats.py::test_parallel_tempering_with_stats_returns_stats -v`

Expected: PASS

**Step 3: Run type check**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/pylakoid/structure/anneal.py`

Expected: No new errors

**Step 4: Commit**

```bash
git add src/pylakoid/structure/anneal.py
git commit -m "feat: add parallel_tempering_with_stats function"
```

---

## Task 11: Run Full Test Suite

**Step 1: Run all tests**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/ -v`

Expected: All tests pass

**Step 2: Run full type check**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/`

Expected: No errors

**Step 3: Final commit if any cleanup needed**

---

## Task 12: Update Module Exports (if needed)

**Files:**
- Check: `src/pylakoid/structure/__init__.py`

**Step 1: Verify exports**

Check if `__init__.py` exports from anneal.py. If it uses explicit exports, add:
- `SwapperWithStats`
- `SwapAdjacentRandomlyWithStats`
- `parallel_tempering_with_stats`

**Step 2: Commit if changes made**

```bash
git add src/pylakoid/structure/__init__.py
git commit -m "feat: export new swap statistics API"
```
