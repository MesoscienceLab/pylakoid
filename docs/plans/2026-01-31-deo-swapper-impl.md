# DEO Swapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Deterministic Even-Odd (DEO) swapping for parallel tempering to achieve O(m) round-trip time.

**Architecture:** Add `_deo_sweep` helper function, `SwapDEO` class (conforms to `Swapper`), and `SwapDEOWithStats` class (conforms to `SwapperWithStats`). All existing code remains unchanged.

**Tech Stack:** JAX, Equinox, jaxtyping

---

## Task 1: Test DEO Stats Shape and Attempt Count

**Files:**
- Create: `tests/test_deo_swap.py`

**Step 1: Write the failing test**

Create `tests/test_deo_swap.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_deo_stats_shape_and_attempt_count -v`

Expected: FAIL with `AttributeError: module 'pylakoid.structure.anneal' has no attribute 'SwapDEOWithStats'`

**Step 3: Skip implementation (will be done in later task)**

**Step 4: Skip (test should fail at this point)**

**Step 5: Commit test file**

```bash
git add tests/test_deo_swap.py
git commit -m "test: add failing test for DEO stats shape and count

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Test DEO All Swaps Accepted (Equal Energies)

**Files:**
- Modify: `tests/test_deo_swap.py`

**Step 1: Add the failing test**

Append to `tests/test_deo_swap.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_deo_all_swaps_accepted_equal_energies -v`

Expected: FAIL (no SwapDEOWithStats yet)

**Step 3: Skip implementation**

**Step 4: Skip**

**Step 5: Commit**

```bash
git add tests/test_deo_swap.py
git commit -m "test: add failing test for DEO 100% acceptance with equal energies

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Test DEO No Swaps Accepted (Unfavorable Conditions)

**Files:**
- Modify: `tests/test_deo_swap.py`

**Step 1: Add the failing test**

Append to `tests/test_deo_swap.py`:

```python
def test_deo_no_swaps_accepted_unfavorable():
    """With unfavorable energy ordering and low temp, no swaps accepted."""
    num_replicas = 5

    center_of_mass = jnp.zeros((num_replicas, 2, 2))
    angle = jnp.zeros((num_replicas, 2))
    radius = jnp.ones((num_replicas, 2))
    protein_type = jnp.zeros((num_replicas, 2), dtype=jnp.int32)
    multi_membrane = anneal.MultiMembrane(center_of_mass, angle, radius, protein_type)

    # Ascending energy with descending temperature = very unfavorable
    # p_accept = exp((1/kBT[j] - 1/kBT[j+1]) * (E[j] - E[j+1]))
    # With E[j] < E[j+1] and kBT[j] > kBT[j+1], the exponent is negative
    energy = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    kBT = jnp.array([0.001, 0.0008, 0.0006, 0.0004, 0.0002])

    key = jax.random.key(456)
    swapper = anneal.SwapDEOWithStats()

    _, _, swap_stats = swapper(multi_membrane, energy, kBT, key)

    # All pairs should have 1 attempt and 0 successes
    assert jnp.all(swap_stats[:, 0] == 1)
    assert jnp.all(swap_stats[:, 1] == 0)
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_deo_no_swaps_accepted_unfavorable -v`

Expected: FAIL

**Step 3: Skip implementation**

**Step 4: Skip**

**Step 5: Commit**

```bash
git add tests/test_deo_swap.py
git commit -m "test: add failing test for DEO 0% acceptance with unfavorable conditions

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Test DEO Swap Correctness

**Files:**
- Modify: `tests/test_deo_swap.py`

**Step 1: Add the failing test**

Append to `tests/test_deo_swap.py`:

```python
def test_deo_swap_correctness():
    """Verify membrane data actually swaps positions when accepted."""
    num_replicas = 4

    # Create distinguishable membranes - each has unique center_of_mass
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

    # Equal energies = all swaps accepted
    energy = jnp.array([50.0, 50.0, 50.0, 50.0])
    kBT = jnp.array([1000.0, 800.0, 600.0, 400.0])

    key = jax.random.key(789)
    swapper = anneal.SwapDEOWithStats()

    new_membrane, new_energy, _ = swapper(multi_membrane, energy, kBT, key)

    # With all swaps accepted in DEO:
    # Even sweep: swap (0,1) and (2,3) -> [1,0,3,2]
    # Odd sweep: swap (1,2) -> [1,3,0,2]
    # Final order should be: original replicas [1, 3, 0, 2]

    # Check that membranes have been permuted (not in original order)
    # At minimum, position 0 should not have the [0,0] membrane
    assert not jnp.allclose(new_membrane.center_of_mass[0], center_of_mass[0])
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_deo_swap_correctness -v`

Expected: FAIL

**Step 3: Skip implementation**

**Step 4: Skip**

**Step 5: Commit**

```bash
git add tests/test_deo_swap.py
git commit -m "test: add failing test for DEO swap correctness

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Test DEO Parity Coverage

**Files:**
- Modify: `tests/test_deo_swap.py`

**Step 1: Add the failing test**

Append to `tests/test_deo_swap.py`:

```python
def test_deo_parity_coverage():
    """Both even and odd pairs are attempted in a single DEO call."""
    # 6 replicas = 5 pairs
    # Even pairs: 0, 2, 4 (indices in stats array)
    # Odd pairs: 1, 3
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

    # All 5 pairs should have exactly 1 attempt
    assert swap_stats.shape == (5, 2)
    assert jnp.all(swap_stats[:, 0] == 1)

    # Specifically check even pairs (0, 2, 4) and odd pairs (1, 3)
    even_attempts = swap_stats[0, 0] + swap_stats[2, 0] + swap_stats[4, 0]
    odd_attempts = swap_stats[1, 0] + swap_stats[3, 0]
    assert even_attempts == 3
    assert odd_attempts == 2
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_deo_parity_coverage -v`

Expected: FAIL

**Step 3: Skip implementation**

**Step 4: Skip**

**Step 5: Commit**

```bash
git add tests/test_deo_swap.py
git commit -m "test: add failing test for DEO parity coverage

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Test SwapDEO (without stats) Returns 2-Tuple

**Files:**
- Modify: `tests/test_deo_swap.py`

**Step 1: Add the failing test**

Append to `tests/test_deo_swap.py`:

```python
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

    # Should return exactly 2 elements (like Swapper type)
    assert len(result) == 2
    new_membrane, new_energy = result
    assert isinstance(new_membrane, anneal.MultiMembrane)
    assert new_energy.shape == (num_replicas,)
```

**Step 2: Run test to verify it fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_swap_deo_returns_two_tuple -v`

Expected: FAIL

**Step 3: Skip implementation**

**Step 4: Skip**

**Step 5: Commit**

```bash
git add tests/test_deo_swap.py
git commit -m "test: add failing test for SwapDEO 2-tuple return

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Implement _deo_sweep Helper Function

**Files:**
- Modify: `src/pylakoid/structure/anneal.py`

**Step 1: Write the failing test**

Already have failing tests from Tasks 1-6.

**Step 2: Verify tests still fail**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py -v`

Expected: All 6 tests FAIL

**Step 3: Implement _deo_sweep**

Add after the `SwapAdjacentRandomlyWithStats` class (around line 829) in `anneal.py`:

```python
def _deo_sweep(
    multi_membrane: MultiMembrane,
    energy: Float[Array, " m"],
    kBT: Float[Array, " m"],
    key: Key[Array, ""],
    parity: int,
) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, " n_pairs"], Bool[Array, " n_pairs"]]:
    """
    Attempt swaps on all pairs with given parity.

    Parameters:
        multi_membrane: The MultiMembrane to swap.
        energy: Energies of each replica.
        kBT: Temperature of each replica.
        key: JAX random key.
        parity: 0 for even pairs (0-1, 2-3, ...), 1 for odd pairs (1-2, 3-4, ...).

    Returns:
        multi_membrane: Updated membrane after swaps.
        energy: Updated energies after swaps.
        pair_indices: Which pair indices were attempted.
        accepted: Boolean mask of which pairs were accepted.
    """
    n_replicas = energy.shape[0]
    pair_indices = jnp.arange(parity, n_replicas - 1, 2)
    n_pairs = pair_indices.shape[0]

    # Handle empty case (e.g., 2 replicas with odd parity)
    if n_pairs == 0:
        return multi_membrane, energy, pair_indices, jnp.array([], dtype=jnp.bool_)

    # Compute acceptance probabilities for all pairs at once
    j = pair_indices
    log_p_accept = (1 / kBT[j] - 1 / kBT[j + 1]) * (energy[j] - energy[j + 1])

    rands = jax.random.uniform(key, (n_pairs,))
    accepted = (log_p_accept > 0) | (rands < jnp.exp(log_p_accept))

    # Apply accepted swaps via fori_loop
    def apply_swap(
        i: Int[Array, ""], state: tuple[MultiMembrane, Float[Array, " m"]]
    ) -> tuple[MultiMembrane, Float[Array, " m"]]:
        membrane, e = state
        ji = pair_indices[i]

        def swap(x: Float[Array, "m ..."]) -> Float[Array, "m ..."]:
            return x.at[ji].set(x[ji + 1]).at[ji + 1].set(x[ji])

        return typing.cast(
            tuple[MultiMembrane, Float[Array, " m"]],
            jax.lax.cond(
                accepted[i],
                lambda: (jax.tree.map(swap, membrane), swap(e)),
                lambda: (membrane, e),
            ),
        )

    multi_membrane, energy = typing.cast(
        tuple[MultiMembrane, Float[Array, " m"]],
        jax.lax.fori_loop(0, n_pairs, apply_swap, (multi_membrane, energy)),
    )
    return multi_membrane, energy, pair_indices, accepted
```

**Step 4: Verify tests still fail (need SwapDEO classes)**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py -v`

Expected: Still FAIL (no SwapDEO classes yet)

**Step 5: Commit**

```bash
git add src/pylakoid/structure/anneal.py
git commit -m "feat: add _deo_sweep helper for DEO swapping

Implements the core sweep logic that attempts swaps on all pairs
with a given parity (even or odd) in a single pass.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Implement SwapDEO Class

**Files:**
- Modify: `src/pylakoid/structure/anneal.py`

**Step 1: Tests already exist**

**Step 2: Verify test_swap_deo_returns_two_tuple still fails**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_swap_deo_returns_two_tuple -v`

Expected: FAIL

**Step 3: Implement SwapDEO**

Add after `_deo_sweep` in `anneal.py`:

```python
class SwapDEO(eqx.Module):
    """
    Deterministic Even-Odd swapper for O(m) round-trip time.

    Each call performs one full DEO cycle:
    1. Attempt all even-indexed pairs: (0,1), (2,3), (4,5), ...
    2. Attempt all odd-indexed pairs: (1,2), (3,4), (5,6), ...

    This achieves linear O(m) round-trip time vs O(m²) for random selection.
    """

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"]]:
        """
        Perform one full DEO swap cycle.

        Parameters:
            multi_membrane: The MultiMembrane in which replicas may be swapped.
            energy: The energies of the replicas.
            kBT: The energy scales of the replicas.
            key: A JAX random key.

        Returns:
            MultiMembrane: The resulting MultiMembrane after swapping.
            Float[Array, " m"]: The resulting energies after swapping.
        """
        k_even, k_odd = jax.random.split(key)
        multi_membrane, energy, _, _ = _deo_sweep(
            multi_membrane, energy, kBT, k_even, parity=0
        )
        multi_membrane, energy, _, _ = _deo_sweep(
            multi_membrane, energy, kBT, k_odd, parity=1
        )
        return multi_membrane, energy
```

**Step 4: Run test to verify it passes**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py::test_swap_deo_returns_two_tuple -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/pylakoid/structure/anneal.py
git commit -m "feat: add SwapDEO class

Implements Swapper interface with deterministic even-odd swapping
for O(m) round-trip time.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Implement SwapDEOWithStats Class

**Files:**
- Modify: `src/pylakoid/structure/anneal.py`

**Step 1: Tests already exist**

**Step 2: Verify remaining tests still fail**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py -v`

Expected: test_swap_deo_returns_two_tuple PASSES, others FAIL

**Step 3: Implement SwapDEOWithStats**

Add after `SwapDEO` in `anneal.py`:

```python
class SwapDEOWithStats(eqx.Module):
    """
    Deterministic Even-Odd swapper that returns per-pair swap statistics.

    Each call performs one full DEO cycle:
    1. Attempt all even-indexed pairs: (0,1), (2,3), (4,5), ...
    2. Attempt all odd-indexed pairs: (1,2), (3,4), (5,6), ...

    Each pair is attempted exactly once per call.
    """

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
        """
        Perform one full DEO swap cycle and return statistics.

        Parameters:
            multi_membrane: The MultiMembrane in which replicas may be swapped.
            energy: The energies of the replicas.
            kBT: The energy scales of the replicas.
            key: A JAX random key.

        Returns:
            MultiMembrane: The resulting MultiMembrane after swapping.
            Float[Array, " m"]: The resulting energies after swapping.
            Int[Array, "m_minus_1 2"]: Per-pair stats. Column 0 is attempts (always 1),
                column 1 is successes (0 or 1).
        """
        n_pairs_total = energy.shape[0] - 1
        stats = jnp.zeros((n_pairs_total, 2), dtype=jnp.int32)

        k_even, k_odd = jax.random.split(key)

        # Even sweep
        multi_membrane, energy, even_pairs, even_accepted = _deo_sweep(
            multi_membrane, energy, kBT, k_even, parity=0
        )
        stats = stats.at[even_pairs, 0].add(1)
        stats = stats.at[even_pairs, 1].add(even_accepted.astype(jnp.int32))

        # Odd sweep
        multi_membrane, energy, odd_pairs, odd_accepted = _deo_sweep(
            multi_membrane, energy, kBT, k_odd, parity=1
        )
        stats = stats.at[odd_pairs, 0].add(1)
        stats = stats.at[odd_pairs, 1].add(odd_accepted.astype(jnp.int32))

        return multi_membrane, energy, stats
```

**Step 4: Run all tests to verify they pass**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/test_deo_swap.py -v`

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/pylakoid/structure/anneal.py
git commit -m "feat: add SwapDEOWithStats class

Implements SwapperWithStats interface with deterministic even-odd
swapping. Each pair is attempted exactly once per call, with stats
tracking attempts and successes.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/ -v`

Expected: All tests PASS (including existing swap_stats tests)

**Step 2: Run type checker**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/`

Expected: No errors (or only pre-existing ones)

**Step 3: Commit if any fixes needed**

If fixes were needed, commit them.

---

## Task 11: Update Module Exports (if needed)

**Files:**
- Check: `src/pylakoid/structure/__init__.py`

**Step 1: Verify exports**

Check if `__init__.py` explicitly exports classes. If so, add `SwapDEO` and `SwapDEOWithStats`.

**Step 2: Test import**

Run: `/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -c "from pylakoid.structure.anneal import SwapDEO, SwapDEOWithStats; print('OK')"`

Expected: `OK`

**Step 3: Commit if changes needed**

---

## Summary

After completing all tasks:
- `SwapDEO` class: drop-in replacement for `SwapAdjacentRandomly`
- `SwapDEOWithStats` class: drop-in replacement for `SwapAdjacentRandomlyWithStats`
- `_deo_sweep` helper: shared logic for even/odd sweeps
- All existing tests still pass
- 6 new DEO-specific tests pass
