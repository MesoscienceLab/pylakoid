# DEO Swapper Design

## Goal

Implement Deterministic Even-Odd (DEO) swapping for parallel tempering to achieve O(m) round-trip time instead of O(m²) with random selection. This is especially important for production runs with ~120 replicas.

## Background

The current `SwapAdjacentRandomly` class randomly selects one pair per call, leading to diffusive mixing with O(m²) expected round-trip time. DEO systematically alternates between even and odd pairs, creating a "conveyor belt" effect that achieves O(m) round-trip time.

For 120 replicas: ~14,400 cycles (random) vs ~240 cycles (DEO) = 60x improvement.

## Algorithm

A single DEO cycle consists of two sweeps:

1. **Even sweep**: Attempt swaps on pairs (0,1), (2,3), (4,5), ...
2. **Odd sweep**: Attempt swaps on pairs (1,2), (3,4), (5,6), ...

Within each sweep, pairs don't overlap, so acceptance decisions can be computed in parallel. Each pair is attempted exactly once per cycle.

## Implementation

### New Classes

```python
class SwapDEO(eqx.Module):
    """Deterministic Even-Odd swapper for O(m) round-trip time.

    Each call performs one full DEO cycle:
    1. Attempt all even-indexed pairs: (0,1), (2,3), (4,5), ...
    2. Attempt all odd-indexed pairs: (1,2), (3,4), (5,6), ...
    """

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"]]:
        ...


class SwapDEOWithStats(eqx.Module):
    """Like SwapDEO but returns per-pair swap statistics.

    Stats array shape: (m-1, 2) where column 0 is attempts, column 1 is successes.
    Each pair has exactly 1 attempt per call.
    """

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
        ...
```

### Shared Sweep Logic

```python
def _deo_sweep(
    multi_membrane: MultiMembrane,
    energy: Float[Array, " m"],
    kBT: Float[Array, " m"],
    key: Key[Array, ""],
    parity: int,  # 0 for even, 1 for odd
) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, " n_pairs"], Bool[Array, " n_pairs"]]:
    """Attempt swaps on all pairs with given parity.

    Returns:
        multi_membrane: Updated membrane after swaps
        energy: Updated energies after swaps
        pair_indices: Which pairs were attempted (for stats tracking)
        accepted: Boolean mask of which pairs were accepted
    """
    n_replicas = energy.shape[0]
    pair_indices = jnp.arange(parity, n_replicas - 1, 2)
    n_pairs = pair_indices.shape[0]

    # Compute acceptance probabilities for all pairs at once
    j = pair_indices
    log_p_accept = (1/kBT[j] - 1/kBT[j+1]) * (energy[j] - energy[j+1])

    rands = jax.random.uniform(key, (n_pairs,))
    accepted = (log_p_accept > 0) | (rands < jnp.exp(log_p_accept))

    # Apply accepted swaps via fori_loop
    def apply_swap(i, state):
        membrane, e = state
        ji = pair_indices[i]

        def swap(x):
            return x.at[ji].set(x[ji+1]).at[ji+1].set(x[ji])

        return jax.lax.cond(
            accepted[i],
            lambda: (jax.tree.map(swap, membrane), swap(e)),
            lambda: (membrane, e),
        )

    multi_membrane, energy = jax.lax.fori_loop(
        0, n_pairs, apply_swap, (multi_membrane, energy)
    )
    return multi_membrane, energy, pair_indices, accepted
```

### Statistics Tracking

In `SwapDEOWithStats.__call__`:

```python
n_pairs_total = energy.shape[0] - 1
stats = jnp.zeros((n_pairs_total, 2), dtype=jnp.int32)

# Even sweep
multi_membrane, energy, even_pairs, even_accepted = _deo_sweep(
    multi_membrane, energy, kBT, k_even, parity=0
)
stats = stats.at[even_pairs, 0].add(1)  # attempts
stats = stats.at[even_pairs, 1].add(even_accepted.astype(jnp.int32))  # successes

# Odd sweep
multi_membrane, energy, odd_pairs, odd_accepted = _deo_sweep(
    multi_membrane, energy, kBT, k_odd, parity=1
)
stats = stats.at[odd_pairs, 0].add(1)
stats = stats.at[odd_pairs, 1].add(odd_accepted.astype(jnp.int32))

return multi_membrane, energy, stats
```

## Temperature Indexing Convention

- Replica 0 = HOT (highest kBT)
- Replica m-1 = COLD (lowest kBT)
- Pair index `i` connects replicas `i` (hotter) and `i+1` (colder)
- Stats array `stats[i]` corresponds to pair `(i, i+1)`

## API Compatibility

### Type Conformance

- `SwapDEO` conforms to `Swapper` type alias
- `SwapDEOWithStats` conforms to `SwapperWithStats` type alias

### Drop-in Replacement

Works with existing `parallel_tempering()` and `parallel_tempering_with_stats()` functions:

```python
# Before
swapper = anneal.SwapAdjacentRandomlyWithStats()

# After
swapper = anneal.SwapDEOWithStats()
```

### Semantic Difference

| Aspect | SwapAdjacentRandomly | SwapDEO |
|--------|---------------------|---------|
| Attempts per call | `num_swaps` random pairs | All m-1 pairs |
| `num_swaps` parameter | Yes | No |
| Stats per call | Variable per pair | Exactly 1 per pair |
| Round-trip time | O(m²) | O(m) |

## Backward Compatibility

All existing code unchanged:
- `SwapAdjacentRandomly` class
- `SwapAdjacentRandomlyWithStats` class
- `Swapper` type alias
- `SwapperWithStats` type alias
- `parallel_tempering()` function
- `parallel_tempering_with_stats()` function

## Testing Strategy

### Test 1: Stats shape and counting
- Verify stats shape is `(m-1, 2)`
- Verify `stats[:, 0].sum() == m-1` (each pair attempted exactly once)
- Verify `stats[:, 1].sum() <= m-1` (successes ≤ attempts)

### Test 2: Deterministic acceptance
- Equal energies across all replicas → all swaps accepted
- Verify stats shows 100% acceptance

### Test 3: Deterministic rejection
- Energies in "wrong" order with very low temperature → no swaps accepted
- Verify stats shows 0% acceptance

### Test 4: Swap correctness
- Start with known membrane states at each replica
- Force a specific swap to be accepted
- Verify the membrane data actually swapped positions

### Test 5: Parity coverage
- With m replicas, verify both even and odd pairs are attempted
- For m=5: pairs 0,1,2,3 should all have exactly 1 attempt

## Files to Modify

- `src/pylakoid/structure/anneal.py`: Add `SwapDEO`, `SwapDEOWithStats`, `_deo_sweep`
- `tests/test_deo_swap.py`: New test file for DEO-specific tests

## Future Work (Not in This PR)

- Adaptive temperature ladder tuning based on swap statistics
- Round-trip time tracking
