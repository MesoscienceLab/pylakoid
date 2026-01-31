# Parallel Tempering Swap Statistics Design

## Goal

Track when temperature swaps succeed between replicas in parallel tempering, enabling analysis scripts to log swap events and diagnose mixing quality.

## Constraints

- Must work inside `jax.lax.fori_loop` (JAX-compatible accumulator)
- Must maintain backward compatibility (existing code unchanged)
- Needs proper tests

## Design Decisions

1. **Granularity**: Per-pair statistics (attempts/successes for each adjacent temperature pair)
2. **API approach**: Two separate classes (`SwapAdjacentRandomly` unchanged, new `SwapAdjacentRandomlyWithStats`)
3. **Parallel tempering**: New `parallel_tempering_with_stats()` function mirrors the swapper approach

## Data Structure

```python
# Shape: (m-1, 2) where m = number of replicas
# swap_stats[i, 0] = attempts for pair (i, i+1)
# swap_stats[i, 1] = successes for pair (i, i+1)
swap_stats: Int[Array, "m_minus_1 2"]
```

For 4 replicas (T0 < T1 < T2 < T3):
- `swap_stats[0]` → pair (0,1)
- `swap_stats[1]` → pair (1,2)
- `swap_stats[2]` → pair (2,3)

Acceptance rates: `swap_stats[:, 1] / swap_stats[:, 0]`

## New Code

### Type Alias

```python
SwapperWithStats = Callable[
    [MultiMembrane, Float[Array, " m"], Float[Array, " m"], Key[Array, ""]],
    tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]],
]
```

### Private Shared Implementation

```python
def _swap_adjacent_randomly_impl(
    multi_membrane: MultiMembrane,
    energy: Float[Array, " m"],
    kBT: Float[Array, " m"],
    keys: Key[Array, "num_swaps"],
    num_swaps: int,
    track_stats: bool,
) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
    """Core swap logic. Returns stats array (zeros if track_stats=False)."""
    ...
```

### New Swapper Class

```python
class SwapAdjacentRandomlyWithStats(eqx.Module):
    """Like SwapAdjacentRandomly but returns swap statistics."""

    num_swaps: int

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"], Int[Array, "m_minus_1 2"]]:
        ...
```

### New Parallel Tempering Function

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
    ...
```

## Caller Usage Pattern

```python
def run_parallel_tempering_with_stats(...):
    swapper = anneal.SwapAdjacentRandomlyWithStats(len(tempering_params))
    keys = jax.random.split(key, n_swaps)

    initial_stats = jnp.zeros((len(tempering_params) - 1, 2), dtype=jnp.int32)

    def step(i, state):
        membrane, energy, cumulative_stats = state
        membrane, energy, epoch_stats = anneal.parallel_tempering_with_stats(...)
        return membrane, energy, cumulative_stats + epoch_stats

    return jax.lax.fori_loop(0, n_swaps, step, (membrane, initial_energy, initial_stats))
```

## Testing Strategy

### Test 1: Stats shape and counting
- Verify shape is `(m-1, 2)`
- Verify `swap_stats[:, 0].sum() == num_swaps`

### Test 2: Correctness under deterministic conditions
- Equal energies → all swaps accepted
- Unfavorable ordering with low temperature → no swaps accepted

### Test 3: Backward compatibility
- `SwapAdjacentRandomly` returns 2-tuple
- `parallel_tempering` works unchanged
- Results match between old/new (ignoring stats)

## Files to Modify

- `src/pylakoid/structure/anneal.py`: Add new types, classes, and functions
- `tests/test_swap_stats.py`: New test file

## Unchanged (Backward Compatible)

- `SwapAdjacentRandomly` class
- `Swapper` type alias
- `parallel_tempering()` function
- All existing callers
