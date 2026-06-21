# ABOUTME: Builds intensive kBT schedules for membrane tempering runs.
# ABOUTME: Validates scalar ladder inputs before constructing geometric rungs.

import math

import jax.numpy as jnp


def intensive_kbt_ladder(n_proteins, *, kbt_hot, kbt_cold, rungs_per_sqrt_n):
    if not math.isfinite(n_proteins) or n_proteins < 1:
        raise ValueError("n_proteins must be finite and at least 1")
    if not math.isfinite(kbt_hot):
        raise ValueError("kbt_hot must be finite")
    if not math.isfinite(kbt_cold) or kbt_cold <= 0:
        raise ValueError("kbt_cold must be finite and positive")
    if kbt_hot <= kbt_cold:
        raise ValueError("kbt_hot must be greater than kbt_cold")
    if not math.isfinite(rungs_per_sqrt_n) or rungs_per_sqrt_n <= 0:
        raise ValueError("rungs_per_sqrt_n must be finite and positive")

    n_rungs = max(2, round(rungs_per_sqrt_n * math.sqrt(n_proteins)))
    return jnp.geomspace(kbt_hot, kbt_cold, n_rungs)
