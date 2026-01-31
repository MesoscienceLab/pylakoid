# pylakoid Library

## Running Python Code

Use the pylakoid_314 conda environment. **Do NOT try to activate conda** - call the Python executable directly:

```bash
/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python script.py
```

For running tests:
```bash
/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pytest tests/
```

For type checking:
```bash
/Users/doranraccah/miniconda3/envs/pylakoid_314/bin/python -m pyright src/
```

## Project Structure

```
pylakoid/
├── src/pylakoid/
│   ├── structure/      # Core membrane simulation code
│   │   ├── anneal.py   # Parallel tempering, Monte Carlo, swapping
│   │   ├── checker.py  # Position constraint checkers
│   │   └── ...
│   └── cli/            # Command-line interface
├── tests/              # pytest tests
└── docs/               # mkdocs documentation
```

## Key Modules

- `structure/anneal.py` - Contains `parallel_tempering()`, `run_monte_carlo()`, `SwapAdjacentRandomly` swapper
- `structure/checker.py` - Position constraint checkers like `CenterOfMassInCircleChecker`
- `structure/pdb.py` - PDB file parsing

## Related Repository

This library is used by the **SpinachMembraneAnalysis** project:

```
/Users/doranraccah/Dropbox/Writing2026/Manuscripts/pylakoid/SpinachMembraneAnalysis
```

That project contains analysis scripts that use pylakoid for membrane simulation:
- `analyze_membrane_frozen.py` - Optimizes protein orientations (frozen positions)
- `analyze_membrane_swap.py` - Optimizes with PSII supercomplex type swapping
- `add_lhcii_and_anneal.py` - Adds LHCII trimers and anneals

## Pending Feature Request

**Parallel Tempering Swap Statistics**

The analysis scripts need to track when temperature swaps succeed in parallel tempering. Currently `SwapAdjacentRandomly` (in `anneal.py`) doesn't return swap success information.

Requirements:
1. Return swap statistics from `SwapAdjacentRandomly` without breaking existing API
2. Must work inside JAX's `jax.lax.fori_loop` (no Python side effects)
3. Maintain backward compatibility with existing code
4. Add proper tests

Key code location: `src/pylakoid/structure/anneal.py` lines 650-714 (`SwapAdjacentRandomly` class)

## Development Notes

- Uses JAX for GPU/TPU acceleration and JIT compilation
- Type annotations use jaxtyping (e.g., `Float[Array, "n 2"]`)
- Strict pyright type checking enabled
- Code style enforced with ruff
