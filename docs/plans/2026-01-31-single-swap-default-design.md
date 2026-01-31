# Single Swap Default Design

## Problem

`SwapAdjacentRandomly` and `SwapAdjacentRandomlyWithStats` do `num_swaps` consecutive swap attempts per call with no local MC moves between them. After a swap is accepted, subsequent attempts provide no sampling benefit - configurations need local moves to explore new phase space before another swap is useful.

## Solution

Make `num_swaps` optional with default value of 1. Update all callers to use the default, getting proper interleaving of swaps and local MC moves via the existing outer loop in `parallel_tempering_loop`.

## Changes

**In `anneal.py`:**
- `SwapAdjacentRandomly`: `num_swaps: int` → `num_swaps: int = 1`
- `SwapAdjacentRandomlyWithStats`: same change

**Callers to update (remove argument):**
- `pylakoid/src/pylakoid/cli/simulate.py:272`
- `SpinachMembraneAnalysis/analyze_membrane_swap.py:291`
- `SpinachMembraneAnalysis/add_lhcii_and_anneal.py:354`
- `SpinachMembraneAnalysis/analyze_membrane_frozen.py:182`

**Tests:**
- Existing tests with explicit `num_swaps` remain valid
- Add test verifying default does 1 attempt per call
