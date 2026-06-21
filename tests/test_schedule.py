# ABOUTME: Carried test for pylakoid.structure.schedule.intensive_kbt_ladder.
# ABOUTME: Pins intensive (size-independent) kBT bounds, sqrt(N) rung scaling, geometric spacing, the 2-rung floor, and degenerate-input raises.

import numpy as np
import pytest

from pylakoid.structure.schedule import intensive_kbt_ladder


def _ladder(n_proteins, *, kbt_hot=1000.0, kbt_cold=2.0, rungs_per_sqrt_n=4.0):
    # Coerce to a numpy float64 array so the test is agnostic to whether the
    # implementation returns a numpy or a JAX array. Note: coercion does not add
    # precision -- a float32 jnp.geomspace stays float32-accurate, hence the
    # rel=1e-4 tolerances below.
    return np.asarray(
        intensive_kbt_ladder(
            n_proteins,
            kbt_hot=kbt_hot,
            kbt_cold=kbt_cold,
            rungs_per_sqrt_n=rungs_per_sqrt_n,
        ),
        dtype=np.float64,
    )


# --- Intensive bounds: endpoints are kbt_hot/kbt_cold, independent of size ---


def test_endpoints_equal_bounds():
    ladder = _ladder(100)
    assert ladder.ndim == 1
    assert ladder[0] == pytest.approx(1000.0, rel=1e-4)
    assert ladder[-1] == pytest.approx(2.0, rel=1e-4)


def test_bounds_independent_of_size():
    # The whole point of the fix: a bigger/denser membrane must NOT get a hotter
    # or colder ladder. An extensive impl that scales the bounds with N (the SMA
    # bug, kBT_hot ~ total energy) would give different endpoints here.
    small = _ladder(25)
    big = _ladder(225)
    assert small[0] == pytest.approx(big[0], rel=1e-4)  # same hot end
    assert small[-1] == pytest.approx(big[-1], rel=1e-4)  # same cold end
    assert big[0] == pytest.approx(1000.0, rel=1e-4)
    assert big[-1] == pytest.approx(2.0, rel=1e-4)


# --- Rung count scales as sqrt(N) (not linear, not constant) ---


def test_rung_count_sqrt_scaling():
    # rungs_per_sqrt_n=4 over perfect squares -> exact integer rung counts:
    # round(4 * sqrt(N)).
    assert len(_ladder(25)) == 20  # 4 * sqrt(25) = 20
    assert len(_ladder(100)) == 40  # 4 * sqrt(100) = 40
    assert len(_ladder(225)) == 60  # 4 * sqrt(225) = 60


def test_rung_count_discriminates_sqrt_from_linear():
    # 4x the proteins -> 2x the rungs (sqrt(4)=2); 9x -> 3x (sqrt(9)=3).
    # A linear (n_rungs proportional to N) impl gives 4x and 9x; a constant
    # impl gives 1x. Only the sqrt(N) law produces exactly 2x and 3x.
    base = len(_ladder(25))
    assert len(_ladder(100)) == 2 * base  # 100 = 4 * 25
    assert len(_ladder(225)) == 3 * base  # 225 = 9 * 25


def test_rung_count_rounds_to_nearest():
    # round(rungs_per_sqrt_n * sqrt(n_proteins)) -- pinned in BOTH directions so
    # floor/truncation and ceil are both excluded, while the exact-half (x.5)
    # tie convention is deliberately left unpinned (no .5 boundary is used):
    #   n=50: sqrt(50)*4 = 28.284 -> 28 (rounds DOWN; a ceil impl gives 29)
    #   n=30: sqrt(30)*4 = 21.909 -> 22 (rounds UP; a floor/int impl gives 21)
    assert len(_ladder(50)) == 28
    assert len(_ladder(30)) == 22


# --- Geometric spacing (not arithmetic/linear) ---


def test_geometric_spacing_three_rungs():
    # n=4, rungs_per_sqrt_n=1.5 -> round(1.5 * 2) = 3 rungs; 100 -> 1 geometric
    # is [100, 10, 1]. A linear ladder would put 50.5 in the middle, not 10.
    ladder = _ladder(4, kbt_hot=100.0, kbt_cold=1.0, rungs_per_sqrt_n=1.5)
    assert len(ladder) == 3
    np.testing.assert_allclose(ladder, [100.0, 10.0, 1.0], rtol=1e-4)


def test_geometric_spacing_five_rungs():
    # n=25, rungs_per_sqrt_n=1 -> 5 rungs; 10000 -> 1 geometric is a decade ladder.
    ladder = _ladder(25, kbt_hot=10000.0, kbt_cold=1.0, rungs_per_sqrt_n=1.0)
    assert len(ladder) == 5
    np.testing.assert_allclose(ladder, [10000.0, 1000.0, 100.0, 10.0, 1.0], rtol=1e-4)


def test_constant_ratio_between_rungs():
    # Geometric => every consecutive ratio is the same (and < 1, i.e. cooling).
    ladder = _ladder(100)  # 40 rungs, 1000 -> 2
    ratios = ladder[1:] / ladder[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-4)
    assert ratios[0] < 1.0


# --- Monotonic hot -> cold ---


def test_monotonic_decreasing():
    ladder = _ladder(100)
    assert np.all(np.diff(ladder) < 0.0)


# --- Minimum 2 rungs (both endpoints always present) ---


def test_min_two_rungs_when_count_would_be_zero():
    # n=1, rungs_per_sqrt_n=0.5 -> round(0.5) -> 0 (or 1 under round-half-up);
    # either way the floor lifts it to 2 so both endpoints survive.
    ladder = _ladder(1, kbt_hot=500.0, kbt_cold=5.0, rungs_per_sqrt_n=0.5)
    assert len(ladder) == 2
    np.testing.assert_allclose(ladder, [500.0, 5.0], rtol=1e-4)


def test_min_two_rungs_when_count_would_be_one():
    # n=1, rungs_per_sqrt_n=1.0 -> round(1.0) = 1 -> floored to 2.
    ladder = _ladder(1, kbt_hot=500.0, kbt_cold=5.0, rungs_per_sqrt_n=1.0)
    assert len(ladder) == 2


# --- Degenerate input raises loudly (never a silent nan / empty / 1-element) ---


@pytest.mark.parametrize("n_proteins", [0, -1, -10, 0.5, 0.999])
def test_raises_on_n_proteins_below_one(n_proteins):
    # The contract is n_proteins < 1 raises -- this includes fractional counts
    # in (0, 1), so an impl that guards `<= 0` instead of `< 1` is caught.
    with pytest.raises(ValueError):
        intensive_kbt_ladder(
            n_proteins, kbt_hot=1000.0, kbt_cold=2.0, rungs_per_sqrt_n=4.0
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_raises_on_nonfinite_n_proteins(bad):
    with pytest.raises(ValueError):
        intensive_kbt_ladder(bad, kbt_hot=1000.0, kbt_cold=2.0, rungs_per_sqrt_n=4.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_raises_on_nonfinite_kbt_hot(bad):
    with pytest.raises(ValueError):
        intensive_kbt_ladder(100, kbt_hot=bad, kbt_cold=2.0, rungs_per_sqrt_n=4.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_raises_on_nonfinite_kbt_cold(bad):
    with pytest.raises(ValueError):
        intensive_kbt_ladder(100, kbt_hot=1000.0, kbt_cold=bad, rungs_per_sqrt_n=4.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_raises_on_nonfinite_rungs_per_sqrt_n(bad):
    with pytest.raises(ValueError):
        intensive_kbt_ladder(100, kbt_hot=1000.0, kbt_cold=2.0, rungs_per_sqrt_n=bad)


@pytest.mark.parametrize("kbt_cold", [0.0, -1.0])
def test_raises_on_nonpositive_kbt_cold(kbt_cold):
    # A geometric ladder is undefined for a non-positive endpoint.
    with pytest.raises(ValueError):
        intensive_kbt_ladder(
            100, kbt_hot=1000.0, kbt_cold=kbt_cold, rungs_per_sqrt_n=4.0
        )


@pytest.mark.parametrize("kbt_hot,kbt_cold", [(2.0, 10.0), (10.0, 10.0)])
def test_raises_when_hot_not_above_cold(kbt_hot, kbt_cold):
    # Inverted (hot < cold) or flat (hot == cold): there is no hot->cold range.
    with pytest.raises(ValueError):
        intensive_kbt_ladder(
            100, kbt_hot=kbt_hot, kbt_cold=kbt_cold, rungs_per_sqrt_n=4.0
        )


@pytest.mark.parametrize("rungs_per_sqrt_n", [0.0, -2.0])
def test_raises_on_nonpositive_rungs_per_sqrt_n(rungs_per_sqrt_n):
    # A non-positive rung density is meaningless; raise rather than silently
    # collapse to the 2-rung floor and hide a caller bug.
    with pytest.raises(ValueError):
        intensive_kbt_ladder(
            100, kbt_hot=1000.0, kbt_cold=2.0, rungs_per_sqrt_n=rungs_per_sqrt_n
        )


# --- Keyword-only API (the design specifies `*`) ---


def test_bounds_are_keyword_only():
    with pytest.raises(TypeError):
        intensive_kbt_ladder(100, 1000.0, 2.0, 4.0)  # positional kw-only args
