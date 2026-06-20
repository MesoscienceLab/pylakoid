# ABOUTME: Carried test for pylakoid.structure.metrics error/clash density primitives.
# ABOUTME: Pins the energy-per-area normalization, the int() clash-count convention, and degenerate-boundary raises.

import jax.numpy as jnp
import pytest

from pylakoid.structure.anneal import Membrane
from pylakoid.structure.metrics import clash_density, error_density


def _membrane():
    # Four proteins with huge radii so every pair is within the cutoff in
    # one_particle_energy -> a constant fake force field yields a fully
    # hand-computable total_energy (it sums the force field over every pair).
    com = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    angle = jnp.zeros(4)
    radius = jnp.full((4,), 1.0e6)
    protein_type = jnp.array([0, 1, 2, 0])
    return Membrane(com, angle, radius, protein_type)


def _const_ff(value):
    # A force field (signature com1, angle1, type1, com2, angle2, type2 -> energy)
    # that returns a constant regardless of the pair, so the summed energy is known.
    def ff(com1, angle1, type1, com2, angle2, type2):
        return jnp.asarray(value)

    return ff


# Ordered polygon vertices in the boundary's native units (the records' boundary
# points are in angstroms). Shoelace areas computed by hand below.
_SQUARE = jnp.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])  # area 400
_RECT_2X = jnp.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])  # area 800 (2x)
# Right triangle: shoelace area = 300, but its axis-aligned bounding box is 30*20 = 600.
# A bounding-box-area implementation would compute 600 and fail here.
_TRIANGLE = jnp.array([[0.0, 0.0], [30.0, 0.0], [0.0, 20.0]])  # area 300, bbox 600
# The square wound clockwise: signed shoelace is -400; absolute area is 400.
# A signed-shoelace implementation (no abs) would yield a negative density.
_SQUARE_CW = jnp.array([[0.0, 0.0], [0.0, 20.0], [20.0, 20.0], [20.0, 0.0]])  # |area| 400

# Constant fake force field 2.45 over all 12 ordered pairs of 4 proteins ->
# total_energy = 0.5 * 4 * 3 * 2.45 = 14.7.
_FF = _const_ff(2.45)
_ENERGY = 14.7


def test_error_density_is_energy_over_area_per_100():
    # area(square) = 400 -> area/100 = 4 -> density = 14.7 / 4 = 3.675.
    # Pins both the division by area AND the /100 factor (an impl dividing by the
    # raw area, not area/100, is off by 100x and fails here).
    d = error_density(_membrane(), _SQUARE, _FF)
    assert float(d) == pytest.approx(_ENERGY / (400.0 / 100.0), rel=1e-4)


def test_error_density_scales_inverse_with_area():
    # Same membrane + force field (identical energy), 2x area -> half the density.
    # Catches an impl that returns the raw energy without normalizing by area.
    d1 = error_density(_membrane(), _SQUARE, _FF)
    d2 = error_density(_membrane(), _RECT_2X, _FF)
    assert float(d2) == pytest.approx(float(d1) / 2.0, rel=1e-4)
    assert float(d2) == pytest.approx(_ENERGY / (800.0 / 100.0), rel=1e-4)


def test_error_density_uses_polygon_area_not_bounding_box():
    # Triangle: true shoelace area 300 (density 14.7/3 = 4.9), bounding box 600
    # (which would give 14.7/6 = 2.45). Catches a bounding-box-area implementation.
    d = error_density(_membrane(), _TRIANGLE, _FF)
    assert float(d) == pytest.approx(_ENERGY / (300.0 / 100.0), rel=1e-4)  # 4.9


def test_error_density_uses_absolute_area_for_reversed_orientation():
    # Clockwise winding: a signed-shoelace impl yields area -400 -> negative density.
    # The contract is absolute area, so the density must be the same positive value
    # as the counter-clockwise square.
    d = error_density(_membrane(), _SQUARE_CW, _FF)
    assert float(d) > 0.0
    assert float(d) == pytest.approx(_ENERGY / (400.0 / 100.0), rel=1e-4)  # 3.675


def test_error_density_zero_when_no_clashes():
    d = error_density(_membrane(), _SQUARE, _const_ff(0.0))
    assert float(d) == pytest.approx(0.0, abs=1e-6)


def test_clash_density_applies_integer_count_convention():
    # clash_density truncates the (float) energy to an integer clash COUNT before
    # normalizing: int(14.7) = 14 -> 14/4 = 3.5, distinct from error_density's 3.675.
    # This pins the paper's integer-count convention that separates the two functions.
    cd = clash_density(_membrane(), _SQUARE, _FF)
    ed = error_density(_membrane(), _SQUARE, _FF)
    assert float(cd) == pytest.approx(14.0 / (400.0 / 100.0), rel=1e-4)  # 3.5
    assert float(ed) == pytest.approx(14.7 / (400.0 / 100.0), rel=1e-4)  # 3.675
    assert abs(float(cd) - float(ed)) > 0.1


def test_clash_density_uses_absolute_polygon_area():
    # clash_density normalizes by the same shoelace-abs polygon area as error_density,
    # independently pinned here: triangle (area 300, bbox 600) -> int(14.7)/3 = 4.6667
    # (a bounding-box impl would give 14/6 = 2.333), and a clockwise winding still
    # yields the positive polygon area -> 14/4 = 3.5.
    cd_tri = clash_density(_membrane(), _TRIANGLE, _FF)
    cd_cw = clash_density(_membrane(), _SQUARE_CW, _FF)
    assert float(cd_tri) == pytest.approx(14.0 / (300.0 / 100.0), rel=1e-4)  # 4.6667
    assert float(cd_cw) == pytest.approx(14.0 / (400.0 / 100.0), rel=1e-4)  # 3.5


def test_error_density_raises_on_too_few_boundary_points():
    for boundary in (
        jnp.zeros((0, 2)),                       # empty
        jnp.array([[0.0, 0.0]]),                 # single point
        jnp.array([[0.0, 0.0], [20.0, 0.0]]),    # two points
    ):
        with pytest.raises(ValueError):
            error_density(_membrane(), boundary, _FF)


def test_error_density_raises_on_collinear_boundary():
    # Three collinear points -> zero polygon area -> must raise, not divide by zero
    # or return NaN/inf.
    collinear = jnp.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    with pytest.raises(ValueError):
        error_density(_membrane(), collinear, _FF)


def test_clash_density_raises_on_degenerate_boundary():
    # The wrapper must enforce the same boundary validity as error_density: both the
    # fewer-than-3-vertices case and the zero-area (collinear) case.
    two_points = jnp.array([[0.0, 0.0], [20.0, 0.0]])
    collinear = jnp.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    with pytest.raises(ValueError):
        clash_density(_membrane(), two_points, _FF)
    with pytest.raises(ValueError):
        clash_density(_membrane(), collinear, _FF)
