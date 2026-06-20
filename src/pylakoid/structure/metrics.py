# ABOUTME: Computes membrane energy and clash densities normalized by polygon area.
# ABOUTME: Validates ordered boundary polygons with an orientation-independent shoelace area.

import numpy as np

from pylakoid.structure.anneal import total_energy


def _shoelace_area(boundary) -> float:
    vertices = np.asarray(boundary, dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("boundary must contain at least three 2D vertices")

    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))

    if not np.isfinite(area) or area <= 1.0e-12:
        raise ValueError("boundary polygon area must be non-zero")

    return area


def error_density(membrane, boundary, force_field):
    area = _shoelace_area(boundary)
    return total_energy(membrane, force_field) / (area / 100.0)


def clash_density(membrane, boundary, exact_ff):
    area = _shoelace_area(boundary)
    return int(total_energy(membrane, exact_ff)) / (area / 100.0)
