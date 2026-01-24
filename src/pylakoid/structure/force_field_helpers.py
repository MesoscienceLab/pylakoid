import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Key

from pylakoid.structure.force_field import (
    AtomicOverlapForceField,
    ClampedSplineForceField,
    SplineForceField,
    TypewiseForceField,
)
from pylakoid.structure import make_force_field as mff
from pylakoid.structure.make_force_field_helpers import deserialize_force_field
from pylakoid.structure.pdb import parse_pdb
from pylakoid.structure.vdw_radii import vdw_radii


def prepare_clamped_spline_force_field(
    path: str,
    pdb1_path: str,
    pdb2_path: str,
    pt1: int,
    pt2: int,
    override_max_clashes: float | None,
) -> ClampedSplineForceField:
    """
    Prepares a `ClampedSplineForceField` for downstream usage as part of a `TypewiseForceField`.

    Parameters:
        path: The file from which to load the spline parameters.
        pdb1_path: The path to the PDB file for the first protein.
        pdb2_path: The path to the PDB file for the second protein.
        pt1: The type of the first protein assumed by the `SplineForceField`.
        pt2: The type of the second protein assumed by the `SplineForceField`.
        override_max_clashes: If `None`, `max_clashes` is the largest value from the spline
            parameters. Otherwise, `max_clashes = override_max_clashes`.

    Returns:
        The `ClampedSplineForceField`.
    """
    parameters = deserialize_force_field(path, pdb1_path, pdb2_path)

    spline = jsp.interpolate.RegularGridInterpolator(parameters[0:3], parameters[3])
    sff = SplineForceField(spline, jnp.array(pt1), jnp.array(pt2))  # pyright: ignore [reportUnknownMemberType]

    r_min = jnp.min(parameters[0])
    r_max = jnp.max(parameters[0])
    max_clashes = (
        jnp.array(override_max_clashes)  # pyright: ignore [reportUnknownMemberType]
        if override_max_clashes is not None
        else 1.0 * jnp.max(parameters[3])  # promote to float
    )
    return ClampedSplineForceField(sff, r_min, r_max, max_clashes)


def prepare_force_field(
    path: dict[tuple[str, str], str],
    pdb_path: dict[str, str],
    protein_type_map: dict[str, int],
    override_max_clashes: dict[tuple[str, str], float],
) -> TypewiseForceField:
    """
    Prepares a `TypewiseForceField` for use in membrane annealing based on a protein type map
    (see `pylakoid.structure.anneal_helpers.prepare_protein_type`).

    Parameters:
        path: A mapping from a pair of `str` protein types to a path from which spline parameters
            will be loaded.
        pdb_path: A mapping from a `str` protein type to the path to the corresponding PDB file.
        protein_type_map: A mapping from a `str` protein type to an internal `int` `protein_type`.
        override_max_clashes: A mapping from a pair of `str` protein types to the
            `override_max_clashes` value which will be passed to
            `prepare_clamped_spline_force_field`. If a key is missing, then `None` will be used.

    Returns:
        The `TypewiseForceField`.
    """
    num_types = max([t + 1 for t in protein_type_map.values()])
    inv_protein_type_map: dict[int, str] = {v: k for k, v in protein_type_map.items()}
    ffs: list[ClampedSplineForceField] = []
    for pt1 in range(num_types):
        for pt2 in range(num_types):
            k1, k2 = (inv_protein_type_map[pt1], inv_protein_type_map[pt2])
            if (k1, k2) in path:
                ffs.append(
                    prepare_clamped_spline_force_field(
                        path[(k1, k2)],
                        pdb_path[k1],
                        pdb_path[k2],
                        pt1,
                        pt2,
                        override_max_clashes.get((k1, k2), None),
                    )
                )
            elif (k2, k1) in path:
                ffs.append(
                    prepare_clamped_spline_force_field(
                        path[(k2, k1)],
                        pdb_path[k2],
                        pdb_path[k1],
                        pt2,
                        pt1,
                        override_max_clashes.get((k2, k1), None),
                    )
                )
            else:
                raise Exception(f"Missing force field for {(k1, k2)} and {(k2, k1)}")
    return TypewiseForceField(tuple(ffs), jnp.array(num_types))  # pyright: ignore [reportUnknownMemberType]


def prepare_exact_force_field(
    pdb_path: dict[str, str],
    num_groups: dict[str, int],
    protein_type_map: dict[str, int],
    key: Key[Array, ""],
) -> TypewiseForceField:
    """
    Prepares a `TypewiseForceField` for use in membrane annealing based on a protein type map
    (see `pylakoid.structure.anneal_helpers.prepare_protein_type`).

    Parameters:
        pdb_path: A mapping from a `str` protein type to the path to the corresponding PDB file.
        protein_type_map: A mapping from a `str` protein type to an internal `int` `protein_type`.
        key: A JAX random key used to generate random numbers. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

    Returns:
        The `TypewiseForceField`.
    """
    preprocess_fn = eqx.filter_jit(mff.preprocess_for_count_sphere_intersections)
    preprocessed: dict[int, mff.Preprocess] = {}
    for name, path in pdb_path.items():
        pdb = parse_pdb(name, path)
        x = jnp.array([atom.coord for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
        r = jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
        k1, key = jax.random.split(key, 2)
        preprocessed[protein_type_map[name]] = preprocess_fn(x, r, num_groups[name], k1)
        del k1

    num_types = max([t + 1 for t in protein_type_map.values()])
    ffs: list[AtomicOverlapForceField] = []
    for pt1 in range(num_types):
        for pt2 in range(num_types):
            ffs.append(AtomicOverlapForceField(preprocessed[pt1], preprocessed[pt2]))
    return TypewiseForceField(tuple(ffs), jnp.array(num_types))  # pyright: ignore [reportUnknownMemberType]
