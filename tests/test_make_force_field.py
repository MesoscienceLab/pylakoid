import typing

import Bio.PDB.Structure
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from pylakoid.structure.make_force_field import (
    preprocess_for_count_sphere_intersections,
    count_sphere_intersections_fast,
    Preprocess,
)
from pylakoid.structure.pdb import parse_pdb
from pylakoid.structure.vdw_radii import vdw_radii


def do_spheres_intersect(
    x1: Float[Array, "3"],
    r1: Float[Array, ""],
    x2: Float[Array, "3"],
    r2: Float[Array, ""],
) -> Bool[Array, ""]:
    dist = typing.cast(Float[Array, ""], jnp.linalg.norm(x1 - x2))
    return dist <= r1 + r2


def count_sphere_intersections(
    x1: Float[Array, "p 3"],
    r1: Float[Array, " p"],
    x2: Float[Array, "q 3"],
    r2: Float[Array, " q"],
) -> Int[Array, ""]:
    return jnp.sum(
        jax.vmap(
            jax.vmap(do_spheres_intersect, (None, None, 0, 0)), (0, 0, None, None)
        )(x1, r1, x2, r2)
    )


def count_atom_overlaps(
    x1: Float[Array, "p 3"],
    r1: Float[Array, " p"],
    x2: Float[Array, "p 3"],
    r2: Float[Array, " p"],
    com1: Float[Array, "3"],
    phi1: Float[Array, ""],
    com2: Float[Array, "3"],
    phi2: Float[Array, ""],
) -> Int[Array, ""]:
    def rotate(x: Float[Array, "p 3"], angle: Float[Array, ""]) -> Float[Array, "p 3"]:
        cos = jnp.cos(angle)
        sin = jnp.sin(angle)
        rot = jnp.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])  # pyright: ignore [reportUnknownMemberType]
        return jnp.matrix_transpose(rot @ jnp.matrix_transpose(x))

    x1 = rotate(x1, phi1) + com1
    x2 = rotate(x2, phi2) + com2
    return count_sphere_intersections(x1, r1, x2, r2)


def preprocess_protein(
    pdb: Bio.PDB.Structure.Structure, n: int, key: Key[Array, ""]
) -> Preprocess:
    x = jnp.array([atom.coord for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    r = jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    return preprocess_for_count_sphere_intersections(x, r, n, key)


def test_does_count_match_count_fast():
    n_iter = 1000
    key = jax.random.key(0)
    pdb1 = parse_pdb("cytb6f", "tests/6rqf_dimer_opm.pdb")
    pdb2 = parse_pdb("lhcii", "tests/1rwt_trimer_opm.pdb")
    n1 = 800
    n2 = 800
    r_max = 100.0

    x1 = jnp.array([atom.coord for atom in pdb1.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    r1 = jnp.array([vdw_radii[atom.element] for atom in pdb1.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    x2 = jnp.array([atom.coord for atom in pdb2.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    r2 = jnp.array([vdw_radii[atom.element] for atom in pdb2.get_atoms()])  # pyright: ignore [reportUnknownMemberType]

    k1, k2, key = jax.random.split(key, 3)
    preprocess1 = preprocess_protein(pdb1, n1, k1)
    del k1
    preprocess2 = preprocess_protein(pdb2, n2, k2)
    del k2

    k1, k2, k3, k4 = jax.random.split(key, 4)
    com1 = jax.random.uniform(k1, (n_iter, 3), minval=0.0, maxval=r_max)  # pyright: ignore [reportUnknownMemberType]
    del k1
    phi1 = jax.random.uniform(k2, (n_iter,), minval=0.0, maxval=2 * jnp.pi)  # pyright: ignore [reportUnknownMemberType]
    del k2
    com2 = jax.random.uniform(k3, (n_iter, 3), minval=0.0, maxval=r_max)  # pyright: ignore [reportUnknownMemberType]
    del k3
    phi2 = jax.random.uniform(k4, (n_iter,), minval=0.0, maxval=2 * jnp.pi)  # pyright: ignore [reportUnknownMemberType]
    del k4

    count_fn = eqx.filter_jit(count_atom_overlaps)
    count_fast_fn = eqx.filter_jit(count_sphere_intersections_fast)
    for i in range(n_iter):
        count = count_fn(x1, r1, x2, r2, com1[i], phi1[i], com2[i], phi2[i]).item()
        count_fast = count_fast_fn(
            preprocess1, preprocess2, com1[i], phi1[i], com2[i], phi2[i]
        ).item()
        assert count == count_fast
