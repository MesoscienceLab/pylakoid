import typing

import Bio.PDB.Structure
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jaxtyping import Array, Int, Key

import pylakoid.structure.force_field as ff
import pylakoid.structure.make_force_field as mff
from pylakoid.structure.pdb import parse_pdb
from pylakoid.structure.vdw_radii import vdw_radii

# This test only passes in float64 mode due to small numerical errors, because
# there is a sharp threshold on overlap counting.
jax.config.update("jax_enable_x64", True)  # pyright: ignore [reportUnknownMemberType]


def preprocess_protein(
    pdb: Bio.PDB.Structure.Structure, n: int, key: Key[Array, ""]
) -> mff.Preprocess:
    x = jnp.array([atom.coord for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    r = jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    return mff.preprocess_for_count_sphere_intersections(x, r, n, key)


def test_aoff_matches_sff():
    key = jax.random.key(0)
    n_iter = 1000

    k1, k2, key = jax.random.split(key, 3)
    preprocess1 = preprocess_protein(
        parse_pdb("cytb6f", "tests/6rqf_dimer_opm.pdb"),
        800,
        k1,
    )
    del k1
    preprocess2 = preprocess_protein(
        parse_pdb("lhcii", "tests/1rwt_trimer_opm.pdb"),
        800,
        k2,
    )
    del k2

    pt1 = jnp.array(1, dtype=jnp.int32)  # pyright: ignore [reportUnknownMemberType]
    pt2 = jnp.array(2, dtype=jnp.int32)  # pyright: ignore [reportUnknownMemberType]

    aoff = eqx.filter_jit(ff.AtomicOverlapForceField(preprocess1, preprocess2))

    r = jnp.linspace(0, 50, 10)  # pyright: ignore [reportUnknownMemberType]
    phi1 = jnp.linspace(0, 2 * jnp.pi, 10)  # pyright: ignore [reportUnknownMemberType]
    phi2 = jnp.linspace(0, 2 * jnp.pi, 10)  # pyright: ignore [reportUnknownMemberType]
    k1, key = jax.random.split(key, 2)
    counts = typing.cast(
        Int[Array, "10 10 10"],
        jax.jit(mff.count_sphere_intersections_parallel)(  # pyright: ignore [reportUnknownMemberType]
            preprocess1, preprocess2, r, phi1, phi2, k1
        ),
    )
    del k1
    spline = jsp.interpolate.RegularGridInterpolator((r, phi1, phi2), counts)
    sff = eqx.filter_jit(ff.SplineForceField(spline, pt1, pt2))

    k1, k2, k3, k4, k5, key = jax.random.split(key, 6)
    com1 = jax.random.normal(k1, (n_iter, 2))  # pyright: ignore [reportUnknownMemberType]
    del k1
    theta = jax.random.uniform(k2, (n_iter,), minval=0, maxval=2 * jnp.pi)  # pyright: ignore [reportUnknownMemberType]
    del k2
    dist = jax.random.choice(
        k3, r[1:-1], (n_iter,)
    )  # Don't sample end points to avoid out of bounds on sff
    del k3
    angle1 = jax.random.choice(k4, phi1, (n_iter,)) + theta
    del k4
    angle2 = jax.random.choice(k5, phi1, (n_iter,)) + theta
    del k5

    for i in range(n_iter):
        cos = jnp.cos(theta[i])
        sin = jnp.sin(theta[i])
        rot = jnp.array([[cos, -sin], [sin, cos]])  # pyright: ignore [reportUnknownMemberType]
        com2 = com1[i] + rot @ jnp.array([dist[i], 0])  # pyright: ignore [reportUnknownMemberType]

        aoff_val = aoff(com1[i], angle1[i], pt1, com2, angle2[i], pt2)
        sff_val = sff(com1[i], angle1[i], pt1, com2, angle2[i], pt2)
        assert jnp.isclose(aoff_val, sff_val)
