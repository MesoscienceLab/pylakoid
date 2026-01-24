import pathlib

import Bio.PDB.Structure
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import pytest

import pylakoid.structure.make_force_field as mff
from pylakoid.structure.make_force_field_helpers import (
    serialize_force_field,
    deserialize_force_field,
)
from pylakoid.structure.pdb import parse_pdb
from pylakoid.structure.vdw_radii import vdw_radii


def test_round_trip(tmp_path: pathlib.Path):
    out_path = str(tmp_path / "out.safetensors")
    pdb_path1 = "tests/1rwt_trimer_opm.pdb"
    pdb_path2 = "tests/6rqf_dimer_opm.pdb"
    key = jax.random.key(0)

    k1, key = jax.random.split(key, 2)
    r = jnp.linspace(0.0, 50.0, 10)  # pyright: ignore [reportUnknownMemberType]
    phi1 = jnp.linspace(0.0, 2 * jnp.pi, 10)  # pyright: ignore [reportUnknownMemberType]
    phi2 = jnp.linspace(0.0, 2 * jnp.pi, 10)  # pyright: ignore [reportUnknownMemberType]
    serialize_force_field(
        out_path,
        pdb_path1,
        500,
        pdb_path2,
        1000,
        r,
        phi1,
        phi2,
        k1,
    )
    del k1

    params = deserialize_force_field(out_path, pdb_path1, pdb_path2)
    assert jnp.all(params[0] == r)
    assert jnp.all(params[1] == phi1)
    assert jnp.all(params[2] == phi2)
    assert params[3].shape == (r.shape[0], phi1.shape[0], phi2.shape[0])

    def get_x_and_r(
        pdb: Bio.PDB.Structure.Structure,
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        return (
            jnp.array([atom.coord for atom in pdb.get_atoms()]),  # pyright: ignore [reportUnknownMemberType]
            jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()]),  # pyright: ignore [reportUnknownMemberType]
        )

    pdb1 = parse_pdb("protein1", pdb_path1)
    pdb2 = parse_pdb("protein2", pdb_path2)
    k1, k2, key = jax.random.split(key, 3)
    preprocess_fn = eqx.filter_jit(mff.preprocess_for_count_sphere_intersections)
    preprocess1 = preprocess_fn(*get_x_and_r(pdb1), 500, k1)
    del k1
    preprocess2 = preprocess_fn(*get_x_and_r(pdb2), 1000, k2)
    del k2

    n_iter = 100
    k1, key = jax.random.split(key, 2)
    ind = jax.random.randint(  # pyright: ignore [reportUnknownMemberType]
        k1,
        (n_iter, 3),
        minval=0,
        maxval=jnp.array(params[3].shape),  # pyright: ignore [reportUnknownMemberType]
    )
    del k1
    count_fn = eqx.filter_jit(mff.count_sphere_intersections_fast)
    for i in range(n_iter):
        r_ind, phi1_ind, phi2_ind = ind[i]
        count = count_fn(
            preprocess1,
            preprocess2,
            jnp.zeros(3),  # pyright: ignore [reportUnknownMemberType]
            phi1[phi1_ind],
            jnp.array([r[r_ind], 0, 0]),  # pyright: ignore [reportUnknownMemberType]
            phi2[phi2_ind],
        )
        assert params[3][r_ind, phi1_ind, phi2_ind] == count


def test_pdb_hashes(tmp_path: pathlib.Path):
    out_path = str(tmp_path / "out.safetensors")
    pdb_path1 = "tests/1rwt_trimer_opm.pdb"
    pdb_path2 = "tests/6rqf_dimer_opm.pdb"
    key = jax.random.key(0)

    r = jnp.linspace(0.0, 50.0, 10)  # pyright: ignore [reportUnknownMemberType]
    phi1 = jnp.linspace(0.0, 2 * jnp.pi, 10)  # pyright: ignore [reportUnknownMemberType]
    phi2 = jnp.linspace(0.0, 2 * jnp.pi, 10)  # pyright: ignore [reportUnknownMemberType]
    serialize_force_field(
        out_path,
        pdb_path1,
        500,
        pdb_path2,
        1000,
        r,
        phi1,
        phi2,
        key,
    )

    with pytest.raises(
        Exception, match="pdb1 does not match pdb1 used to create this force field"
    ):
        deserialize_force_field(out_path, pdb_path2, pdb_path2)
    with pytest.raises(
        Exception, match="pdb2 does not match pdb2 used to create this force field"
    ):
        deserialize_force_field(out_path, pdb_path1, pdb_path1)
