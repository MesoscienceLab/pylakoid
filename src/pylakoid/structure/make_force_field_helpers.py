import hashlib
import json
import struct

import Bio.PDB.Structure
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key
import numpy as np
import safetensors.numpy

from pylakoid.structure import make_force_field as mff
from pylakoid.structure.pdb import parse_pdb
from pylakoid.structure.vdw_radii import vdw_radii


def hash_file(path: str) -> str:
    """
    Calculate the SHA-256 hash of the bytes contained in the file at `path`.

    Parameters:
        path: The path to the file that will be hashed.

    Returns:
        The 32 character hexadecimal digest of the SHA-256 hash of the file.
    """
    with open(path, "rb") as f:
        content = f.read()
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()


def serialize_force_field(
    out_path: str,
    pdb_path1: str,
    num_groups1: int,
    pdb_path2: str,
    num_groups2: int,
    r: Float[Array, " r"],
    phi1: Float[Array, " p"],
    phi2: Float[Array, " q"],
    key: Key[Array, ""],
):
    """
    Calculate and serialize a force field. Performs calculations in parallel using JAX.
    If running on CPU, add the environment variable
    `XLA_FLAGS="--xla_force_host_platform_device_count=<NUMBER_OF_CPUS>"` (where you should
    substitute `<NUMBER_OF_CPUS>` for the actual number you want to use) to parallelize.

    Parameters:
        out_path: Where the force field data will be written.
        pdb_path1: The path to the PDB file for the first protein.
        num_groups1: The number of groups that should be used when preprocessing the first protein.
        pdb_path2: The path to the PDB file for the second protein.
        num_groups2: The number of groups that should be used when preprocessing the second protein.
        r: Translate the second set of spheres by `r` along the x axis (after rotation).
        phi1: Angle to rotate the first set of spheres about the z-axis.
        phi2: Angle to rotate the second set of spheres about the z-axis.
        key: A JAX random key used to generate random numbers. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)
    """
    is_f64 = jnp.array(0).dtype == jnp.float64  # pyright: ignore [reportUnknownMemberType]
    metadata = {
        "pdb1_hash": hash_file(pdb_path1),
        "pdb2_hash": hash_file(pdb_path2),
        "precision": "float64" if is_f64 else "float32",
    }

    pdb1 = parse_pdb("protein1", pdb_path1)
    pdb2 = parse_pdb("protein2", pdb_path2)

    def get_x_and_r(
        pdb: Bio.PDB.Structure.Structure,
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        return (
            jnp.array([atom.coord for atom in pdb.get_atoms()]),  # pyright: ignore [reportUnknownMemberType]
            jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()]),  # pyright: ignore [reportUnknownMemberType]
        )

    k1, k2, key = jax.random.split(key, 3)
    preprocess_fn = eqx.filter_jit(mff.preprocess_for_count_sphere_intersections)
    preprocess1 = preprocess_fn(*get_x_and_r(pdb1), num_groups1, k1)
    del k1
    preprocess2 = preprocess_fn(*get_x_and_r(pdb2), num_groups2, k2)
    del k2

    count = eqx.filter_jit(mff.count_sphere_intersections_parallel)(
        preprocess1, preprocess2, r, phi1, phi2, key
    )
    del key

    tensors = {
        "r": np.asarray(r),
        "phi1": np.asarray(phi1),
        "phi2": np.asarray(phi2),
        "count": np.asarray(count),
    }
    safetensors.numpy.save_file(tensors, out_path, metadata)  # pyright: ignore [reportUnknownMemberType]


def parse_metadata(in_path: str) -> dict[str, str]:
    """
    Parse the metadata in a safetensors file.

    Parameters:
        in_path: The path to the safetensors file.

    Returns:
        The metadata dictionary contained in the file.
    """
    with open(in_path, "rb") as f:
        # Read the first 8 bytes as little endian
        header_length = struct.unpack("<Q", f.read(8))[0]
        # Read header_length bytes as UTF-8 then parse as JSON
        header = json.loads(f.read(header_length).decode())
        return header["__metadata__"]


def deserialize_force_field(
    in_path: str,
    pdb_path1: str,
    pdb_path2: str,
) -> tuple[
    Float[Array, " r"],
    Float[Array, " p"],
    Float[Array, " q"],
    Int[Array, "r p q"],
]:
    """
    Deserialize a force field. Validates that the PDB hashes from the metadata match.

    Parameters:
        in_path: The path to the safetensors file.
        pdb_path1: The path to the PDB file for the first protein.
        pdb_path2: The path to the PDB file for the second protein.

    Returns:
        The `r` parameter from when the force field was serialized
        The `phi1` parameter from when the force field was serialized
        The `phi2` parameter from when the force field was serialized
        The counts corresponding to specific values of `(r, phi1, phi2)`
    """
    metadata = parse_metadata(in_path)
    if metadata["pdb1_hash"] != hash_file(pdb_path1):
        raise Exception("pdb1 does not match pdb1 used to create this force field")
    if metadata["pdb2_hash"] != hash_file(pdb_path2):
        raise Exception("pdb2 does not match pdb2 used to create this force field")

    tensors = safetensors.numpy.load_file(in_path)  # pyright: ignore [reportUnknownMemberType]
    return (
        jnp.asarray(tensors["r"]),  # pyright: ignore [reportUnknownMemberType]
        jnp.asarray(tensors["phi1"]),  # pyright: ignore [reportUnknownMemberType]
        jnp.asarray(tensors["phi2"]),  # pyright: ignore [reportUnknownMemberType]
        jnp.asarray(tensors["count"]),  # pyright: ignore [reportUnknownMemberType]
    )
