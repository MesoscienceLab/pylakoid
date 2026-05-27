import typing

import Bio.PDB.Structure
import jax.numpy as jnp
from jaxtyping import Array, Int, Float

from pylakoid.structure.vdw_radii import vdw_radii


def get_radius(pdb: Bio.PDB.Structure.Structure) -> Float[Array, ""]:
    """
    Compute the radius of a protein, accounting for the radii of each atom.

    Parameters:
        pdb: The protein for which the radius will be computed.

    Returns:
        The radius of the protein.
    """
    x = jnp.array([atom.coord[:2] for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    r = jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    return jnp.max(typing.cast(Float[Array, " a"], jnp.linalg.norm(x, axis=1)) + r)  # pyright: ignore [reportUnknownMemberType]


def prepare_radius(
    pdb: dict[str, Bio.PDB.Structure.Structure],
    protein_type_map: dict[str, int],
    protein_types: Int[Array, " p"],
) -> Float[Array, " p"]:
    """
    Prepare the `radius` parameter for `Membrane.__init__`.

    Parameters:
        pdb: A dictionary mapping `str` protein type to PDB.
        protein_type_map: A dictionary mapping `str` protein type to internal `int` `protein_type`.
        protein_types: The list of types for every protein you want to include in the `Membrane`. If
            the same protein appears multiple times in the membrane, the corresponding protein type
            should be repeated.

    Returns:
        The radius of the protein.
    """
    sorted_protein_types = sorted(protein_type_map.items(), key=lambda x: x[1])
    pdb_radius = jnp.array(  # pyright: ignore [reportUnknownMemberType]
        [get_radius(pdb[name]) for name, _pt in sorted_protein_types]
    )
    return pdb_radius[protein_types]


def prepare_protein_type(
    protein_types: list[str],
) -> tuple[Int[Array, " p"], dict[str, int]]:
    """
    Prepare the `protein_type` parameter for `Membrane.__init__` from a list of protein types.

    Parameters:
        protein_types: The list of types for every protein you want to include in the `Membrane`. If
            the same protein appears multiple times in the membrane, the corresponding protein type
            should be repeated.

    Returns:
        A tuple containing
            - The `protein_type` for the proteins in the membrane in the format needed by `Membrane.__init__`
            - A map from `str` protein type to internal `int` `protein_type`.
    """

    cache: dict[str, int] = {}
    for t in protein_types:
        if t not in cache:
            cache[t] = len(cache)

    pt = jnp.array([cache[t] for t in protein_types], dtype=jnp.int32)  # pyright: ignore [reportUnknownMemberType]
    return (pt, cache)


def prepare_swappable(
    can_swap: list[tuple[str, str]], protein_type_map: dict[str, int]
) -> Int[Array, "s s"]:
    """
    Prepare the `swappable` parameter for `run_monte_carlo` from a list of swappable pairs and
    a protein type map (see `prepare_protein_type`).

    Parameters:
        can_swap: A list of pairs of `str` protein types that can be swapped.
        protein_type_map: A dictionary mapping `str` protein type to internal `int` `protein_type`.

    Returns:
        The `protein_type` for the proteins in the membrane in the format needed by
            `Membrane.__init__`
    """

    num_types = max([t + 1 for t in protein_type_map.values()])
    swappable = jnp.zeros((num_types, num_types), dtype=jnp.bool)  # pyright: ignore [reportUnknownMemberType]
    for i_str, j_str in can_swap:
        i = protein_type_map[i_str]
        j = protein_type_map[j_str]
        swappable = swappable.at[i, j].set(True)
        swappable = swappable.at[j, i].set(True)
    return swappable
