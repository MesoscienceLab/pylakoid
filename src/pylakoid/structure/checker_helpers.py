import Bio.PDB.Structure
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float
import shapely

from pylakoid.structure import checker


def prepare_boundary(boundary: shapely.Polygon) -> Float[Array, "b 2"]:
    """
    Prepare the `boundary` parameter for `InsidePolygonChecker.__init__` from a `shapely.Polygon`.

    Parameters:
        boundary: The boundary of the membrane.

    Returns:
        The boundary of the membrane in the format needed by `InsidePolygonChecker.__init__`.
    """

    return jnp.array(boundary.exterior.coords)  # pyright: ignore [reportUnknownMemberType]


def get_atomic_hull2d(pdb: Bio.PDB.Structure.Structure) -> Float[Array, "h 2"]:
    """
    Compute the 2d convex hull of atoms from a PDB.

    Parameters:
        pdb: The PDB from which you want to get the 2d convex hull.

    Returns:
        The 2d convex hull of `pdb`.
    """

    x = np.array([atom.coord[:2] for atom in pdb.get_atoms()])
    convex_hull = shapely.convex_hull(shapely.MultiPoint(x))  # pyright: ignore [reportUnknownMemberType]
    if not isinstance(convex_hull, shapely.Polygon):
        raise Exception("invalid convex hull")
    return jnp.array(convex_hull.exterior.coords)  # pyright: ignore [reportUnknownMemberType]


def prepare_atomic_hull_inside_polygon_checker(
    boundary: Float[Array, "b 2"],
    pdb: dict[str, Bio.PDB.Structure.Structure],
    protein_type_map: dict[str, int],
) -> checker.InsidePolygonChecker:
    """
    Prepare an `InsidePolygonChecker` using the 2d atomic hulls of proteins.

    Parameters:
        boundary: The polygon that proteins must be inside of.
        pdb: A dictionary mapping `str` protein type to PDB.
        protein_type_map: A dictionary mapping `str` protein type to internal `int` `protein_type`.

    Returns:
        The `InsidePolygonChecker`.
    """
    sorted_protein_types = sorted(protein_type_map.items(), key=lambda x: x[1])
    points_list = [get_atomic_hull2d(pdb[name]) for name, _pt in sorted_protein_types]
    points = jnp.concat(points_list, axis=0)
    type_index = jnp.pad(  # pyright: ignore [reportUnknownMemberType]
        jnp.cumsum(jnp.array([p.shape[0] for p in points_list], dtype=jnp.int32)),  # pyright: ignore [reportUnknownMemberType]
        (1, 0),
    )
    return checker.InsidePolygonChecker(boundary, points, type_index)
