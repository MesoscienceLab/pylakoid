import Bio.PDB.Structure
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import matplotlib.axes
import matplotlib.patches
import shapely

from pylakoid.structure import anneal
from pylakoid.structure.checker import transform2d
from pylakoid.structure.vdw_radii import vdw_radii


def union_batches(
    geometries: list[shapely.Geometry], batch_size: int
) -> list[shapely.Geometry]:
    """
    Process a list of geometries into batches and union those batches.

    Parameters:
        geometries: The list of geometries to union.
        batch_size: The number of geometries per batch.

    Returns:
        The list of batchwise unioned geometries.
    """

    batches: list[shapely.Geometry] = []
    for i, geo in enumerate(geometries):
        if i % batch_size == 0:
            batches.append(geo)
        else:
            batches[-1] = shapely.union(batches[-1], geo)  # pyright: ignore [reportUnknownMemberType]
    return batches


def get_protein_shape(pdb: Bio.PDB.Structure.Structure) -> Float[Array, "n 2"]:
    """
    Compute the exterior of a protein based on the projection of all atomic Van
    der Waals volumes onto the XY plane.

    Parameters:
        pdb: The protein for which the shape will be computed.

    Returns:
        The exterior of the protein.
    """

    geometries: list[shapely.Geometry] = [
        shapely.Point(atom.coord[:2]).buffer(vdw_radii[atom.element])  # pyright: ignore [reportUnknownMemberType]
        for atom in pdb.get_atoms()
    ]
    while len(geometries) >= 2:
        geometries = union_batches(geometries, 2)
    geometry = geometries[0]  # pyright: ignore [reportUnknownMemberType]
    if not isinstance(geometry, shapely.Polygon):
        raise Exception("invalid geometry")
    return jnp.array(geometry.exterior.coords)  # pyright: ignore [reportUnknownMemberType]


def set_axes_limits(
    ax: matplotlib.axes.Axes,
    boundary: Float[Array, "b 2"],
):
    """
    Set axis limits based on a given membrane boundary.

    Parameters:
        ax: The axes whose limits will be set.
        boundary: The boundary of the membrane.
    """

    xmin, ymin = jnp.min(boundary, axis=0).tolist()
    xmax, ymax = jnp.max(boundary, axis=0).tolist()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_proteins(
    ax: matplotlib.axes.Axes,
    membrane: anneal.Membrane,
    shapes: dict[str, Float[Array, "n 2"]],
    protein_type_map: dict[str, int],
    color_map: dict[int, str],
):
    """
    Plot all proteins in a membrane.

    Parameters:
        ax: The axes on which the proteins will be plotted.
        membrane: The membrane which contains the proteins to be plotted.
        shapes: The precomputed protein shapes. See `get_protein_shape`.
        protein_type_map: A dictionary mapping `str` protein type to internal `int` `protein_type`.
        color_map: The colors to be used when plotting the proteins.
    """

    shape_by_type = {pt: shapes[name] for name, pt in protein_type_map.items()}
    for i in range(len(membrane.protein_type)):
        color = color_map[membrane.protein_type[i].item()]
        x = shape_by_type[membrane.protein_type[i].item()]
        polygon = matplotlib.patches.Polygon(
            jax.vmap(transform2d, (0, None, None))(
                x, membrane.center_of_mass[i], membrane.angle[i]
            ),
            closed=True,
            facecolor=color,
            edgecolor="k",
            alpha=0.75,
        )
        ax.add_patch(polygon)


def plot_boundary(
    ax: matplotlib.axes.Axes,
    boundary: Float[Array, "b 2"],
):
    """
    Plot the boundary of a membrane.

    Parameters:
        ax: The axes on which the boundary will be plotted.
        boundary: The boundary of the membrane.
    """

    polygon = matplotlib.patches.Polygon(
        boundary,
        closed=True,
        fill=False,
        edgecolor="k",
    )
    ax.add_patch(polygon)
