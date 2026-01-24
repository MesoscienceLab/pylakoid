import typing

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class Checker(typing.Protocol):
    """
    Type for functions that can be called to check is a configuration is valid while annealing membranes.
    """

    def __call__(
        self,
        com: Float[Array, "2"],
        angle: Float[Array, ""],
        pt: Int[Array, ""],
        index: Int[Array, ""],
        out_sharding: jax.sharding.PartitionSpec | None = None,
    ) -> Bool[Array, ""]:
        """
        Check if the configuration is valid.

        Parameters:
            com: The center of mass of the protein.
            angle: The angle of the protein.
            pt: The protein type of the protein.
            index: The index of the protein.
            out_sharding: The sharding of the result, used for JAX parallelization.

        Returns:
            `True` if the configuration is valid, `False` otherwise.
        """
        ...


def is_point_in_polygon(
    point: Float[Array, "2"],
    vertices: Float[Array, "m 2"],
) -> Bool[Array, ""]:
    """
    Use the ray-casting algorithm with ray directed toward the right to determine if a point is
    inside of an arbitrary polygon. This does not assume the polygon is convex. This implementation
    doesn't handle the edge case where the ray hits a vertex exactly.

    Parameters:
        point: The 2d point that you are testing.
        vertices: The vertices of the polygon that may or may not contain `point`. The behavior is
            unchanged if vertices is padded with the initial vertex.

    Returns:
        `True` if `point` is inside `vertices`, `False` otherwise.
    """
    v = jnp.concat([vertices, jnp.expand_dims(vertices[0], 0)], axis=0)

    edge_x = jnp.stack([v[:-1, 0], v[1:, 0]], axis=1)
    edge_y = jnp.stack([v[:-1, 1], v[1:, 1]], axis=1)
    y_min_ind = jnp.argmin(edge_y, axis=1)

    x1 = edge_x[:, y_min_ind]
    x2 = edge_x[:, 1 - y_min_ind]
    y1 = edge_y[:, y_min_ind]
    y2 = edge_y[:, 1 - y_min_ind]

    x = point[0]
    y = point[1]
    valid_x = x * (y2 - y1) >= (x2 - x1) * (y - y1) + x1 * (y2 - y1)
    valid_y = (y1 < y) & (y < y2)
    return jnp.sum(valid_x * valid_y) % 2 == 1


def transform2d(
    x: Float[Array, "2"],
    com: Float[Array, "2"],
    angle: Float[Array, ""],
) -> Float[Array, "2"]:
    """
    Uses `com` and `angle` to transform `x`.

    Parameters:
        x: The 2d vectors to transform.
        com: The amount to translate `x` (after rotation).
        angle: The angle to rotate `x` about the z axis.

    Returns:
        The transformed `x`.
    """
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    r = jnp.array([[cos, -sin], [sin, cos]])  # pyright: ignore [reportUnknownMemberType]
    return r @ x + com


class InsidePolygonChecker(eqx.Module):
    """
    Checks if a set of points is inside a polygon, where the set of points is determined by protein type.

    Attributes:
        boundary: The polygon that must contain the points.
        points: The set of all points for all protein types, grouped such that all points
            for a single protein type are contiguous.
        type_index: The boundaries of the points for each protein type, so
            `points[type_index[i]:type_index[i+1]]` corresponds to all points for a protein
            of type `i`.
    """

    boundary: Float[Array, "b 2"]
    points: Float[Array, "n 2"]
    type_index: Int[Array, " t"]

    def __init__(
        self,
        boundary: Float[Array, "b 2"],
        points: Float[Array, "n 2"],
        type_index: Int[Array, " t"],
    ):
        """
        Initialize an `InsidePolygonChecker`.

        Parameters:
            boundary: The polygon that must contain the points.
            points: The set of all points for all protein types, grouped such that all points
                for a single protein type are contiguous.
            type_index: The boundaries of the points for each protein type, so
                `points[type_index[i]:type_index[i+1]]` corresponds to all points for a protein
                of type `i`.
        """
        self.boundary = boundary
        self.points = points
        self.type_index = type_index

    def __call__(
        self,
        com: Float[Array, "2"],
        angle: Float[Array, ""],
        pt: Int[Array, ""],
        index: Int[Array, ""],
        out_sharding: jax.sharding.PartitionSpec | None = None,
    ) -> Bool[Array, ""]:
        """
        Check that the points associated with protein type `pt` are inside the boundary after
        transforming by `com` and `angle`.

        Parameters:
            com: The center of mass of the protein to check.
            angle: The angle of the protein to check.
            pt: The protein type of the protein to check.
            index: The index of the protein to check. Unused.
            out_sharding: The sharding of the result, used for JAX parallelization.

        Returns:
            `True` if inside the boundary, `False` otherwise.
        """

        def f(i: Int[Array, ""], res: Bool[Array, ""]) -> Bool[Array, ""]:
            return res & is_point_in_polygon(
                transform2d(self.points[i], com, angle), self.boundary
            )

        vary = None if out_sharding is None else tuple(out_sharding)  # pyright: ignore [reportUnknownVariableType]
        initial_value = typing.cast(
            Float[Array, ""],
            jax.lax.pvary(jnp.array(True), vary),  # pyright: ignore [reportUnknownMemberType]
        )
        return typing.cast(
            Bool[Array, ""],
            jax.lax.fori_loop(  # pyright: ignore [reportUnknownMemberType]
                self.type_index[pt],
                self.type_index[pt + 1],
                f,
                initial_value,
            ),
        )


class CenterOfMassInCircleChecker(eqx.Module):
    """
    Checks if the center of mass of a protein is inside a circle, where the circle depends on the index of the protein.

    Attributes:
        center: The 2d center of the circle, where `center[i]` is the center for the protein with index `i`.
        radius: The radius of the circle, where `radius[i]` is the radius for the protein with index `i`.
    """

    center: Float[Array, "p 2"]
    radius: Float[Array, " p"]

    def __init__(
        self,
        center: Float[Array, "p 2"],
        radius: Float[Array, " p"],
    ):
        """
        Initialize a `CenterOfMassInCircleChecker`.

        Parameters:
            center: The 2d center of the circle, where `center[i]` is the center for the protein with index `i`.
            radius: The radius of the circle, where `radius[i]` is the radius for the protein with index `i`.
        """
        self.center = center
        self.radius = radius

    def __call__(
        self,
        com: Float[Array, "2"],
        angle: Float[Array, ""],
        pt: Int[Array, ""],
        index: Int[Array, ""],
        out_sharding: jax.sharding.PartitionSpec | None = None,
    ) -> Bool[Array, ""]:
        """
        Check that `com` is inside the `circle[i]`.

        Parameters:
            com: The center of mass of the protein to check.
            angle: The angle of the protein to check. Unused.
            pt: The protein type of the protein to check. Unused.
            index: The index of the protein to check.
            out_sharding: The sharding of the result, used for JAX parallelization. Unused.

        Returns:
            `True` if inside the circle, `False` otherwise.
        """
        dist = typing.cast(Float[Array, ""], jnp.linalg.norm(com - self.center[index]))  # pyright: ignore [reportUnknownMemberType]
        return dist <= self.radius[index]


class MultiChecker(eqx.Module):
    """
    Check if multiple checkers are all valid.

    Attributes:
        checkers: The checkers to apply.
    """

    checkers: tuple[Checker, ...]

    def __init__(
        self,
        checkers: tuple[Checker, ...],
    ):
        """
        Initialize a `MultiChecker`.

        Parameters:
            checkers: The checkers to apply.
        """
        self.checkers = checkers

    def __call__(
        self,
        com: Float[Array, "2"],
        angle: Float[Array, ""],
        pt: Int[Array, ""],
        index: Int[Array, ""],
        out_sharding: jax.sharding.PartitionSpec | None = None,
    ) -> Bool[Array, ""]:
        """
        Check that all of the checkers are valid.

        Parameters:
            com: The center of mass of the protein to check.
            angle: The angle of the protein to check.
            pt: The protein type of the protein to check.
            index: The index of the protein to check.
            out_sharding: The sharding of the result, used for JAX parallelization.

        Returns:
            `True` if all checkers return `True`, `False` otherwise.
        """
        res = jnp.array(True)  # pyright: ignore [reportUnknownMemberType]
        for c in self.checkers:
            res = res & c(com, angle, pt, index, out_sharding=out_sharding)
        return res
