from collections.abc import Callable
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Int

from pylakoid.structure import make_force_field as mff

ForceField = Callable[
    [
        Float[Array, "2"],
        Float[Array, ""],
        Int[Array, ""],
        Float[Array, "2"],
        Float[Array, ""],
        Int[Array, ""],
    ],
    Float[Array, ""],
]
"""
Type for functions that can be called to evaluate a force field while annealing membranes.

Parameters:
    com1 (Float[Array, "2"]): The center of mass of the first protein.
    angle1 (Float[Array, ""]): The angle of the first protein.
    type1 (Int[Array, ""]): The protein type of the first protein.
    com2 (Float[Array, "2"]): The center of mass of the second protein.
    angle2 (Float[Array, ""]): The angle of the second protein.
    type2 (Int[Array, ""]): The protein type of the second protein.

Returns:
    (Float[Array, ""]): The energy from the force field.
"""


class TypewiseForceField(eqx.Module):
    """
    Used to select the correct `ForceField` based on the protein types.

    Attributes:
        ff: The underlying `ForceField` objects. `ff[num_types*i+j]` will be used to calculate
            the force field if `pt1 = i` and `pt2 = j`. Must have length `num_types^2`.
        num_types: The number of protein types.
    """

    ff: tuple[ForceField, ...]
    num_types: Int[Array, ""]

    def __init__(
        self,
        ff: tuple[ForceField, ...],
        num_types: Int[Array, ""],
    ):
        """
        Initialize a `TypewiseForceField`.

        Parameters:
            ff: The underlying `ForceField` objects. `ff[num_types*i+j]` will be used to calculate
                the force field if `pt1 = i` and `pt2 = j`. Must have length `num_types^2`.
            num_types: The number of protein types.
        """
        self.ff = ff
        self.num_types = num_types

    def __call__(
        self,
        com1: Float[Array, "2"],
        angle1: Float[Array, ""],
        pt1: Int[Array, ""],
        com2: Float[Array, "2"],
        angle2: Float[Array, ""],
        pt2: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        Calculate the force field between two proteins based on their types `pt1` and `pt2`. This
        delegates the actual calculation to another `ForceField`.

        Parameters:
            com1: The 2d center of mass of the first protein.
            angle1: The angle to rotate the first protein about the z-axis.
            pt1: The type of the first protein.
            com1: The 2d center of mass of the second protein.
            angle2: The angle to rotate the second protein about the z-axis.
            pt2: The type of the second protein.

        Returns:
            The computed force field.
        """
        return typing.cast(
            Float[Array, ""],
            jax.lax.switch(  # pyright: ignore [reportUnknownMemberType]
                self.num_types * pt1 + pt2,
                [
                    lambda *args, ff=ff: ff(*args)  # pyright: ignore [reportUnknownLambdaType]
                    for ff in self.ff
                ],
                com1,
                angle1,
                pt1,
                com2,
                angle2,
                pt2,
            ),
        )


class SplineForceField(eqx.Module):
    """
    Approximate a force field using splines. This does not make any assumption about the spline
    method, as the spline calculations are delegated to
    `jax.scipy.interpolate.RegularGridInterpolator`.

    Attributes:
        spline: This will actually perform the spline calculations.
        pt1: The type of the first protein assumed by `spline`.
        pt2: The type of the second protein assumed by `spline`.
    """

    spline: jsp.interpolate.RegularGridInterpolator
    pt1: Int[Array, ""]
    pt2: Int[Array, ""]

    def __init__(
        self,
        spline: jsp.interpolate.RegularGridInterpolator,
        pt1: Int[Array, ""],
        pt2: Int[Array, ""],
    ):
        """
        Initialize a `SplineForceField`.

        Parameters:
            spline: This will actually perform the spline calculations.
            pt1: The type of the first protein assumed by `spline`.
            pt2: The type of the second protein assumed by `spline`.
        """
        self.spline = spline
        self.pt1 = pt1
        self.pt2 = pt2

    def __call__(
        self,
        com1: Float[Array, "2"],
        angle1: Float[Array, ""],
        pt1: Int[Array, ""],
        com2: Float[Array, "2"],
        angle2: Float[Array, ""],
        pt2: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        Calculate the force field between two proteins. It is the caller's responsibility to
        ensure that the sets `{ pt1, pt2 }` and `{ self.pt1, self.pt2 }` are equal, for
        example by wrapping this with `TypewiseForceField`.

        Parameters:
            com1: The 2d center of mass of the first protein.
            angle1: The angle to rotate the first protein about the z-axis.
            pt1: The type of the first protein.
            com1: The 2d center of mass of the second protein.
            angle2: The angle to rotate the second protein about the z-axis.
            pt2: The type of the second protein.

        Returns:
            The computed force field.
        """

        def f(
            com1: Float[Array, "2"],
            angle1: Float[Array, ""],
            pt1: Int[Array, ""],
            com2: Float[Array, "2"],
            angle2: Float[Array, ""],
            pt2: Int[Array, ""],
        ) -> Float[Array, ""]:
            # Choose pt1 to be the origin.
            com1, angle1, com2, angle2 = typing.cast(
                tuple[
                    Float[Array, "2"],
                    Float[Array, ""],
                    Float[Array, "2"],
                    Float[Array, ""],
                ],
                jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                    pt1 == self.pt1,
                    lambda: (com1, angle1, com2, angle2),
                    lambda: (com2, angle2, com1, angle1),
                ),
            )

            # Rotate the axes until the vector from pt1 to pt2 is on the positive x-axis.
            # You don't actually need to rotate com1 and com2 because the result only depends
            # on the norm of the difference.
            delta = com2 - com1
            theta = jnp.atan2(delta[1], delta[0])
            angle1 = (angle1 - theta) % (2 * jnp.pi)
            angle2 = (angle2 - theta) % (2 * jnp.pi)

            dist = typing.cast(Float[Array, ""], jnp.linalg.norm(delta))
            return typing.cast(
                Float[Array, ""],
                self.spline(jnp.stack([dist, angle1, angle2], axis=0))[0],
            )

        return (
            f(com1, angle1, pt1, com2, angle2, pt2)
            + f(com2, angle2, pt2, com1, angle1, pt1)
        ) / 2


class ClampedSplineForceField(eqx.Module):
    """
    Clamp a `SplineForceField` based on minimum distance `r_min` and maximum distance `r_max`.

    Attributes:
        sff: The `SplineForceField` used if `r_min <= |com1-com2| <= r_max`.
        r_min: The minimum distance to use `sff`. The force field returns `max_clashes` if distance
            is less than `r_min`.
        r_max: The maximum distance to use `sff`. The force field returns 0 if distance is greater
            than `r_max`.
        max_clashes: The value of the force field when distance is less than `r_min`.
    """

    sff: SplineForceField
    r_min: Float[Array, ""]
    r_max: Float[Array, ""]
    max_clashes: Float[Array, ""]

    def __init__(
        self,
        sff: SplineForceField,
        r_min: Float[Array, ""],
        r_max: Float[Array, ""],
        max_clashes: Float[Array, ""],
    ):
        """
        Initialize a `ClampedSplineForceField`.

        Parameters:
            sff: The `SplineForceField` used if `r_min <= |com1-com2| <= r_max`.
            r_min: The minimum distance to use `sff`. The force field returns `max_clashes` if
                distance is less than `r_min`.
            r_max: The maximum distance to use `sff`. The force field returns 0 if distance is
                greater than `r_max`.
            max_clashes: The value of the force field when distance is less than `r_min`.
        """
        self.sff = sff
        self.r_min = r_min
        self.r_max = r_max
        self.max_clashes = max_clashes

    def __call__(
        self,
        com1: Float[Array, "2"],
        angle1: Float[Array, ""],
        pt1: Int[Array, ""],
        com2: Float[Array, "2"],
        angle2: Float[Array, ""],
        pt2: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        Calculate the force field between two proteins. It is the caller's responsibility to
        ensure that the parameters respect the requirements of the underlying `SplineForceField`.

        Parameters:
            com1: The 2d center of mass of the first protein.
            angle1: The angle to rotate the first protein about the z-axis.
            pt1: The type of the first protein.
            com1: The 2d center of mass of the second protein.
            angle2: The angle to rotate the second protein about the z-axis.
            pt2: The type of the second protein.

        Returns:
            The computed force field.
        """
        dist = typing.cast(Float[Array, ""], jnp.linalg.norm(com2 - com1))
        return typing.cast(
            Float[Array, ""],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                (self.r_min <= dist) & (dist <= self.r_max),
                lambda: self.sff(com1, angle1, pt1, com2, angle2, pt2),
                lambda: jax.lax.select(
                    dist < self.r_min,
                    self.max_clashes,
                    jnp.array(0.0),  # pyright: ignore [reportUnknownMemberType]
                ),
            ),
        )


class ZeroForceField(eqx.Module):
    """
    Force field that always returns zero energy. Useful for testing.
    """

    def __call__(
        self,
        com1: Float[Array, "2"],
        angle1: Float[Array, ""],
        pt1: Int[Array, ""],
        com2: Float[Array, "2"],
        angle2: Float[Array, ""],
        pt2: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        Return zero energy regardless of configuration.

        Parameters:
            com1: The 2d center of mass of the first protein.
            angle1: The angle to rotate the first protein about the z-axis.
            pt1: The type of the first protein.
            com2: The 2d center of mass of the second protein.
            angle2: The angle to rotate the second protein about the z-axis.
            pt2: The type of the second protein.

        Returns:
            Zero energy.
        """
        return jnp.array(0.0)  # pyright: ignore [reportUnknownMemberType]


class AtomicOverlapForceField(eqx.Module):
    """
    Calculate the number of atomic overlaps using a fast method based on preprocessing. This is
    exact (up to numerical precision) so it is still probably slower than `SplineForceField` in
    most cases.

    Attributes:
        preprocess1: The preprocessed information about the first protein. See
            `pylakoid.structure.make_force_field.preprocess_for_count_sphere_intersections`.
        preprocess2: The preprocessed information about the second protein. See
            `pylakoid.structure.make_force_field.preprocess_for_count_sphere_intersections`.
    """

    preprocess1: mff.Preprocess
    preprocess2: mff.Preprocess

    def __init__(
        self,
        preprocess1: mff.Preprocess,
        preprocess2: mff.Preprocess,
    ):
        """
        Initialize an `AtomicOverlapForceField`.

        Parameters:
            preprocess1: The preprocessed information about the first protein. See
                `pylakoid.structure.make_force_field.preprocess_for_count_sphere_intersections`.
            preprocess2: The preprocessed information about the second protein. See
                `pylakoid.structure.make_force_field.preprocess_for_count_sphere_intersections`.
        """
        self.preprocess1 = preprocess1
        self.preprocess2 = preprocess2

    def __call__(
        self,
        com1: Float[Array, "2"],
        angle1: Float[Array, ""],
        pt1: Int[Array, ""],
        com2: Float[Array, "2"],
        angle2: Float[Array, ""],
        pt2: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        Calculate the force field between two proteins.

        Parameters:
            com1: The 2d center of mass of the first protein.
            angle1: The angle to rotate the first protein about the z-axis.
            pt1: The type of the first protein. Not used.
            com1: The 2d center of mass of the second protein.
            angle2: The angle to rotate the second protein about the z-axis.
            pt2: The type of the second protein. Not used.

        Returns:
            The computed force field.
        """
        return mff.count_sphere_intersections_fast(
            self.preprocess1,
            self.preprocess2,
            jnp.pad(com1, (0, 1)),  # pyright: ignore [reportUnknownMemberType]
            angle1,
            jnp.pad(com2, (0, 1)),  # pyright: ignore [reportUnknownMemberType]
            angle2,
        )
