import typing

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key


class Preprocess(eqx.Module):
    """
    Stores the preprocessed information used to count sphere intersections fast.

    Attributes:
        x: The centers of the spheres, sorted so that each group corresponds to a
            contiguous segment.
        r: The radii of the spheres, sorted so that `r[i]` is the radius of the
            sphere centered at `x[i]`.
        group_index: The group boundaries in `x`, so `x[group_index[i]:group_index[i+1]]`
            contains the centers of all these spheres in group `i`.
        group_center: The representative point of each group.
        group_radius: The maximum possible distance of any point in any sphere from
            the group's representative point.
    """

    x: Float[Array, "p 3"]
    r: Float[Array, " p"]
    group_index: Int[Array, " a"]
    group_center: Float[Array, " b"]
    group_radius: Float[Array, " b"]

    def __init__(
        self,
        x: Float[Array, "p 3"],
        r: Float[Array, " p"],
        group_index: Int[Array, " a"],
        group_center: Float[Array, " b"],
        group_radius: Float[Array, " b"],
    ):
        """
        Initialize a `Preprocess`.

        Attributes:
            x: The centers of the spheres, sorted so that each group corresponds to a
                contiguous segment.
            r: The radii of the spheres, sorted so that `r[i]` is the radius of the
                sphere centered at `x[i]`.
            group_index: The group boundaries in `x`, so `x[group_index[i]:group_index[i+1]]`
                contains the centers of all these spheres in group `i`.
            group_center: The representative point of each group.
            group_radius: The maximum possible distance of any point in any sphere from
                the group's representative point.
        """
        self.x = x
        self.r = r
        self.group_index = group_index
        self.group_center = group_center
        self.group_radius = group_radius


def preprocess_for_count_sphere_intersections(
    x: Float[Array, "p 3"],
    r: Float[Array, " p"],
    n: int,
    key: Key[Array, ""],
) -> Preprocess:
    """
    Preprocess a set of spheres to facilitate fast counting of sphere intersections.

    Parameters:
        x: The centers of all the spheres in the set.
        r: The radii of all the spheres in the set, so `r[i]` is the radius of the sphere
            centered at `x[i]`.
        n: The number of groups to produce. Good values tend to be around `x.shape[0]^(2/3)`
            but the optimal will vary based on the specific set of spheres.
        key: A JAX random key used to generate random numbers. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

    Returns:
        The preprocessed information.
    """
    selected = jax.random.choice(key, x, (n,), replace=False)  # pyright: ignore [reportUnknownMemberType]

    def f(u: Float[Array, "3"], v: Float[Array, "3"]) -> Float[Array, ""]:
        return typing.cast(Float[Array, ""], jnp.linalg.norm(u - v))

    distance_from_selected = jax.vmap(jax.vmap(f, (None, 0)), (0, None))(x, selected)
    group = jnp.argmin(distance_from_selected, axis=1)

    ind = jnp.argsort(group)
    sorted_group = group[ind]
    sorted_x = x[ind]
    sorted_r = r[ind]

    group_size = jnp.unique_counts(group, size=n, fill_value=n)[1]
    group_index = jnp.pad(jnp.cumsum(group_size), (1, 0))  # pyright: ignore [reportUnknownMemberType]

    def calc_group_sum(
        i: Int[Array, ""], res: Float[Array, "n 3"]
    ) -> Float[Array, "n 3"]:
        return res.at[sorted_group[i]].add(sorted_x[i])

    group_sum = typing.cast(
        Float[Array, "n 3"],
        jax.lax.fori_loop(0, sorted_group.shape[0], calc_group_sum, jnp.zeros((n, 3))),  # pyright: ignore [reportUnknownMemberType]
    )
    group_center = group_sum / jnp.expand_dims(group_size, 1)

    dist_from_group_rep = (
        typing.cast(
            Float[Array, " p"],
            jnp.linalg.norm(sorted_x - group_center[sorted_group], axis=1),
        )
        + sorted_r
    )

    def g(i: Int[Array, ""], res: Float[Array, " n"]) -> Float[Array, " n"]:
        g = sorted_group[i]
        dist = dist_from_group_rep[i]
        return res.at[g].max(dist)

    group_radius = typing.cast(
        Float[Array, " n"],
        jax.lax.fori_loop(0, x.shape[0], g, jnp.zeros(n)),  # pyright: ignore [reportUnknownMemberType]
    )

    return Preprocess(sorted_x, sorted_r, group_index, group_center, group_radius)


def count_sphere_intersections_between_groups(
    x1: Float[Array, "p 3"],
    r1: Float[Array, " p"],
    g1_start: Int[Array, ""],
    g1_end: Int[Array, ""],
    x2: Float[Array, "p 3"],
    r2: Float[Array, " p"],
    g2_start: Int[Array, ""],
    g2_end: Int[Array, ""],
) -> Int[Array, ""]:
    """
    Count the intersections between two sets of spheres which are each subsets of another
    set of spheres. This function is a useful building block for a full fast intersection
    counting algorithm, because it allows counting for any group size without using
    dynamically sized arrays (which are not compatible with JAX just-in-time compilation).

    Parameters:
        x1: The centers of the first set of spheres.
        r1: The radii of the first set of spheres, so `r1[i]` is the radius of the sphere
            centered at `x1[i]`.
        g1_start: The start index of the subset of the first set of spheres.
        g1_end: The end index of the subset of the first set of spheres.
        x2: The centers of the second set of spheres.
        r2: The radii of the second set of spheres, so `r2[i]` is the radius of the sphere
            centered at `x2[i]`.
        g2_start: The start index of the subset of the second set of spheres.
        g2_end: The end index of the subset of the second set of spheres.

    Returns:
        The number of sphere intersections between the subsets.
    """

    def outer(i1: Int[Array, ""], res: Int[Array, ""]) -> Int[Array, ""]:
        def inner(i2: Int[Array, ""], res: Int[Array, ""]) -> Int[Array, ""]:
            dist = typing.cast(Float[Array, ""], jnp.linalg.norm(x1[i1] - x2[i2]))
            return res + (dist <= r1[i1] + r2[i2])

        return res + typing.cast(
            Int[Array, ""],
            jax.lax.fori_loop(g2_start, g2_end, inner, jnp.zeros_like(x1[0, 0])),  # pyright: ignore [reportUnknownMemberType]
        )

    return typing.cast(
        Int[Array, ""],
        jax.lax.fori_loop(g1_start, g1_end, outer, jnp.zeros_like(x1[0, 0])),  # pyright: ignore [reportUnknownMemberType]
    )


def count_sphere_intersections_fast(
    preprocess1: Preprocess,
    preprocess2: Preprocess,
    com1: Float[Array, "3"],
    angle1: Float[Array, ""],
    com2: Float[Array, "3"],
    angle2: Float[Array, ""],
) -> Int[Array, ""]:
    """
    Count the intersections between pairs of spheres in two sets. This requires the sets of
    spheres to be preprocessed. Then the preprocessed information is transformed by `com1`,
    `angle1`, `com2`, and `angle2` before actually counting intersections.

    Parameters:
        preprocess1: The preprocessed information about the first set of spheres.
        preprocess2: The preprocessed information about the second set of spheres.
        com1: Vector to translate the first set of spheres (after rotation).
        angle1: Angle to rotate the first set of spheres about the z-axis.
        com2: Vector to translate the second set of spheres (after rotation).
        angle2: Angle to rotate the second set of spheres about the z-axis.

    Returns:
        The total number of sphere intersections.
    """

    def rotate(x: Float[Array, "p 3"], angle: Float[Array, ""]) -> Float[Array, "p 3"]:
        cos = jnp.cos(angle)
        sin = jnp.sin(angle)
        rot = jnp.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])  # pyright: ignore [reportUnknownMemberType]
        return jnp.matrix_transpose(rot @ jnp.matrix_transpose(x))

    x1 = rotate(preprocess1.x, angle1) + com1
    group_center1 = rotate(preprocess1.group_center, angle1) + com1
    x2 = rotate(preprocess2.x, angle2) + com2
    group_center2 = rotate(preprocess2.group_center, angle2) + com2

    def f(u: Float[Array, "3"], v: Float[Array, "3"]) -> Float[Array, ""]:
        return typing.cast(Float[Array, ""], jnp.linalg.norm(u - v))

    group_dist = jax.vmap(jax.vmap(f, (None, 0)), (0, None))(
        group_center1, group_center2
    )

    close = group_dist <= jnp.expand_dims(
        preprocess1.group_radius, 1
    ) + jnp.expand_dims(preprocess2.group_radius, 0)

    def body(i: Int[Array, ""], res: Int[Array, ""]) -> Int[Array, ""]:
        g2 = i % close.shape[1]
        g1 = ((i - g2) // close.shape[1]) % close.shape[0]

        def do_count():
            return res + count_sphere_intersections_between_groups(
                x1,
                preprocess1.r,
                preprocess1.group_index[g1],
                preprocess1.group_index[g1 + 1],
                x2,
                preprocess2.r,
                preprocess2.group_index[g2],
                preprocess2.group_index[g2 + 1],
            )

        return typing.cast(
            Int[Array, ""],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                close[g1, g2],
                do_count,
                lambda: res,  # pyright: ignore [reportUnknownMemberType]
            ),
        )

    return typing.cast(
        Int[Array, ""],
        jax.lax.fori_loop(0, close.size, body, jnp.zeros_like(x1[0, 0])),  # pyright: ignore [reportUnknownMemberType]
    )


def count_sphere_intersections_parallel(
    preprocess1: Preprocess,
    preprocess2: Preprocess,
    r: Float[Array, " r"],
    phi1: Float[Array, " p"],
    phi2: Float[Array, " q"],
    key: Key[Array, ""],
) -> Float[Array, "r p q"]:
    """
    Count the intersections between pairs of spheres in two sets in a variety of configurations.
    Performs calculations in parallel using JAX. If running on CPU, add the environment variable
    `XLA_FLAGS="--xla_force_host_platform_device_count=<NUMBER_OF_CPUS>"` (where you should
    substitute `<NUMBER_OF_CPUS>` for the actual number you want to use) to parallelize.

    Parameters:
        preprocess1: The preprocessed information about the first set of spheres.
        preprocess2: The preprocessed information about the second set of spheres.
        r: Translate the second set of spheres by `r` along the x axis (after rotation).
        phi1: Angle to rotate the first set of spheres about the z-axis.
        phi2: Angle to rotate the second set of spheres about the z-axis.
        key: A JAX random key used to generate random numbers. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

    Returns:
        The number of intersections in various configurations. If the result is called `counts`
        then `counts[i,j,k]` is the number of intersections with configuration defined by
        `r[i]`, `phi1[j]`, and `phi2[k]`.
    """
    r_grid, phi1_grid, phi2_grid = jnp.meshgrid(r, phi1, phi2, indexing="ij")
    args = jnp.stack(
        [
            jnp.ravel(r_grid),
            jnp.ravel(phi1_grid),
            jnp.ravel(phi2_grid),
        ],
        axis=1,
    )

    n_dev = jax.device_count()  # pyright: ignore [reportUnknownMemberType]
    excess = args.shape[0] % n_dev
    pad = n_dev - excess if excess != 0 else 0
    args = jnp.pad(args, ((0, pad), (0, 0)))  # pyright: ignore [reportUnknownMemberType]

    perm = jax.random.permutation(key, args.shape[0])  # pyright: ignore [reportUnknownMemberType]
    inverse_perm = jnp.zeros_like(perm).at[perm].set(jnp.arange(args.shape[0]))  # pyright: ignore [reportUnknownMemberType]
    args = args[perm]

    def f(arg: Float[Array, "3"]) -> Int[Array, ""]:
        return count_sphere_intersections_fast(
            preprocess1,
            preprocess2,
            jnp.zeros(3),  # pyright: ignore [reportUnknownMemberType]
            arg[1],
            jnp.array([arg[0], 0, 0]),  # pyright: ignore [reportUnknownMemberType]
            arg[2],
        )

    def g(args: Float[Array, "n 3"]) -> Int[Array, " n"]:
        return typing.cast(Int[Array, " n"], jax.lax.map(f, args))  # pyright: ignore [reportUnknownMemberType]

    mesh = jax.make_mesh((n_dev,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))  # pyright: ignore [reportUnknownMemberType]
    counts = typing.cast(
        Int[Array, " n"],
        jax.shard_map(  # pyright: ignore [reportUnknownMemberType]
            g,
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec("x"),
            out_specs=jax.sharding.PartitionSpec("x"),
        )(args),
    )
    counts = jnp.array(  # pyright: ignore [reportUnknownMemberType]
        counts,
        out_sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
    )
    counts = counts[inverse_perm]
    counts = counts[:-pad] if pad > 0 else counts

    return jnp.reshape(counts, (r.shape[0], phi1.shape[0], phi2.shape[0]))  # pyright: ignore [reportUnknownMemberType]
