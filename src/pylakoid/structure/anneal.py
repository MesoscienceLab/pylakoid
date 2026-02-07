from collections.abc import Callable
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from pylakoid.structure.checker import Checker
from pylakoid.structure.force_field import ForceField


class Membrane(eqx.Module):
    """
    Represents a membrane and all of the proteins it contains. Used for membrane annealing.

    Attributes:
        center_of_mass: The center of mass of every protein.
        angle: The angle of every protein.
        protein_type: The protein type of every protein. This is used for force field evaluation and for swapping.
        radius: The maximum norm of the 2d convex hull of every protein.
    """

    center_of_mass: Float[Array, "p 2"]
    angle: Float[Array, " p"]
    protein_type: Int[Array, " p"]
    radius: Float[Array, " p"]

    def __init__(
        self,
        center_of_mass: Float[Array, "p 2"],
        angle: Float[Array, " p"],
        radius: Float[Array, " p"],
        protein_type: Int[Array, " p"],
    ):
        """
        Initialize `Membrane`. This doesn't perform any sanity checks so passing correct
        parameters is the caller's responsibility.

        Parameters:
            center_of_mass: The center of mass of every protein.
            angle: The angle of every protein.
            protein_type: The protein type of every protein. This is used for force field evaluation and for swapping.
            radius: The maximum norm of the 2d convex hull of every protein.
        """
        self.center_of_mass = center_of_mass
        self.angle = angle
        self.radius = radius
        self.protein_type = protein_type


def one_particle_energy(
    membrane: Membrane,
    index: Int[Array, ""],
    force_field: ForceField,
    out_sharding: jax.sharding.PartitionSpec | None = None,
) -> Float[Array, ""]:
    """
    Calculate the energy of a single protein in `membrane` due to all other proteins.

    Parameters:
        membrane: The `Membrane` containing the protein.
        index: The index of the protein in `membrane` whose energy you want to calculate.
        force_field: The `ForceField` used for energy calculations.

    Returns:
        The energy of protein `index` in `membrane` due to all other proteins.
    """

    def body(i: Int[Array, ""], e_tot: Float[Array, ""]) -> Float[Array, ""]:
        this_com = membrane.center_of_mass[index]
        this_angle = membrane.angle[index]
        this_radius = membrane.radius[index]
        this_type = membrane.protein_type[index]
        other_com = membrane.center_of_mass[i]
        other_angle = membrane.angle[i]
        other_radius = membrane.radius[i]
        other_type = membrane.protein_type[i]
        dist = typing.cast(Float[Array, ""], jnp.linalg.norm(this_com - other_com))
        return typing.cast(
            Float[Array, ""],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                (i != index) & (dist < this_radius + other_radius),
                lambda: e_tot
                + force_field(
                    this_com, this_angle, this_type, other_com, other_angle, other_type
                ),
                lambda: e_tot,
            ),
        )

    vary = None if out_sharding is None else tuple(out_sharding)  # pyright: ignore [reportUnknownVariableType]
    initial_value = typing.cast(Float[Array, ""], jax.lax.pvary(jnp.array(0.0), vary))  # pyright: ignore [reportUnknownMemberType]
    return typing.cast(
        Float[Array, ""],
        jax.lax.fori_loop(0, membrane.center_of_mass.shape[0], body, initial_value),  # pyright: ignore [reportUnknownMemberType]
    )


def total_energy(
    membrane: Membrane,
    force_field: ForceField,
) -> Float[Array, ""]:
    """
    Calculate the total energy of all proteins in `membrane`.

    Parameters:
        membrane: The `Membrane` containing the protein.
        force_field: The `ForceField` used for energy calculations.

    Returns:
        The total energy of `membrane`.
    """

    def body(i: Int[Array, ""], e_tot: Float[Array, ""]) -> Float[Array, ""]:
        return e_tot + one_particle_energy(membrane, i, force_field) / 2

    return typing.cast(
        Float[Array, ""],
        jax.lax.fori_loop(0, len(membrane.protein_type), body, 0.0),  # pyright: ignore [reportUnknownMemberType])
    )


def run_monte_carlo(
    membrane: Membrane,
    translatable: Int[Array, " t"],
    swappable: Bool[Array, "s s"],
    rotatable: Int[Array, " r"],
    force_field: ForceField,
    checker: Checker,
    r_max: Float[Array, ""],
    phi_max: Float[Array, ""],
    kBT: Float[Array, ""],
    n_steps: int,
    key: Key[Array, ""],
    initial_energy: Float[Array, ""] | None = None,
    out_sharding: jax.sharding.PartitionSpec | None = None,
) -> tuple[Membrane, Float[Array, ""] | None]:
    """
    Run `n_steps` of Monte Carlo simulation where each step consists of

    - Translation (if `translatable.shape[0] > 0`)
    - Swapping (if `swappable.shape[0] > 0`)
    - Rotatable (if `rotatable.shape[0] > 0`)

    Parameters:
        membrane: The `Membrane` that you want to simulate.
        translatable: The indices of the proteins in `membrane` that can be translated.
        swappable: `swappable[t1, t2]` is `True` if proteins of type `t1` and `t2` can be swapped, and `False` otherwise. Must be symmetric.
        rotatable: The indices of the proteins in `membrane` that can be rotated.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        r_max: The maximum change in each coordinate of `membrane.center_of_mass[i]` when translating protein `i`.
        phi_max: The maximum change in `membrane.angle[i]` when rotating protein `i`.
        kBT: The energy scale used in the Metropolis-Hastings acceptance criteria.
        n_steps: The number of steps of Monte Carlo simulation to perform.
        key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)
        initial_energy: The energy of the input `Membrane` or `None`.

    Returns:
        The `Membrane` after the Monte Carlo simulation.
        The energy of the returned `Membrane` if `initial_energy is not None` otherwise `None`.
    """
    keys = jax.random.split(key, (n_steps, 3))

    def body(
        i: Int[Array, ""], args: tuple[Membrane, Float[Array, ""] | None]
    ) -> tuple[Membrane, Float[Array, ""] | None]:
        membrane, initial_energy = args
        if translatable.shape[0] > 0:
            membrane, initial_energy = sample_translation(
                membrane,
                translatable,
                force_field,
                checker,
                r_max,
                kBT,
                keys[i, 0],
                initial_energy=initial_energy,
                out_sharding=out_sharding,
            )
        membrane, initial_energy = sample_swap(
            membrane,
            swappable,
            force_field,
            checker,
            kBT,
            keys[i, 1],
            initial_energy=initial_energy,
            out_sharding=out_sharding,
        )
        if rotatable.shape[0] > 0:
            membrane, initial_energy = sample_rotation(
                membrane,
                rotatable,
                force_field,
                checker,
                phi_max,
                kBT,
                keys[i, 2],
                initial_energy=initial_energy,
                out_sharding=out_sharding,
            )
        return membrane, initial_energy

    return typing.cast(
        tuple[Membrane, Float[Array, ""] | None],
        jax.lax.fori_loop(0, n_steps, body, (membrane, initial_energy)),  # pyright: ignore [reportUnknownMemberType]
    )


def sample_rotation(
    membrane: Membrane,
    rotatable: Int[Array, " m"],
    force_field: ForceField,
    checker: Checker,
    phi_max: Float[Array, ""],
    kBT: Float[Array, ""],
    key: Key[Array, ""],
    initial_energy: Float[Array, ""] | None = None,
    out_sharding: jax.sharding.PartitionSpec | None = None,
) -> tuple[Membrane, Float[Array, ""] | None]:
    """
    Sample a rotation for the Monte Carlo simulation.

    Parameters:
        membrane: The `Membrane` that you want to simulate.
        rotatable: The indices of the proteins in `membrane` that can be rotated.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        phi_max: The maximum change in `membrane.angle[i]` when rotating protein `i`.
        kBT: The energy scale used in the Metropolis-Hastings acceptance criteria.
        key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)
        initial_energy: The energy of the input `Membrane` or `None`.

    Returns:
        The `Membrane` after the rotation.
        The energy of the returned `Membrane` if `initial_energy is not None` otherwise `None`.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    del key

    index = jax.random.choice(k1, rotatable)
    del k1

    rotation = jax.random.uniform(k2, (), minval=-phi_max, maxval=phi_max)  # pyright: ignore [reportUnknownMemberType]
    del k2

    new_angle = membrane.angle.at[index].add(rotation)
    new_membrane = Membrane(
        membrane.center_of_mass,
        new_angle,
        membrane.radius,
        membrane.protein_type,
    )
    is_valid = checker(
        new_membrane.center_of_mass[index],
        new_membrane.angle[index],
        new_membrane.protein_type[index],
        index,
        out_sharding=out_sharding,
    )

    def f(key: Key[Array, ""]) -> tuple[Membrane, Float[Array, ""] | None]:
        e_tot_init = one_particle_energy(
            membrane, index, force_field, out_sharding=out_sharding
        )
        e_tot_final = one_particle_energy(
            new_membrane, index, force_field, out_sharding=out_sharding
        )
        dE = e_tot_final - e_tot_init
        rand = jax.random.uniform(key, ())  # pyright: ignore [reportUnknownMemberType]
        del key
        new_energy = initial_energy + dE if initial_energy is not None else None
        v = typing.cast(
            tuple[Membrane, Float[Array, ""] | None],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                (dE < 0) | (rand < jnp.exp(-dE / kBT)),
                lambda: (new_membrane, new_energy),
                lambda: (membrane, initial_energy),
            ),
        )
        return v

    res = typing.cast(
        tuple[Membrane, Float[Array, ""] | None],
        jax.lax.cond(is_valid, lambda: f(k3), lambda: (membrane, initial_energy)),  # pyright: ignore [reportUnknownMemberType] # noqa: F821
    )
    del k3

    return res


def sample_swap(
    membrane: Membrane,
    swappable: Bool[Array, "s s"],
    force_field: ForceField,
    checker: Checker,
    kBT: Float[Array, ""],
    key: Key[Array, ""],
    initial_energy: Float[Array, ""] | None = None,
    out_sharding: jax.sharding.PartitionSpec | None = None,
) -> tuple[Membrane, Float[Array, ""] | None]:
    """
    Sample a swap for the Monte Carlo simulation.

    Parameters:
        membrane: The `Membrane` that you want to simulate.
        swappable: `swappable[t1, t2]` is `True` if proteins of type `t1` and `t2` can be swapped, and `False` otherwise. Must be symmetric.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        kBT: The energy scale used in the Metropolis-Hastings acceptance criteria.
        key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)
        initial_energy: The energy of the input `Membrane` or `None`.

    Returns:
        The `Membrane` after the swap.
        The energy of the returned `Membrane` if `initial_energy is not None` otherwise `None`.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    del key

    index1 = jax.random.randint(  # pyright: ignore [reportUnknownMemberType]
        k1, (), minval=0, maxval=membrane.center_of_mass.shape[0]
    )
    del k1
    pt1 = membrane.protein_type[index1]
    r1 = membrane.radius[index1]

    swap_eligible = jnp.where(
        swappable[pt1, membrane.protein_type],
        size=membrane.protein_type.shape[0],
    )[0]
    index2 = jax.random.randint(  # pyright: ignore [reportUnknownMemberType]
        k2, (), minval=0, maxval=jnp.sum(swappable[pt1, membrane.protein_type])
    )
    del k2
    pt2 = membrane.protein_type[index2]
    r2 = membrane.radius[index2]

    new_protein_type = membrane.protein_type.at[index1].set(pt2).at[index2].set(pt1)
    new_radius = membrane.radius.at[index1].set(r2).at[index2].set(r1)
    new_membrane = Membrane(
        membrane.center_of_mass,
        membrane.angle,
        new_radius,
        new_protein_type,
    )
    is_valid = (
        jnp.any(swap_eligible)
        & checker(
            new_membrane.center_of_mass[index1],
            new_membrane.angle[index1],
            new_membrane.protein_type[index1],
            index1,
            out_sharding=out_sharding,
        )
        & checker(
            new_membrane.center_of_mass[index2],
            new_membrane.angle[index2],
            new_membrane.protein_type[index2],
            index2,
            out_sharding=out_sharding,
        )
    )

    def energy(membrane: Membrane):
        e_tot = one_particle_energy(
            membrane,
            index1,
            force_field,
            out_sharding=out_sharding,
        ) + one_particle_energy(
            membrane,
            index2,
            force_field,
            out_sharding=out_sharding,
        )
        com1 = membrane.center_of_mass[index1]
        angle1 = membrane.angle[index1]
        radius1 = membrane.radius[index1]
        type1 = membrane.protein_type[index1]
        com2 = membrane.center_of_mass[index2]
        angle2 = membrane.angle[index2]
        radius2 = membrane.radius[index2]
        type2 = membrane.protein_type[index2]
        dist = typing.cast(Float[Array, ""], jnp.linalg.norm(com1 - com2))
        return typing.cast(
            Float[Array, ""],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                dist < radius1 + radius2,
                lambda: e_tot - force_field(com1, angle1, type1, com2, angle2, type2),
                lambda: e_tot,
            ),
        )

    def f(key: Key[Array, ""]) -> tuple[Membrane, Float[Array, ""] | None]:
        e_tot_init = energy(membrane)
        e_tot_final = energy(new_membrane)
        dE = e_tot_final - e_tot_init
        rand = jax.random.uniform(key, ())  # pyright: ignore [reportUnknownMemberType]
        del key
        new_energy = initial_energy + dE if initial_energy is not None else None
        v = typing.cast(
            tuple[Membrane, Float[Array, ""] | None],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                (dE < 0) | (rand < jnp.exp(-dE / kBT)),
                lambda: (new_membrane, new_energy),
                lambda: (membrane, initial_energy),
            ),
        )
        return v

    res = typing.cast(
        tuple[Membrane, Float[Array, ""] | None],
        jax.lax.cond(is_valid, lambda: f(k3), lambda: (membrane, initial_energy)),  # pyright: ignore [reportUnknownMemberType] # noqa: F821
    )
    del k3

    return res


def sample_translation(
    membrane: Membrane,
    translatable: Int[Array, " m"],
    force_field: ForceField,
    checker: Checker,
    r_max: Float[Array, ""],
    kBT: Float[Array, ""],
    key: Key[Array, ""],
    initial_energy: Float[Array, ""] | None = None,
    out_sharding: jax.sharding.PartitionSpec | None = None,
) -> tuple[Membrane, Float[Array, ""] | None]:
    """
    Sample a translation for the Monte Carlo simulation.

    Parameters:
        membrane: The `Membrane` that you want to simulate.
        translatable: The indices of the proteins in `membrane` that can be translated.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        r_max: The maximum change in each coordinate of `membrane.center_of_mass[i]` when translating protein `i`.
        kBT: The energy scale used in the Metropolis-Hastings acceptance criteria.
        key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)
        initial_energy: The energy of the input `Membrane` or `None`.

    Returns:
        The `Membrane` after the translation.
        The energy of the returned `Membrane` if `initial_energy is not None` otherwise `None`.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    del key

    index = jax.random.choice(k1, translatable)
    del k1

    translation = jax.random.uniform(k2, (2,), minval=-r_max, maxval=r_max)  # pyright: ignore [reportUnknownMemberType]
    del k2

    new_com = membrane.center_of_mass.at[index].add(translation)
    new_membrane = Membrane(
        new_com,
        membrane.angle,
        membrane.radius,
        membrane.protein_type,
    )
    is_valid = checker(
        new_membrane.center_of_mass[index],
        new_membrane.angle[index],
        new_membrane.protein_type[index],
        index,
        out_sharding=out_sharding,
    )

    def f(key: Key[Array, ""]) -> tuple[Membrane, Float[Array, ""] | None]:
        e_tot_init = one_particle_energy(
            membrane, index, force_field, out_sharding=out_sharding
        )
        e_tot_final = one_particle_energy(
            new_membrane, index, force_field, out_sharding=out_sharding
        )
        dE = e_tot_final - e_tot_init
        rand = jax.random.uniform(key, ())  # pyright: ignore [reportUnknownMemberType]
        del key
        new_energy = initial_energy + dE if initial_energy is not None else None
        v = typing.cast(
            tuple[Membrane, Float[Array, ""] | None],
            jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                (dE < 0) | (rand < jnp.exp(-dE / kBT)),
                lambda: (new_membrane, new_energy),
                lambda: (membrane, initial_energy),
            ),
        )
        return v

    res = typing.cast(
        tuple[Membrane, Float[Array, ""] | None],
        jax.lax.cond(is_valid, lambda: f(k3), lambda: (membrane, initial_energy)),  # pyright: ignore [reportUnknownMemberType] # noqa: F821
    )
    del k3

    return res


def simulated_annealing(
    membrane: Membrane,
    translatable: Int[Array, " t"],
    swappable: Bool[Array, "s s"],
    rotatable: Int[Array, " r"],
    force_field: ForceField,
    checker: Checker,
    r_max_values: Float[Array, " v"],
    phi_max_values: Float[Array, " v"],
    kBT_values: Float[Array, " v"],
    n_steps_per_temp: int,
    key: Key[Array, ""],
    initial_energy: Float[Array, ""] | None = None,
) -> tuple[Membrane, Float[Array, ""] | None]:
    """
    Run `n_steps_per_temp` of Monte Carlo simulation at each `(r_max, phi_max, kBT)`.

    Parameters:
        membrane: The `Membrane` that you want to simulate.
        translatable: The indices of the proteins in `membrane` that can be translated.
        swappable: `swappable[t1, t2]` is `True` if proteins of type `t1` and `t2` can be swapped, and `False` otherwise. Must be symmetric.
        rotatable: The indices of the proteins in `membrane` that can be rotated.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        r_max_values: The array of maximum changes in each coordinate of `membrane.center_of_mass[i]` when translating protein `i`.
        phi_max_values: The array of maximum changes in `membrane.angle[i]` when rotating protein `i`.
        kBT_values: The array of energy scales used in the Metropolis-Hastings acceptance criteria.
        n_steps_per_temp: The number of steps of Monte Carlo simulation to perform at each `(r_max, phi_max, kBT)`.
        key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)
        initial_energy: The energy of the input `Membrane` or `None`.

    Returns:
        The `Membrane` after simulated annealing.
        The energy of the returned `Membrane` if `initial_energy is not None` otherwise `None`.
    """
    keys = jax.random.split(key, (r_max_values.shape[0],))

    def body(
        i: Int[Array, ""], args: tuple[Membrane, Float[Array, ""] | None]
    ) -> tuple[Membrane, Float[Array, ""] | None]:
        membrane, initial_energy = args
        return run_monte_carlo(
            membrane,
            translatable,
            swappable,
            rotatable,
            force_field,
            checker,
            r_max_values[i],
            phi_max_values[i],
            kBT_values[i],
            n_steps_per_temp,
            keys[i],
            initial_energy=initial_energy,
        )

    return typing.cast(
        tuple[Membrane, Float[Array, ""] | None],
        jax.lax.fori_loop(0, r_max_values.shape[0], body, (membrane, initial_energy)),  # pyright: ignore [reportUnknownMemberType]
    )


class MultiMembrane(eqx.Module):
    """
    Represents a collection of membranes and all of the proteins they contain. Used for parallel tempering.

    Attributes:
        center_of_mass: The center of mass of every protein in every membrane.
        angle: The angle of every protein in every membrane.
        protein_type: The protein type of every protein in every membrane. This is used for force field evaluation and for swapping.
        radius: The maximum norm of the 2d convex hull of every protein in every membrane.
    """

    center_of_mass: Float[Array, "m p 2"]
    angle: Float[Array, "m p"]
    protein_type: Int[Array, "m p"]
    radius: Float[Array, "m p"]

    def __init__(
        self,
        center_of_mass: Float[Array, "m p 2"],
        angle: Float[Array, "m p"],
        radius: Float[Array, "m p"],
        protein_type: Int[Array, "m p"],
    ):
        """
        Initialize a `MultiMembrane`.

        Attributes:
            center_of_mass: The center of mass of every protein in every membrane.
            angle: The angle of every protein in every membrane.
            protein_type: The protein type of every protein in every membrane. This is used for force field evaluation and for swapping.
            radius: The maximum norm of the 2d convex hull of every protein in every membrane.
        """
        self.center_of_mass = center_of_mass
        self.angle = angle
        self.radius = radius
        self.protein_type = protein_type


def total_energies(
    multi_membrane: MultiMembrane,
    force_field: ForceField,
) -> Float[Array, " m"]:
    """
    Calculate the total energy of all proteins in every membrane of `multi_membrane`.

    Parameters:
        multi_membrane: The `MultiMembrane` containing the membranes.
        force_field: The `ForceField` used for energy calculations.

    Returns:
        The total energy of every membrane in `multi_membrane`.
    """

    def f(membrane: Membrane) -> Float[Array, ""]:
        return total_energy(membrane, force_field)

    return typing.cast(Float[Array, " m"], jax.lax.map(f, multi_membrane))  # pyright: ignore [reportUnknownMemberType]


Swapper = Callable[
    [
        MultiMembrane,
        Float[Array, " m"],
        Float[Array, " m"],
        Key[Array, ""],
    ],
    tuple[MultiMembrane, Float[Array, " m"]],
]
"""
Type for functions that can be called to swap membranes while parallel tempering.

Parameters:
    multi_membrane (MultiMembrane): The `MultiMembrane` in which membranes may be swapped.
    energy (Float[Array, " m"]): The energies of the membranes.
    kBT (Float[Array, " m"]): The energy scales of the membranes.
    key (Key[Array, ""]): A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

Returns:
    (MultiMembrane): The resulting `MultiMembrane` after swapping.
    (Float[Array, " m"]): The resulting energies after swapping.
"""


class SwapAdjacentRandomly(eqx.Module):
    """
    Repeats the following `num_swaps` times. Chooses a pair of adjacent membranes at random, and
    attempts to swap them with probability from Metropolis-Hastings.

    Attributes:
        num_swaps: The number of times to attempt swaps.
    """

    num_swaps: int

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"]]:
        """
        Swap adjacent membranes at random.

        Parameters:
            multi_membrane: The `MultiMembrane` in which membranes may be swapped.
            energy: The energies of the membranes.
            kBT: The energy scales of the membranes.
            key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

        Returns:
            (MultiMembrane): The resulting `MultiMembrane` after swapping.
            (Float[Array, " m"]): The resulting energies after swapping.
        """
        keys = jax.random.split(key, self.num_swaps)
        del key

        def f(
            i: Int[Array, ""], args: tuple[MultiMembrane, Float[Array, " m"]]
        ) -> tuple[MultiMembrane, Float[Array, " m"]]:
            membrane, energy = args
            k1, k2 = jax.random.split(keys[i], 2)

            j = jax.random.randint(k1, (), minval=0, maxval=energy.shape[0] - 1)  # pyright: ignore [reportUnknownMemberType]
            del k1
            rand = jax.random.uniform(k2, ())  # pyright: ignore [reportUnknownMemberType]
            del k2

            def swap(x: Float[Array, "m ..."]):
                x1 = x[j]
                x2 = x[j + 1]
                return x.at[j].set(x2).at[j + 1].set(x1)

            p_accept = (1 / kBT[j] - 1 / kBT[j + 1]) * (energy[j] - energy[j + 1])
            return typing.cast(
                tuple[MultiMembrane, Float[Array, " m"]],
                jax.lax.cond(  # pyright: ignore [reportUnknownMemberType]
                    (p_accept > 1) | (rand < p_accept),
                    lambda: (jax.tree.map(swap, membrane), swap(energy)),
                    lambda: (membrane, energy),
                ),
            )

        return typing.cast(
            tuple[MultiMembrane, Float[Array, " m"]],
            jax.lax.fori_loop(0, self.num_swaps, f, (multi_membrane, energy)),  # pyright: ignore [reportUnknownMemberType]
        )


class SwapEvenOdd(eqx.Module):
    """
    Swap membranes (0, 1), (2, 3), ... then (1, 2), (3, 4), ...
    """

    def __call__(
        self,
        multi_membrane: MultiMembrane,
        energy: Float[Array, " m"],
        kBT: Float[Array, " m"],
        key: Key[Array, ""],
    ) -> tuple[MultiMembrane, Float[Array, " m"]]:
        """
        Swap membranes (0, 1), (2, 3), ... then (1, 2), (3, 4), ...

        Parameters:
            multi_membrane: The `MultiMembrane` in which membranes may be swapped.
            energy: The energies of the membranes.
            kBT: The energy scales of the membranes.
            key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

        Returns:
            (MultiMembrane): The resulting `MultiMembrane` after swapping.
            (Float[Array, " m"]): The resulting energies after swapping.
        """

        def swap_with_offset(
            offset: int,
            multi_membrane: MultiMembrane,
            energy: Float[Array, " m"],
            kBT: Float[Array, " m"],
            key: Key[Array, ""],
        ):
            index = jnp.arange(offset, energy.shape[0], 2)  # pyright: ignore [reportUnknownMemberType]
            p_accept = (1 / kBT[index] - 1 / kBT[index + 1]) * (
                energy[index] - energy[index + 1]
            )
            rand = jax.random.uniform(key, (index.shape[0],))  # pyright: ignore [reportUnknownMemberType]
            should_swap = (p_accept > 1) | (rand < p_accept)
            new_index = (
                jnp.arange(0, energy.shape[0])  # pyright: ignore [reportUnknownMemberType]
                .at[index]
                .set(index + should_swap)
                .at[index + 1]
                .set(index + 1 - should_swap)
            )

            def swap(x: Float[Array, "m ..."]):
                return x[new_index]

            return jax.tree.map(swap, multi_membrane), swap(energy)

        k1, k2 = jax.random.split(key, 2)
        del key
        multi_membrane, energy = swap_with_offset(0, multi_membrane, energy, kBT, k1)
        del k1
        multi_membrane, energy = swap_with_offset(1, multi_membrane, energy, kBT, k2)
        del k2
        return multi_membrane, energy


def parallel_tempering(
    multi_membrane: MultiMembrane,
    translatable: Int[Array, " t"],
    swappable: Bool[Array, "s s"],
    rotatable: Int[Array, " r"],
    force_field: ForceField,
    checker: Checker,
    swapper: Swapper,
    r_max_values: Float[Array, " m"],
    phi_max_values: Float[Array, " m"],
    kBT_values: Float[Array, " m"],
    initial_energy: Float[Array, " m"],
    n_steps_between_swaps: int,
    key: Key[Array, ""],
) -> tuple[MultiMembrane, Float[Array, " m"]]:
    """
    Run `n_steps_between_swaps` of Monte Carlo simulation for each membrane in `multi_membrane`,
    then use `swapper` to swap membranes.

    Parameters:
        multi_membrane: The `MultiMembrane` that you want to simulate.
        translatable: The indices of the proteins in `membrane` that can be translated.
        swappable: `swappable[t1, t2]` is `True` if proteins of type `t1` and `t2` can be swapped, and `False` otherwise. Must be symmetric.
        rotatable: The indices of the proteins in `membrane` that can be rotated.
        force_field: The `ForceField` used for energy calculations.
        checker: Used to check whether particular moves are permitted.
        swapper: Used to swap membranes for parallel tempering.
        r_max_values: The array of maximum changes in each coordinate when running Monte Carlo
            simulation on `multi_membrane[i]`.
        phi_max_values: The array of maximum changes in angle when running Monte Carlo
            simulation on `multi_membrane[i]`.
        kBT_values: The array of energy scales used in the Metropolis-Hastings acceptance
            criteria when running Monte Carlo simulation on `multi_membrane[i]`.
        initial_energy: The energy of the input membranes.
        n_steps_between_swaps: The number of steps of Monte Carlo simulation to perform
            for membrane.
        key: A JAX random key used to generate random numbers during the simulation. [Learn more about JAX pseudorandom number generation.](https://docs.jax.dev/en/latest/random-numbers.html)

    Returns:
        The `MultiMembrane` after parallel tempering.
        The energy of the membranes after parallel tempering.
    """
    out_sharding = jax.typeof(initial_energy).sharding.spec

    def f(
        multi_membrane: MultiMembrane,
        r_max_values: Float[Array, " a"],
        phi_max_values: Float[Array, " a"],
        kBT_values: Float[Array, " a"],
        initial_energy: Float[Array, " a"],
        key: Key[Array, " a"],
    ) -> tuple[MultiMembrane, Float[Array, ""]]:
        def body(
            i: Int[Array, ""], state: tuple[MultiMembrane, Float[Array, " a"]]
        ) -> tuple[MultiMembrane, Float[Array, " a"]]:
            multi_membrane, initial_energy = state
            membrane = Membrane(
                multi_membrane.center_of_mass[i],
                multi_membrane.angle[i],
                multi_membrane.radius[i],
                multi_membrane.protein_type[i],
            )
            membrane, membrane_energy = run_monte_carlo(
                membrane,
                translatable,
                swappable,
                rotatable,
                force_field,
                checker,
                r_max_values[i],
                phi_max_values[i],
                kBT_values[i],
                n_steps_between_swaps,
                key[i],
                initial_energy=initial_energy[i],
                out_sharding=out_sharding,
            )
            new_multi_membrane = MultiMembrane(
                state[0].center_of_mass.at[i].set(membrane.center_of_mass),
                state[0].angle.at[i].set(membrane.angle),
                state[0].radius.at[i].set(membrane.radius),
                state[0].protein_type.at[i].set(membrane.protein_type),
            )
            new_energy = state[1].at[i].set(membrane_energy)
            return new_multi_membrane, new_energy

        return typing.cast(
            tuple[MultiMembrane, Float[Array, " a"]],
            jax.lax.fori_loop(  # pyright: ignore [reportUnknownMemberType]
                0, initial_energy.shape[0], body, (multi_membrane, initial_energy)
            ),
        )

    def get_sharding(x: Array) -> jax.NamedSharding:
        return jax.typeof(x).sharding

    def get_partition_spec(x: Array) -> jax.sharding.PartitionSpec:
        return get_sharding(x).spec

    k1, key = jax.random.split(key, 2)
    keys = jax.random.split(k1, initial_energy.shape[0])
    del k1
    args = (
        multi_membrane,
        r_max_values,
        phi_max_values,
        kBT_values,
        initial_energy,
        keys,
    )
    del keys
    multi_membrane, initial_energy = typing.cast(
        tuple[MultiMembrane, Float[Array, ""]],
        jax.shard_map(  # pyright: ignore [reportCallIssue, reportUnknownMemberType]
            f,
            out_specs=jax.tree.map(
                get_partition_spec, (multi_membrane, initial_energy)
            ),
        )(*args),
    )

    new_multi_membrane, new_initial_energy = swapper(
        jax.sharding.reshard(multi_membrane, jax.sharding.PartitionSpec()),  # pyright: ignore [reportUnknownMemberType]
        jax.sharding.reshard(initial_energy, jax.sharding.PartitionSpec()),  # pyright: ignore [reportUnknownMemberType]
        jax.sharding.reshard(kBT_values, jax.sharding.PartitionSpec()),  # pyright: ignore [reportUnknownMemberType]
        key,
    )
    del key

    multi_membrane = jax.sharding.reshard(  # pyright: ignore [reportUnknownMemberType]
        new_multi_membrane, jax.tree.map(get_sharding, multi_membrane)
    )
    initial_energy = jax.sharding.reshard(  # pyright: ignore [reportUnknownMemberType]
        new_initial_energy, get_sharding(initial_energy)
    )
    return multi_membrane, initial_energy
