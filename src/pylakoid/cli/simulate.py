import pathlib
import time
import typing
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key
import numpy as np
import pydantic
import safetensors.numpy
import typer

from pylakoid.structure import anneal
from pylakoid.structure import anneal_helpers as ah
from pylakoid.structure.checker import Checker
from pylakoid.structure import checker_helpers as ch
from pylakoid.structure import force_field as ff
from pylakoid.structure import force_field_helpers as ffh
from pylakoid.structure.pdb import parse_pdb

app = typer.Typer()


class Boundary(pydantic.BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float


class PdbParameters(pydantic.BaseModel):
    path: str
    num_groups: int
    protein_count: int


class Range(pydantic.BaseModel):
    start: float
    end: float


class ParallelTemperingParameters(pydantic.BaseModel):
    num_epochs: int
    num_swaps_per_epoch: int
    num_steps_between_swaps_per_protein: int
    num_membranes: int
    r_max: Range
    phi_max: Range
    kBT: Range


class SimulateConfig(pydantic.BaseModel):
    boundary: Boundary
    pdb: dict[str, PdbParameters]
    force_field_dir: str
    parameters: ParallelTemperingParameters


def make_boundary(
    boundary_xmin: float,
    boundary_xmax: float,
    boundary_ymin: float,
    boundary_ymax: float,
) -> Float[Array, "b 2"]:
    return jnp.array(  # pyright: ignore [reportUnknownMemberType]
        [
            [boundary_xmin, boundary_ymin],
            [boundary_xmin, boundary_ymax],
            [boundary_xmax, boundary_ymax],
            [boundary_xmax, boundary_ymin],
        ]
    )


def make_proteins(
    pdb: dict[str, PdbParameters],
) -> tuple[
    Float[Array, " p"],
    Int[Array, " p"],
    dict[str, int],
]:
    protein_type_map = {name: i for i, name in enumerate(set(pdb.keys()))}
    protein_type_list: list[int] = []
    for name, v in pdb.items():
        protein_type_list += [protein_type_map[name]] * v.protein_count
    protein_type = jnp.array(protein_type_list)  # pyright: ignore [reportUnknownMemberType]
    radius = ah.prepare_radius(
        {name: parse_pdb(name, v.path) for name, v in pdb.items()},
        protein_type_map,
        protein_type,
    )
    return radius, protein_type, protein_type_map


def make_protein_positions(
    boundary_xmin: float,
    boundary_xmax: float,
    boundary_ymin: float,
    boundary_ymax: float,
    radius: Float[Array, " p"],
    key: Key[Array, ""],
) -> tuple[Float[Array, "p 2"], Float[Array, " p"]]:
    com_xmin, com_xmax = boundary_xmin + radius, boundary_xmax - radius
    com_ymin, com_ymax = boundary_ymin + radius, boundary_ymax - radius
    k1, k2, key = jax.random.split(key, 3)
    center_of_mass = jax.random.uniform(  # pyright: ignore [reportUnknownMemberType]
        k1,
        (radius.shape[0], 2),
        minval=jnp.stack([com_xmin, com_ymin], axis=1),
        maxval=jnp.stack([com_xmax, com_ymax], axis=1),
    )
    del k1
    angle = jax.random.uniform(  # pyright: ignore [reportUnknownMemberType]
        k2,
        (radius.shape[0],),
        minval=0.0,
        maxval=2 * jnp.pi,
    )
    del k2
    return center_of_mass, angle


def make_simulation_permissions(
    n_prot: int, protein_type_map: dict[str, int]
) -> tuple[Int[Array, " p"], Int[Array, " p"], Int[Array, "s s"]]:
    translatable = jnp.array(list(range(n_prot)), dtype=jnp.int32)  # pyright: ignore [reportUnknownMemberType]
    rotatable = jnp.array(list(range(n_prot)), dtype=jnp.int32)  # pyright: ignore [reportUnknownMemberType]
    swappable = ah.prepare_swappable([], protein_type_map)
    return translatable, rotatable, swappable


def parallel_tempering_loop(
    membrane: anneal.MultiMembrane,
    tempering_params: Float[Array, "a 3"],
    translatable: Int[Array, " p"],
    rotatable: Int[Array, " p"],
    swappable: Int[Array, " s"],
    force_field: ff.ForceField,
    checker: Checker,
    swapper: anneal.Swapper,
    initial_energy: Float[Array, " m"],
    n_steps_between_swaps_per_protein: int,
    n_swaps: int,
    key: Key[Array, ""],
) -> tuple[anneal.MultiMembrane, Float[Array, " m"], anneal.SwapStats]:
    keys = jax.random.split(key, n_swaps)
    del key

    def f(
        i: Int[Array, ""],
        state: tuple[anneal.MultiMembrane, Float[Array, " m"], anneal.SwapStats],
    ) -> tuple[anneal.MultiMembrane, Float[Array, " m"], anneal.SwapStats]:
        multi_membrane, energy, swap_stats = anneal.parallel_tempering(
            state[0],
            translatable,
            swappable,
            rotatable,
            force_field,
            checker,
            swapper,
            tempering_params[:, 0],
            tempering_params[:, 1],
            tempering_params[:, 2],
            state[1],
            n_steps_between_swaps_per_protein * translatable.shape[0],
            keys[i],
        )
        return (
            multi_membrane,
            energy,
            anneal.SwapStats(
                state[2].n_accepted + swap_stats.n_accepted,
                state[2].n_attempted + swap_stats.n_attempted,
            ),
        )

    return typing.cast(
        tuple[anneal.MultiMembrane, Float[Array, " m"], anneal.SwapStats],
        jax.lax.fori_loop(  # pyright: ignore [reportUnknownMemberType]
            0,
            n_swaps,
            f,
            (
                membrane,
                initial_energy,
                anneal.make_swap_stats(initial_energy.shape[0] - 1),
            ),
        ),
    )


def save_multi_membrane(
    multi_membrane: anneal.MultiMembrane,
    out_path: pathlib.Path,
):
    tensors = {
        "center_of_mass": np.asarray(multi_membrane.center_of_mass),
        "angle": np.asarray(multi_membrane.angle),
        "radius": np.asarray(multi_membrane.radius),
        "protein_type": np.asarray(multi_membrane.protein_type),
    }
    safetensors.numpy.save_file(tensors, out_path)  # pyright: ignore [reportUnknownMemberType]


@app.command()
def simulate(
    config: Annotated[
        pathlib.Path,
        typer.Option(
            help="Path to the configuration file", exists=True, dir_okay=False
        ),
    ],
    out_dir: Annotated[
        pathlib.Path,
        typer.Option(help="Path to the output directory", exists=True, file_okay=False),
    ],
    seed: Annotated[int, typer.Option(help="The JAX random seed")] = 0,
    num_cpu: Annotated[
        int, typer.Option(help="The number of CPUs to use for parallelization")
    ] = 1,
):
    jax.config.update("jax_num_cpu_devices", num_cpu)  # pyright: ignore [reportUnknownMemberType]

    with open(config, "r") as f:
        valid_config = SimulateConfig.model_validate_json(f.read())

    tempering_params = jnp.geomspace(
        jnp.array(  # pyright: ignore [reportUnknownMemberType]
            [
                valid_config.parameters.r_max.start,
                valid_config.parameters.phi_max.start,
                valid_config.parameters.kBT.start,
            ]
        ),
        jnp.array(  # pyright: ignore [reportUnknownMemberType]
            [
                valid_config.parameters.r_max.end,
                valid_config.parameters.phi_max.end,
                valid_config.parameters.kBT.end,
            ]
        ),
        valid_config.parameters.num_membranes,
    )

    force_field_path = {
        (p1, p2): str(
            pathlib.Path(valid_config.force_field_dir) / f"{p1}_{p2}.safetensors"
        )
        for p1 in valid_config.pdb.keys()
        for p2 in valid_config.pdb.keys()
        if (
            pathlib.Path(valid_config.force_field_dir) / f"{p1}_{p2}.safetensors"
        ).exists()
    }

    key = jax.random.key(seed)
    boundary = make_boundary(
        valid_config.boundary.xmin,
        valid_config.boundary.xmax,
        valid_config.boundary.ymin,
        valid_config.boundary.ymax,
    )
    radius, protein_type, protein_type_map = make_proteins(valid_config.pdb)
    k1, key = jax.random.split(key, 2)
    com, angle = jax.vmap(make_protein_positions, (None, None, None, None, None, 0))(
        valid_config.boundary.xmin,
        valid_config.boundary.xmax,
        valid_config.boundary.ymin,
        valid_config.boundary.ymax,
        radius,
        jax.random.split(k1, len(tempering_params)),
    )
    del k1
    translatable, rotatable, swappable = make_simulation_permissions(
        radius.shape[0], protein_type_map
    )
    force_field = ffh.prepare_force_field(
        force_field_path,
        {name: v.path for name, v in valid_config.pdb.items()},
        protein_type_map,
        {},
    )
    checker = ch.prepare_atomic_hull_inside_polygon_checker(
        boundary,
        {name: parse_pdb(name, v.path) for name, v in valid_config.pdb.items()},
        protein_type_map,
    )
    swapper = anneal.SwapAdjacentRandomly(len(tempering_params))
    k1, key = jax.random.split(key, 2)
    force_field_exact = ffh.prepare_exact_force_field(
        {name: v.path for name, v in valid_config.pdb.items()},
        {name: v.num_groups for name, v in valid_config.pdb.items()},
        protein_type_map,
        k1,
    )
    del k1

    multi_membrane = anneal.MultiMembrane(
        com,
        angle,
        jnp.repeat(jnp.expand_dims(radius, 0), com.shape[0], axis=0),
        jnp.repeat(jnp.expand_dims(protein_type, 0), com.shape[0], axis=0),
    )

    energy = anneal.total_energies(multi_membrane, force_field)

    def make_partition_spec(x: Array):
        head = ["x"]
        tail = [None] * (x.ndim - 1)
        return jax.sharding.PartitionSpec(*(head + tail))

    n_dev = jax.device_count()  # pyright: ignore [reportUnknownMemberType]
    mesh = jax.make_mesh((n_dev,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))  # pyright: ignore [reportUnknownMemberType]
    with jax.set_mesh(mesh):
        multi_membrane = jax.device_put(  # pyright: ignore [reportUnknownMemberType]
            multi_membrane, jax.tree.map(make_partition_spec, multi_membrane)
        )
        tempering_params = jax.device_put(  # pyright: ignore [reportUnknownMemberType]
            tempering_params, make_partition_spec(tempering_params)
        )
        energy = jax.device_put(energy, make_partition_spec(energy))  # pyright: ignore [reportUnknownMemberType]

        save_multi_membrane(
            multi_membrane,
            pathlib.Path(out_dir) / "multimembrane.epoch0.safetensors",
        )

        started_tempering = time.time()
        parallel_tempering_fn = eqx.filter_jit(parallel_tempering_loop)
        for i in range(valid_config.parameters.num_epochs):
            k1, key = jax.random.split(key, 2)
            print(f"(Iteration {i + 1}) Starting tempering...")
            start = time.time()
            multi_membrane, energy, stats = parallel_tempering_fn(
                multi_membrane,
                tempering_params,
                translatable,
                rotatable,
                swappable,
                force_field,
                checker,
                swapper,
                energy,
                valid_config.parameters.num_steps_between_swaps_per_protein,
                valid_config.parameters.num_swaps_per_epoch,
                k1,
            )
            del k1
            duration = time.time() - start
            print(
                f"(Iteration {i + 1}) Done tempering... energy is {energy} (took {duration} seconds)"
            )
            acceptance_rate = stats.n_accepted / stats.n_attempted
            acceptance_rate = jnp.where(jnp.isnan(acceptance_rate), 0, acceptance_rate)
            print(f"(Iteration {i + 1}) Swap acceptance rates {acceptance_rate}")
            duration = time.time() - started_tempering
            print(
                f"(Iteration {i + 1}) Total time spent tempering so far is {duration} seconds"
            )

            save_multi_membrane(
                multi_membrane,
                pathlib.Path(out_dir) / f"multimembrane.epoch{i + 1}.safetensors",
            )

    multi_membrane = jax.sharding.reshard(  # pyright: ignore [reportUnknownMemberType]
        multi_membrane, jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )

    membrane = anneal.Membrane(
        multi_membrane.center_of_mass[-1],
        multi_membrane.angle[-1],
        multi_membrane.radius[-1],
        multi_membrane.protein_type[-1],
    )

    energy = anneal.total_energy(membrane, force_field_exact)
    print(f"Exact energy after annealing: {energy}")
