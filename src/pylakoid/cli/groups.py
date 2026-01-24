import pathlib
import time
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import typer

from pylakoid.structure import make_force_field as mff
from pylakoid.structure.pdb import parse_pdb
from pylakoid.structure.vdw_radii import vdw_radii

app = typer.Typer()


@app.command()
def groups(
    pdb_path: Annotated[
        pathlib.Path,
        typer.Argument(help="Path to the PDB", exists=True, dir_okay=False),
    ],
    groups_min: Annotated[
        int, typer.Option(help="The smallest number of groups to evaluate (inclusive)")
    ],
    groups_max: Annotated[
        int, typer.Option(help="The largest number of groups to evaluate (exclusive)")
    ],
    groups_step: Annotated[
        int, typer.Option(help="The step size for number of groups to evaluate")
    ],
    seed: Annotated[int, typer.Option(help="The JAX random seed")] = 0,
):
    key = jax.random.key(seed)
    preprocess_fn = eqx.filter_jit(mff.preprocess_for_count_sphere_intersections)
    count_fn = eqx.filter_jit(mff.count_sphere_intersections_fast)

    pdb = parse_pdb("p1", str(pdb_path))
    x = jnp.array([atom.coord for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    r = jnp.array([vdw_radii[atom.element] for atom in pdb.get_atoms()])  # pyright: ignore [reportUnknownMemberType]
    durations: list[tuple[int, float]] = []
    for num_groups in range(groups_min, groups_max, groups_step):
        k1, key = jax.random.split(key, 2)
        preprocess = preprocess_fn(x, r, num_groups, k1)
        del k1
        start = time.time()
        jax.block_until_ready(  # pyright: ignore [reportUnknownMemberType]
            count_fn(
                preprocess,
                preprocess,
                jnp.zeros(3),  # pyright: ignore [reportUnknownMemberType]
                jnp.zeros(()),  # pyright: ignore [reportUnknownMemberType]
                jnp.zeros(3),  # pyright: ignore [reportUnknownMemberType]
                jnp.zeros(()),  # pyright: ignore [reportUnknownMemberType]
            )
        )
        duration = time.time() - start
        durations.append((num_groups, duration))
    best = min(durations, key=lambda x: x[1])[0]
    print(best)
