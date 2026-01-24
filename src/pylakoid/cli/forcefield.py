import pathlib
import typing
from typing import Annotated

import click
import jax
import jax.numpy as jnp
import pydantic
import typer

from pylakoid.structure import anneal_helpers as ah
from pylakoid.structure import make_force_field_helpers as mffh
from pylakoid.structure.pdb import parse_pdb

app = typer.Typer()


class PathAndNumGroups(pydantic.BaseModel):
    path: str
    num_groups: int


class ForcefieldConfig(pydantic.BaseModel):
    pdb: dict[str, PathAndNumGroups]


@app.command()
def forcefield(
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
    r_steps: Annotated[
        int,
        typer.Option(
            help="Number of r values between 0 and the sum of the protein radii"
        ),
    ],
    phi_steps: Annotated[
        int, typer.Option(help="Number of phi values between 0 and 2*pi")
    ],
    name: Annotated[
        list[str] | None,
        typer.Option(
            help="Generate force fields for this name, can be specified more than once. Defaults to all names in the configuration file"
        ),
    ] = None,
    pair: Annotated[
        list[click.Tuple] | None,
        typer.Option(
            help="Generate force field for this pair, can be specified more than once. Defaults to all pairs in the configuration file",
            click_type=click.Tuple([str, str]),
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="The JAX random seed")] = 0,
    num_cpu: Annotated[
        int, typer.Option(help="The number of CPUs to use for parallelization")
    ] = 1,
):
    jax.config.update("jax_num_cpu_devices", num_cpu)  # pyright: ignore [reportUnknownMemberType]

    with open(config, "r") as f:
        valid_config = ForcefieldConfig.model_validate_json(f.read())

    name_set = set(name if name and len(name) > 0 else (valid_config.pdb.keys()))
    pair_set = set(
        [(min(n1, n2), max(n1, n2)) for (n1, n2) in typing.cast(tuple[str, str], pair)]
        if pair and len(pair) > 0
        else [
            (n1, n2)
            for n1 in valid_config.pdb.keys()
            for n2 in valid_config.pdb.keys()
            if n1 <= n2
        ]
    )

    key = jax.random.key(seed)
    for name1, v1 in valid_config.pdb.items():
        for name2, v2 in valid_config.pdb.items():
            if (
                name1 <= name2
                and name1 in name_set
                and name2 in name_set
                and (name1, name2) in pair_set
            ):
                print(f"Processing ({name1}, {name2})...")
                r_max = ah.get_radius(parse_pdb(name1, v1.path)) + ah.get_radius(
                    parse_pdb(name2, v2.path)
                )
                r = jnp.linspace(0, r_max, r_steps)  # pyright: ignore [reportUnknownMemberType]
                phi1 = jnp.linspace(0, 2 * jnp.pi, phi_steps)  # pyright: ignore [reportUnknownMemberType]
                phi2 = jnp.linspace(0, 2 * jnp.pi, phi_steps)  # pyright: ignore [reportUnknownMemberType]
                k1, key = jax.random.split(key, 2)
                mffh.serialize_force_field(
                    str(pathlib.Path(out_dir) / f"{name1}_{name2}.safetensors"),
                    v1.path,
                    v1.num_groups,
                    v2.path,
                    v2.num_groups,
                    r,
                    phi1,
                    phi2,
                    k1,
                )
                del k1
