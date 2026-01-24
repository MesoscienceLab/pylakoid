import warnings

import Bio.PDB.PDBExceptions
import jax
import typer

from pylakoid.cli.forcefield import app as forcefield_app
from pylakoid.cli.groups import app as groups_app
from pylakoid.cli.simulate import app as simulate_app

warnings.filterwarnings("ignore", category=Bio.PDB.PDBExceptions.PDBConstructionWarning)
jax.config.update("jax_enable_x64", True)  # pyright: ignore [reportUnknownMemberType]

app = typer.Typer()
app.add_typer(forcefield_app)
app.add_typer(groups_app)
app.add_typer(simulate_app)
