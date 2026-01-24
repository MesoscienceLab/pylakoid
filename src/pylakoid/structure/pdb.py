from Bio.PDB.PDBParser import PDBParser
import Bio.PDB.Structure
import numpy as np


def parse_pdb(name: str, path: str) -> Bio.PDB.Structure.Structure:
    """
    Parse a PDB file and center it in the XY plane.

    Parameters:
        name: The name passed to the `PDBParser`.
        path: The path of the PDB file to parse.

    Returns:
        The protein `Structure`.
    """

    structure = PDBParser().get_structure(name, path)  # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(structure, Bio.PDB.Structure.Structure):
        raise Exception("Could not parse PDB")
    # Centering the 2d center of mass is critical. Otherwise there is an extra transformation
    # due to rotating the proteins so they are both on the positive x axis.
    #
    # Unfortunately, Biopython loads in float32 by default which is not precise enough to
    # actually center the 2d center of mass. To make this work, we have to convert to float64
    # first.
    x_2d = np.array(
        [atom.coord[:2] for atom in structure.get_atoms()], dtype=np.float64
    )
    m = np.array([atom.mass for atom in structure.get_atoms()], dtype=np.float64)
    com = np.sum(np.expand_dims(m, 1) * x_2d, axis=0) / np.sum(m)
    # We can do this without worrying about statefulness because structure was created locally.
    structure.transform(rot=np.eye(3), tran=-np.pad(com, (0, 1)))  # pyright: ignore [reportUnknownMemberType]
    return structure
