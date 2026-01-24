# Pylakoid

Pylakoid enables simulation of thylakoid membranes under the assumption that
steric hindrance (measured by counting atoms closer than the sum of their Van
der Waals radii) is the only important protein-protein interaction.

## Package Documentation

Run `mkdocs serve` to view the documentation.

## Command-Line Interface

To see all available commands and full documentation, run `pylakoid --help`.

### groups

Pylakoid uses a fast algorithm to count atoms closer than the sum of their Van
der Waals radii. The performance of this algorithm depends on a parameter
called `num_groups`. The `groups` command is used to estimate reasonable values
for this parameter.

### forcefield

Pylakoid uses interpolated force fields to accelerate simulation. The
`forcefield` command is used to parameterize the interpolated forcefield. This
can be parallelized by passing the `--num-cpu` option.

This command requires a JSON configuration file with the following format
```
{
    "protein1": {
        "path": "/path/to/protein1",
        "num_groups": <NUM_GROUPS_1>
    },
    "protein2": {
        "path": "/path/to/protein2",
        "num_groups": <NUM_GROUPS_2>
    },
    "protein3": {
        "path": "/path/to/protein3",
        "num_groups": <NUM_GROUPS_3>
    }
}
```
where there can be any number of items in the dictionary.

### simulate

Pylakoid simulates membranes using parallel tempering. The `simulate` command
runs these simulations. This can be parallelized by passing the `--num-cpu`
option.

This command requires a JSON configuration file with the following format
```
{
  "boundary": {
    "xmin": <BOUNDARY_X_MIN>,
    "xmax": <BOUNDARY_X_MAX>,
    "ymin": <BOUNDARY_Y_MIN>,
    "ymax": <BOUNDARY_Y_MAX>
  },
  "pdb": {
    "protein1": {
        "path": "/path/to/protein1",
        "num_groups": <NUM_GROUPS_1>,
        "protein_count": <PROTEIN_COUNT_IN_MEMBRANE_1>
    },
    "protein2": {
        "path": "/path/to/protein2",
        "num_groups": <NUM_GROUPS_2>,
        "protein_count": <PROTEIN_COUNT_IN_MEMBRANE_2>
    },
    "protein3": {
        "path": "/path/to/protein3",
        "num_groups": <NUM_GROUPS_3>,
        "protein_count": <PROTEIN_COUNT_IN_MEMBRANE_3>
    }
  },
  "force_field_dir": "/path/to/force_field_dir",
  "parameters": {
    "num_epochs": 10,
    "num_swaps_per_epoch": 100,
    "num_steps_between_swaps_per_protein": 1,
    "num_membranes": 48,
    "r_max": {
      "start": 100,
      "end": 5
    },
    "phi_max": {
      "start": 6.28318,
      "end": 0.08726
    },
    "kBT": {
      "start": 1000000,
      "end": 1
    }
  }
}
```
where all values are examples and should be changed to fit your simulation.
