# Make Force Field

`pylakoid.structure.make_force_field` contains classes and functions to

- efficiently compute intersections between sets of spheres (in this context, a protein can be viewed as a set of spheres)
- efficiently and in parallel calculate parameters for force fields

## Public API

The public API of `pylakoid.structure.make_force_field` consists of the following classes and functions.

::: pylakoid.structure.make_force_field.count_sphere_intersections_fast
    options:
      heading_level: 3

::: pylakoid.structure.make_force_field.count_sphere_intersections_parallel
    options:
      heading_level: 3

::: pylakoid.structure.make_force_field.preprocess_for_count_sphere_intersections
    options:
      heading_level: 3

::: pylakoid.structure.make_force_field.Preprocess
    options:
      heading_level: 3
      members:
      - __init__

## Private API

The private API of `pylakoid.structure.make_force_field` consists of the following functions.

::: pylakoid.structure.make_force_field.count_sphere_intersections_between_groups
    options:
      heading_level: 3
