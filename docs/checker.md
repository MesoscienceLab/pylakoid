# Checker

`pylakoid.structure.checker` contains classes and functions which check if a protein is in a
valid configuration for membrane annealing.

## Public API

The public API of `pylakoid.structure.checker` consists of the following classes and functions.

::: pylakoid.structure.checker.Checker
    options:
      heading_level: 3

::: pylakoid.structure.checker.CenterOfMassInCircleChecker
    options:
      members:
      - __init__
      - __call__
      heading_level: 3

::: pylakoid.structure.checker.ConditionalChecker
    options:
      members:
      - __init__
      - __call__
      heading_level: 3

::: pylakoid.structure.checker.InsidePolygonChecker
    options:
      members:
      - __init__
      - __call__
      heading_level: 3

::: pylakoid.structure.checker.MultiChecker
    options:
      members:
      - __init__
      - __call__
      heading_level: 3

::: pylakoid.structure.checker.transform2d
    options:
      heading_level: 3

## Private API

The private API of `pylakoid.structure.checker` consists of the following functions.

::: pylakoid.structure.checker.is_point_in_polygon
    options:
      heading_level: 3
