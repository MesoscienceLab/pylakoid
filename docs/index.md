# Home

## Major Dependencies

These modules are implemented using [JAX](https://docs.jax.dev/en/latest/). JAX provides a NumPy-like interface, although the unique requirements of JAX lead to some occasional deviations from NumPy. JAX enables transformations such as just-in-time compilation, vectorized mapping, parallelized mapping, reverse and forward mode automatic differentiation, among others. In exchange for these incredibly powerful futures, programs using JAX need to submit to certain restrictions. The most notable restriction is that functions which are transformed by JAX must be [pure](https://en.wikipedia.org/wiki/Pure_function). Purity in turn imposes constraints on [pseudorandom number generation](https://docs.jax.dev/en/latest/random-numbers.html). [JAX 101](https://docs.jax.dev/en/latest/jax-101.html) is a useful resource to understand the basics of how JAX works.

[Equinox](https://docs.kidger.site/equinox/) provides a number of utilities for JAX that are used in this implementation. [jaxtyping](https://docs.kidger.site/jaxtyping/) makes it easier to express types for JAX, and those types are used extensively in this implementation.

## Overview

[Anneal](anneal.md) documents the Monte Carlo simulation for membrane annealing.

[Anneal Helpers](anneal_helpers.md) documents the functions which prepare the inputs needed for membrane annealing.

[Checker](checker.md) documents the classes which check whether configurations are valid for membrane annealing.

[Checker Helpers](checker_helpers.md) documents the functions which prepare checkers.

[Force Field](force_field.md) documents the calculation of force fields for membrane annealing.

[Force Field Helpers](force_field_helpers.md) documents the functions which prepare force fields for use in membrane annealing.

[Make Force Field](make_force_field.md) documents preparing parameters for force fields.

[Make Force Field Helpers](make_force_field_helpers.md) documents serializing and deserializing force fields.

[PDB](pdb.md) documents helpers for loading and preprocessing PDB files for use in pylakoid.

[Visualization](visualization.md) documents functions to visualize membranes.
