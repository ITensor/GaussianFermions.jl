# GaussianFermions.jl

A Julia package providing tools for Gaussian (free) fermion calculations.

GaussianFermions.jl represents non-interacting fermionic systems through their
single-particle orbital matrices and filling vectors. This makes it possible to
efficiently compute ground states, time evolution, correlation functions,
entanglement entropy, and MPS bond dimensions for systems of any size, since all
operations reduce to linear algebra on matrices whose dimension is the number of
single-particle modes (sites) rather than the exponentially large many-body
Hilbert space.

Sites can be labeled by integers (spinless fermions), tuples (higher-dimensional
lattices), or [`Up`](@ref GaussianFermions.Up)/[`Dn`](@ref GaussianFermions.Dn)
spin labels (spinful fermions such as electrons), giving a flexible interface that
mirrors the generality of the underlying formalism.

See the [Background](@ref) page for an overview of the theory, or jump to the
API pages below.

## Contents

```@contents
Pages = [
    "theory.md",
    "operators.md",
    "states.md",
    "measurements.md",
    "time_evolution.md",
    "spinful.md",
]
Depth = 2
```
