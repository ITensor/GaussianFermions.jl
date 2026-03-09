# Creation and Annihilation Operators

GaussianFermions.jl supports acting on Gaussian states with creation and annihilation
operators that are linear combinations of single-site operators. A key result of the
Gaussian fermion formalism is that applying such operators to a Gaussian state produces
another Gaussian state, so these operations remain efficient.

## Background

A general creation operator takes the form

```math
\hat{v}^\dagger = \sum_j v_j \, \hat{c}^\dagger_j
```

where ``v_j`` are complex coefficients and ``\hat{c}^\dagger_j`` creates a fermion
with label ``j``. Similarly, a general annihilation operator has the form

```math
\hat{w} = \sum_j \bar{w}_j \, \hat{c}_j
```

When acting on a Gaussian state ``|\psi\rangle`` with correlation matrix ``C_\psi``,
the resulting states are again Gaussian:

- **Creation**: ``|\tilde{\psi}\rangle = \hat{v}^\dagger |\psi\rangle`` has correlation
  matrix ``C_{\tilde{\psi}} = \|v_0\|^2 \, C_\psi + v_0 v_0^\dagger`` where
  ``v_0 = (1 - C_\psi) v`` is the projection onto the unoccupied subspace.

- **Annihilation**: ``|\tilde{\eta}\rangle = \hat{w} |\psi\rangle`` has correlation
  matrix ``C_{\tilde{\eta}} = \|w_1\|^2 \, C_\psi - w_1 w_1^\dagger`` where
  ``w_1 = C_\psi w`` is the projection onto the occupied subspace.

The resulting states are unnormalized. Their squared norms (``\|v_0\|^2`` or
``\|w_1\|^2``) are stored in the `trace` field of
the returned [`GaussianState`](@ref GaussianFermions.GaussianState).

## Types

```@docs
GaussianFermions.CreationOperator
GaussianFermions.AnnihilationOperator
```

## Applying Operators

```@docs
GaussianFermions.apply(::GaussianFermions.CreationOperator, ::GaussianFermions.GaussianState)
GaussianFermions.apply(::GaussianFermions.AnnihilationOperator, ::GaussianFermions.GaussianState)
```

## Usage

### Single-site operators

To act with a creation or annihilation operator on a single site, use a unit vector:

```julia
import GaussianFermions as gf

N = 10; Nf = 5
H = gf.GaussianOperator(N)
for j in 1:(N-1)
    H = gf.add_hop(H, j, j+1, -1.0)
end
_, ψ = gf.ground_state(H; Nf)

# Create a particle on site 1
v = zeros(N); v[1] = 1.0
Cd = gf.CreationOperator(1:N, v)
ψ_new = gf.apply(Cd, ψ)

# The trace gives ⟨ψ_new|ψ_new⟩
println(gf.trace(ψ_new))
```

### Linear combinations

To act with a superposition of operators, provide the coefficient vector:

```julia
# Annihilate with a random orbital
w = randn(N)
w /= norm(w)
C = gf.AnnihilationOperator(1:N, w)
ψ_ann = gf.apply(C, ψ)
```

### Spinful systems

For spinful (electron) systems using [`Up`](@ref GaussianFermions.Up)/[`Dn`](@ref GaussianFermions.Dn)
labels, the coefficient vector spans all modes:

```julia
using GaussianFermions: Up, Dn

N = 4
ups = [Up(j) for j in 1:N]
dns = [Dn(j) for j in 1:N]
labels = vcat(ups, dns)

H = gf.GaussianOperator(labels)
for j in 1:(N-1)
    H = gf.add_hop(H, Up(j), Up(j+1), -1.0)
    H = gf.add_hop(H, Dn(j), Dn(j+1), -1.0)
end
_, ψ = gf.ground_state(H; Nf=N)

# Create a spin-up particle on site 1
v = zeros(2N)
v[1] = 1.0  # Up(1) is the first label
Cd_up1 = gf.CreationOperator(labels, v)
ψ_new = gf.apply(Cd_up1, ψ)
```
