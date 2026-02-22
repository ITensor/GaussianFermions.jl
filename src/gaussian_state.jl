import LinearAlgebra as la
using NamedArrays: NamedArray, names

"""
GaussianState

Represents a Gaussian (free fermion) state. Orbital or mode labels can be plain integer
site labels (spinless fermions) or [`Up`](@ref)/[`Dn`](@ref) spin labels
(spinful fermions).

A pure state has a `occupancy` vector with entries fₙ=0,1.
A mixed state can have fractional occupancies fₙ ∈ [0,1].

The single-particle density matrix or correlation matrix
for such a state is given by
C_ij = ∑ₙ ϕ_in f_n ϕ̄_jn
"""
struct GaussianState <: AbstractGaussianState
    orbitals::NamedArray
    occupancy::Vector
    nparticles::Integer
end

"""
    orbitals(ϕ::GaussianState)

Return the single-particle orbital matrix of the state `ϕ` as a `NamedArray`.
Rows are labeled by "orbital labels" or "mode labels" and columns by integers.
"""
orbitals(ϕ::GaussianState) = ϕ.orbitals

"""
    occupancy(ϕ::GaussianState)

Return the occupancy vector of `ϕ`. Each entry `fₙ` gives the occupation of
orbital `n`: `fₙ = 0` or `1` for a pure state, or `fₙ ∈ [0,1]` for a mixed state.
"""
occupancy(ϕ::GaussianState) = ϕ.occupancy

"""
    nparticles(ϕ::GaussianState)

Return the number of particles in the state `ϕ`.
"""
nparticles(ϕ::GaussianState) = ϕ.nparticles

Base.length(ϕ::GaussianState) = size(orbitals(ϕ), 1)

Base.copy(ϕ::GaussianState) = GaussianState(orbitals(ϕ), occupancy(ϕ), nparticles(ϕ))

"""
    labels(ϕ::GaussianState)

Return the orbital or mode labels of the state `ϕ`.
"""
labels(ϕ::GaussianState) = names(orbitals(ϕ), 1)

"""
    ispure(ϕ::GaussianState)

Return `true` if `ϕ` is a pure Gaussian state, i.e. all occupancy values are 0 or 1.
"""
ispure(ϕ::GaussianState) = all(f -> (f == 1.0 || f == 0.0), occupancy(ϕ))

"""
    has_spin(ϕ::GaussianState)

Return `true` if the orbital labels of `ϕ` are spin labels ([`Up`](@ref) or [`Dn`](@ref)).
"""
has_spin(ϕ::GaussianState) = first(names(ϕ.orbitals, 1)) isa Spin
