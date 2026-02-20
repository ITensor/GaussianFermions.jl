import LinearAlgebra as la
using NamedArrays: NamedArray, names

"""
GaussianState

Represents a Gaussian state of a single "species"
(e.g. spinless fermions or fermions of the same spin label).
A pure state has a `filling` vector with entries fₙ=0,1.
A mixed state can have fractional fillings fₙ ∈ [0,1].

The single-particle density matrix or correlation matrix
for such a state is given by
C_ij = ∑ₙ ϕ_in f_n ϕ̄_jn 
"""
struct GaussianState <: AbstractGaussianState
    orbitals::NamedArray
    filling::Vector
    nparticles::Integer
end

"""
    orbitals(ϕ::GaussianState)

Return the single-particle orbital matrix of the state `ϕ` as a `NamedArray`.
Rows are labeled by vertices (sites) and columns by orbital indices.
"""
orbitals(ϕ::GaussianState) = ϕ.orbitals

"""
    filling(ϕ::GaussianState)

Return the filling vector of `ϕ`. Each entry `fₙ` gives the occupation of
orbital `n`: `fₙ = 0` or `1` for a pure state, or `fₙ ∈ [0,1]` for a mixed state.
"""
filling(ϕ::GaussianState) = ϕ.filling

"""
    nparticles(ϕ::GaussianState)

Return the number of particles in the state `ϕ`.
"""
nparticles(ϕ::GaussianState) = ϕ.nparticles

Base.length(ϕ::GaussianState) = size(orbitals(ϕ), 1)

Base.copy(ϕ::GaussianState) = GaussianState(orbitals(ϕ), filling(ϕ), nparticles(ϕ))

"""
    vertices(ϕ::GaussianState)

Return the vertex labels (site indices) of the state `ϕ`.
"""
vertices(ϕ::GaussianState) = names(orbitals(ϕ), 1)

"""
    ispure(ϕ::GaussianState)

Return `true` if `ϕ` is a pure Gaussian state, i.e. all filling values are 0 or 1.
"""
ispure(ϕ::GaussianState) = all(f -> (f == 1.0 || f == 0.0), filling(ϕ))

"""
    has_spin(ϕ::GaussianState)

Return `true` if the vertex labels of `ϕ` are spin labels ([`Up`](@ref) or [`Dn`](@ref)).
"""
has_spin(ϕ::GaussianState) = first(names(ϕ.orbitals, 1)) isa Spin
