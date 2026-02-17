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

orbitals(ϕ::GaussianState) = ϕ.orbitals

filling(ϕ::GaussianState) = ϕ.filling

nparticles(ϕ::GaussianState) = ϕ.nparticles

Base.length(ϕ::GaussianState) = size(orbitals(ϕ), 1)

Base.copy(ϕ::GaussianState) = GaussianState(orbitals(ϕ), filling(ϕ), nparticles(ϕ))

vertices(ϕ::GaussianState) = names(orbitals(ϕ), 1)

ispure(ϕ::GaussianState) = all(f -> (f == 1.0 || f == 0.0), filling(ϕ))
