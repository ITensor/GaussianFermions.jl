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
    trace::Float64
    function GaussianState(orbitals::NamedArray, occupancy::Vector, trace::Float64=1.0)
        return new(orbitals, occupancy, trace)
    end
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
    trace(ϕ::GaussianState)

Return the trace of the density matrix of the Gaussian state. For a pure
Gaussian state |ψ⟩ this is ⟨ψ|ψ⟩. Note that this is a separately stored number
and is not computed from the orbital or occupancy data of the Gaussian state.
"""
trace(ϕ::GaussianState) = ϕ.trace

"""
    norm(ϕ::GaussianState)

For a pure Gaussian state |ψ⟩ returns the norm √⟨ψ|ψ⟩.
"""
la.norm(ϕ::GaussianState) = sqrt(trace(ϕ))

Base.length(ϕ::GaussianState) = size(orbitals(ϕ), 1)

Base.copy(ϕ::GaussianState) = GaussianState(orbitals(ϕ), occupancy(ϕ), trace(ϕ))

"""
    normalize(ϕ::GaussianState)

Return a Gaussian state equivalent to ϕ but with unit norm i.e. a trace of 1.
"""
normalize(ϕ::GaussianState) = GaussianState(orbitals(ϕ), occupancy(ϕ), 1.0)

"""
    labels(ϕ::GaussianState)

Return the orbital or mode labels of the state `ϕ`.
"""
labels(ϕ::GaussianState) = names(orbitals(ϕ), 1)

"""
    ispure(ϕ::GaussianState; tol = 1E-12)

Return `true` if `ϕ` is a pure Gaussian state, i.e. all occupancy values are 0 or 1.
The optional `tol` keyword sets the precision at which an occupancy value is 
considered to be 0 or 1.
"""
ispure(ϕ::GaussianState; tol = 1E-12) = all(f -> (abs(f-1) < tol || abs(f) < tol), occupancy(ϕ))

"""
    has_spin(ϕ::GaussianState)

Return `true` if the orbital labels of `ϕ` are spin labels ([`Up`](@ref) or [`Dn`](@ref)).
"""
has_spin(ϕ::GaussianState) = first(names(ϕ.orbitals, 1)) isa Spin
