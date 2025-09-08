using LinearAlgebra: Diagonal

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
  orbitals::Matrix
  filling::Vector
end

function correlation_matrix(ϕ::GaussianState)
  return ϕ.orbitals*Diagonal(ϕ.filling)*ϕ.orbitals'
end
