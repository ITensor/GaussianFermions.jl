import LinearAlgebra as la

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

orbitals(ϕ::GaussianState) = ϕ.orbitals
filling(ϕ::GaussianState) = ϕ.filling
Base.length(ϕ::GaussianState) = size(orbitals(ϕ),1)

function correlation_matrix(ϕ::GaussianState; range=1:length(ϕ))
  orbs = orbitals(ϕ)[range,:]
  return orbs*la.Diagonal(filling(ϕ))*orbs'
end

function entanglement(ϕ::GaussianState, range)
  C = correlation_matrix(ϕ; range)
  occs, _ = la.eigen(C)
  Svn = 0.0
  for ν in occs
    if ν > 0.0
      Svn += -ν*log(ν)
    if ν < 1.0
      Svn += -(1-ν)*log(1-ν)
    end
  end
  return Svn
end

