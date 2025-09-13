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
Base.length(ϕ::GaussianState) = size(orbitals(ϕ), 1)

function correlation_matrix(ϕ::GaussianState; range=1:length(ϕ))
  orbs = orbitals(ϕ)[range, :]
  return orbs*la.Diagonal(filling(ϕ))*orbs'
end

function entanglement(ϕ::GaussianState, range)
  C = correlation_matrix(ϕ; range)
  occs, _ = la.eigen(C)
  Svn = 0.0
  for ν in occs
    if ν > 0.0
      Svn += -ν*log(ν)
    end
    if ν < 1.0
      Svn += -(1-ν)*log(1-ν)
    end
  end
  return Svn
end

inactivity(ν) = abs(2ν-1)

function bond_dimension(ϕ::GaussianState, range, cutoff::Real)
  C = correlation_matrix(ϕ; range)
  occs, _ = la.eigen(C)

  n = length(occs)
  inactivities = sort(inactivity.(occs); rev=true)

  # Decimate spectrum by "freezing" inactive modes
  # Each decimation doubles number of discarded eigenvalues
  # (ndiscard is number of discarded modes)
  fidelity = 1.0
  ndisc_modes = 0
  while fidelity*inactivities[ndisc_modes + 1]+cutoff > 1.0
    fidelity *= inactivities[ndisc_modes + 1]
    ndisc_modes += 1
  end

  nactive_modes = n-ndisc_modes
  inactivities = inactivities[(ndisc_modes + 1):n]

  # Explicitly compute remaining eigenvalues
  eigvals = zeros(2^nactive_modes)
  for (w, inds) in enumerate(Iterators.product(fill(0:1, nactive_modes)...))
    eigvals[w] = fidelity # account for modes already discarded
    for j in 1:nactive_modes
      ν, s = occs[j], inds[j]
      eigvals[w] *= ((1-s)*ν + s*(1-ν))
    end
  end
  eigvals = sort(eigvals)

  # Discard more eigvalues until infidelity exceeds cutoff
  ndisc_evals = 0
  while fidelity+cutoff-eigvals[ndisc_evals + 1] > 1.0
    fidelity -= eigvals[ndisc_evals + 1]
    ndisc_evals += 1
  end

  return 2^nactive_modes-ndisc_evals
end
