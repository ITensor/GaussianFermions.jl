
struct SpinGaussianState <: AbstractGaussianState
  ϕup::GaussianState
  ϕdn::GaussianState
end

up_state(ϕ::SpinGaussianState) = ϕ.ϕup
dn_state(ϕ::SpinGaussianState) = ϕ.ϕdn

up_orbitals(ϕ::SpinGaussianState) = orbitals(up_state(ϕ))
dn_orbitals(ϕ::SpinGaussianState) = orbitals(dn_state(ϕ))

function entanglement(ϕ::SpinGaussianState, range) 
  return entanglement(up_state(ϕ),range)+entanglement(dn_state(ϕ),range)
end

function bond_dimension(ϕ::SpinGaussianState, range, cutoff::Real)
  return bond_dimension(up_state(ϕ),range,cutoff/2)*bond_dimension(dn_state(ϕ),range,cutoff/2)
end
