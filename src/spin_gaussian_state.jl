
"""
"""
struct SpinGaussianState <: AbstractGaussianState
  ϕup::GaussianState
  ϕdn::GaussianState
end

up_state(ϕ::SpinGaussianState) = ϕ.ϕup
dn_state(ϕ::SpinGaussianState) = ϕ.ϕdn
