
"""
"""
struct SpinGaussianState <: AbstractGaussianState
  ϕup::GaussianState
  ϕdn::GaussianState
end
