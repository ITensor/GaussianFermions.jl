import LinearAlgebra as la

function expect(G::GaussianOperator, ψ::GaussianState)
  return la.tr(matrix_elements(G)*correlation_matrix(ψ))
end

function expect(G::SpinGaussianOperator, ψ::SpinGaussianState)
  return expect(up_operator(G), up_state(ψ)) + expect(dn_operator(G), dn_state(ψ))
end

function greens_function(H::GaussianOperator, t::Number)
  ϵ, ϕ = energies_states(G)
  exp_itϵ = [exp(-im*t*ϵ[n]) for n=1:length(ϵ)]
  return ϕ*la.Diagonal(exp_itϵ)*ϕ'
end

function time_evolve(H::GaussianOperator, ψ::GaussianState, t::Number)
  Gt = greens_function(H,t)
  orbs_t = Gt*orbitals(ψ)
  return GaussianState(orbs_t, filling(ψ))
end
