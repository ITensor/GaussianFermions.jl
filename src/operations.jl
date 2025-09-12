using LinearAlgebra: LinearAlgebra

function expect(G::GaussianOperator, ϕ::GaussianState)
  return LinearAlgebra.tr(matrix_elements(G)*correlation_matrix(ϕ))
end

function expect(G::SpinGaussianOperator, ϕ::SpinGaussianState)
  return expect(up_operator(G), up_state(ϕ)) + expect(dn_operator(G), dn_state(ϕ))
end
