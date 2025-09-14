import LinearAlgebra as la

function expect(G::GaussianOperator, ψ::GaussianState)
  return la.tr(matrix_elements(G)*correlation_matrix(ψ))
end

function expect(G::SpinGaussianOperator, ψ::SpinGaussianState)
  return expect(up_operator(G), up_state(ψ)) + expect(dn_operator(G), dn_state(ψ))
end
end
