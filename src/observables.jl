
function expect(G::GaussianOperator, ϕ::GaussianState)
  return trace(G.coefficients, correlation_matrix(ϕ))
end
