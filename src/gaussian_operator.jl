using LinearAlgebra: LinearAlgebra

struct GaussianOperator
  coefficients::Matrix
end

function GaussianOperator(N::Integer)
  return GaussianOperator(zeros(N, N))
end

Base.copy(G) = GaussianOperator(copy(G.coefficients))
Base.length(G) = size(G.coefficients, 1)

function add_cdag_c(G::GaussianOperator, i::Integer, j::Integer, coef::Number=1.0)
  G = copy(G)
  G.coefficients[i, j] += coef
  return G
end

function add_c_cdag(G::GaussianOperator, i::Integer, j::Integer, coef::Number=1.0)
  G = copy(G)
  G.coefficients[j, i] += coef
  return G
end

function add_hop(G::GaussianOperator, i::Integer, j::Integer, coef::Number)
  G = add_cdag_c(G, i, j, coef)
  G = add_c_cdag(G, i, j, coef)
  return G
end

energies_states(G) = LinearAlgebra.eigen(G.coefficients)
