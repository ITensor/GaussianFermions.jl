
struct GaussianOperator
  coefficients::Matrix
end

function GaussianOperator(N::Integer)
  return GaussianOperator(zeros(N, N))
end

function add_hop(G::GaussianOperator, coef::Number, i::Integer, j::Integer)
  G = copy(G)
  G.coefficients[i, j] += coef
  G.coefficients[j, i] += coef
  return G
end

energies_states(G) = eigen(G.coefficients)
