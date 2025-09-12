using LinearAlgebra: LinearAlgebra

struct GaussianOperator
  matrix_elems::Matrix
end

function GaussianOperator(N::Integer)
  return GaussianOperator(zeros(N, N))
end

Base.copy(G::GaussianOperator) = GaussianOperator(copy(G.matrix_elems))
Base.length(G::GaussianOperator) = size(G.matrix_elems, 1)

matrix_elements(G::GaussianOperator) = G.matrix_elems

function add_cdag_c(G::GaussianOperator, i::Integer, j::Integer, coef::Number=1.0)
  G = copy(G)
  G.matrix_elems[i, j] += coef
  return G
end

function add_c_cdag(G::GaussianOperator, i::Integer, j::Integer, coef::Number=1.0)
  G = copy(G)
  G.matrix_elems[j, i] += coef
  return G
end

function add_hop(G::GaussianOperator, i::Integer, j::Integer, coef::Number)
  G = add_cdag_c(G, i, j, coef)
  G = add_c_cdag(G, i, j, coef)
  return G
end

energies_states(G) = LinearAlgebra.eigen(G.matrix_elems)
