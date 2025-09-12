
struct SpinGaussianOperator
  up_operator::GaussianOperator
  dn_operator::GaussianOperator
end

function SpinGaussianOperator(N::Integer)
  return SpinGaussianOperator(GaussianOperator(N), GaussianOperator(N))
end

up_operator(G::SpinGaussianOperator) = G.up_operator
dn_operator(G::SpinGaussianOperator) = G.dn_operator

up_matrix_elements(G::SpinGaussianOperator) = matrix_elements(up_operator(G))
dn_matrix_elements(G::SpinGaussianOperator) = matrix_elements(dn_operator(G))

function Base.copy(G::SpinGaussianOperator)
  SpinGaussianOperator(copy(G.up_operator), copy(G.dn_operator))
end
Base.length(G::SpinGaussianOperator) = length(up_operator(G))

function call_function(G::SpinGaussianOperator, func::Function, args...; spin::String="")
  up_op, dn_op = copy(G.up_operator), copy(G.dn_operator)
  if spin=="up" || spin==""
    up_op = func(up_op, args...)
  end
  if spin=="dn" || spin==""
    dn_op = func(dn_op, args...)
  end
  return SpinGaussianOperator(up_op, dn_op)
end

function add_cdag_c(
  G::SpinGaussianOperator, i::Integer, j::Integer, coef::Number=1.0; spin::String=""
)
  return call_function(G, add_cdag_c, i, j, coef; spin)
end

function add_c_cdag(
  G::SpinGaussianOperator, i::Integer, j::Integer, coef::Number=1.0; spin::String=""
)
  return call_function(G, add_c_cdag, i, j, coef; spin)
end

function add_hop(
  G::SpinGaussianOperator, i::Integer, j::Integer, coef::Number=1.0; spin::String=""
)
  return call_function(G, add_hop, i, j, coef; spin)
end
