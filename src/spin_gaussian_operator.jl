
struct SpinGaussianOperator
  up_operator::GaussianOperator
  dn_operator::GaussianOperator
end

function SpinGaussianOperator(N::Integer)
  return SpinGaussianOperator(GaussianOperator(N), GaussianOperator(N))
end

function add_hop(
  G::SpinGaussianOperator, coef::Number, i::Integer, j::Integer, spin::String=""
)
  up_op, dn_op = copy(G.up_operator), copy(G.dn_operator)
  if spin=="up" || spin==""
    up_op = add_hop(up_op, coef, i, j)
  end
  if spin=="dn" || spin==""
    dn_op = add_hop(dn_op, coef, i, j)
  end
  return SpinGaussianOperator(up_op, dn_op)
end
