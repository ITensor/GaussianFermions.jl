import Base: *, +

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

function (x::Number * G::SpinGaussianOperator)
  return SpinGaussianOperator(x*up_operator(G), x*dn_operator(G))
end

(G::SpinGaussianOperator * x::Number) = x*G

function (A::SpinGaussianOperator + B::SpinGaussianOperator)
  return SpinGaussianOperator(up_operator(A)+up_operator(B), dn_operator(A)+dn_operator(B))
end

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

function expect(G::SpinGaussianOperator, ψ::SpinGaussianState)
  return expect(up_operator(G), up_state(ψ)) + expect(dn_operator(G), dn_state(ψ))
end

function time_evolve(H::SpinGaussianOperator, t::Number, ψ::SpinGaussianState)
  up_state_t = time_evolve(up_operator(H), t, up_state(ψ))
  dn_state_t = time_evolve(dn_operator(H), t, dn_state(ψ))
  return SpinGaussianState(up_state_t, dn_state_t)
end

function time_evolve(H::SpinGaussianOperator, t::Number, O::SpinGaussianOperator)
  O_up_t = time_evolve(up_operator(H), t, up_operator(O))
  O_dn_t = time_evolve(dn_operator(H), t, dn_operator(O))
  return SpinGaussianOperator(O_up_t, O_dn_t)
end
