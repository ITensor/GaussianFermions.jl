
function ground_state(G::GaussianOperator; Nf=nothing)
  N = length(G)
  ϵ, ϕ = energies_states(G)
  if isnothing(Nf)
    Nf = count(<(0), ϵ)
  else
    (Nf > N) && error("Number of fermions Nf cannot be greater than system size")
  end
  filling = vcat(ones(Nf), zeros(N-Nf))
  return GaussianState(ϕ, filling)
end
