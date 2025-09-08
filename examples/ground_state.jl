import GaussianFermions as GF

let
  N = 10
  Nfup = N÷2
  Nfdn = N÷2
  t = 1.0

  H = GF.SpinGaussianOperator(N)
  for j in 1:(N - 1)
    H = GF.add_hop(H, -t, j, j+1, "up")
    H = GF.add_hop(H, -t, j, j+1, "dn")
  end

  @show H

  return nothing

  E0, ϕ0 = GF.ground_state(H; Nfup, Nfdn)

  C = GF.correlation_matrix(ϕ0)

  return nothing
end
