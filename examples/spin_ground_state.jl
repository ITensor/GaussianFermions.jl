import GaussianFermions as gf

let
  N = 20
  Nfup = N÷2
  Nfdn = N÷2
  t = 1.0

  H = gf.SpinGaussianOperator(N)
  for j in 1:(N - 1)
    H = gf.add_hop(H, j, j+1, -t; spin="up")
    H = gf.add_hop(H, j, j+1, -t; spin="dn")
  end

  display(gf.up_matrix_elements(H))

  E0, ϕ0 = gf.ground_state(H; Nfup, Nfdn)
  @show E0
  @show gf.expect(H, ϕ0)

  @show gf.entanglement(ϕ0, 1:5)
  @show gf.bond_dimension(ϕ0, 1:5, 7E-6)

  O = gf.SpinGaussianOperator(N)
  O = gf.add_cdag_c(O, 1, 2; spin="up")
  @show gf.expect(O, ϕ0)

  return nothing
end
