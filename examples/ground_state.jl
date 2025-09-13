import GaussianFermions as gf

let
  N = 10
  Nf = N÷2
  t = 1.0

  H = gf.GaussianOperator(N)
  for j in 1:(N - 1)
    H = gf.add_hop(H, j, j+1, -t)
  end

  display(gf.matrix_elements(H))

  E0, ϕ0 = gf.ground_state(H; Nf)
  @show E0
  @show gf.expect(H, ϕ0)

  @show gf.entanglement(ϕ0,1:5)
  @show gf.bond_dimension(ϕ0,1:5,1E-9)

  return nothing
end
