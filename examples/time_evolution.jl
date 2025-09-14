import GaussianFermions as gf

let
  N = 20
  Nf = N÷2

  H = gf.GaussianOperator(N)
  for j in 2:(N - 1)
    H = gf.add_hop(H, j, j+1, -1)
  end

  B = gf.GaussianOperator(N)
  B = gf.add_hop(H, 1, 2, -1)

  display(gf.matrix_elements(H))

  E0, ϕ0 = gf.ground_state(H; Nf)
  @show E0
  @show gf.expect(H, ϕ0)
  @show gf.entanglement(ϕ0, 1:5)
  @show gf.bond_dimension(ϕ0, 1:5, 1E-7)

  A = gf.GaussianOperator(N)
  A = gf.add_cdag_c(A, 1, 2)

  dt = 0.02
  T = 5.0
  time_range = 0:dt:T
  At = copy(A)
  Avals = zeros(length(time_range))
  for (n, t) in enumerate(time_range)
    Ht = H + f(t)*B
    At = time_evolve(Ht, dt, At)
    Avals[n] = gf.expect(H, ϕ0)
  end

  return nothing
end
