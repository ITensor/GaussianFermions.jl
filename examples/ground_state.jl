using FreeFermions

let
  N = 10
  Nfup = N÷2
  Nfdn = N÷2
  t = 1.0

  H = FreeSpinHamiltonian(N)
  for j=1:(N-1)
    H = add_hop(H,-t,j,j+1,"up")
    H = add_hop(H,-t,j,j+1,"dn")
  end

  E0, ϕ0 = ground_state(H; Nfup, Nfdn)

  C = correlation_matrix(ϕ0)

  return
end
