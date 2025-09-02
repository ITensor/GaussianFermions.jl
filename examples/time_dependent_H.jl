using FreeFermions

let
  N = 10
  Nfup = N÷2
  Nfdn = N÷2
  t1 = 1.0
  t2 = 0.5

  dt = 0.05

  H1 = FreeSpinHamiltonian(N)
  for j=1:(N-1)
    H1 = add_hop(H,-t1,j,j+1,"up")
    H1 = add_hop(H,-t1,j,j+1,"dn")
  end

  H2 = FreeSpinHamiltonian(N)
  for j=1:(N-2)
    H2 = add_hop(H,-t2,j,j+2,"up")
    H2 = add_hop(H,-t2,j,j+2,"dn")
  end

  E0, ϕ0 = ground_state(H1; Nfup, Nfdn)

  ϕt = copy(ϕ0)
  t = 0.0
  for n=1:Nstep
    ϕt = time_evolve(H1+f(t)*H2,dt,ϕt)
    t = t + dt
  end

  return
end
