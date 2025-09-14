using ITensorMPS

let
  N = 20
  t = 1.0

  h = OpSum()
  for j in 1:(N - 1)
    h += -t, "Cdagup", j, "Cup", j+1
    h += -t, "Cdagup", j+1, "Cup", j
    h += -t, "Cdagdn", j, "Cdn", j+1
    h += -t, "Cdagdn", j+1, "Cdn", j
  end
  sites = siteinds("Electron", N)
  H = MPO(h, sites)

  ψ0 = MPS(sites, [isodd(j) ? "Up" : "Dn" for j in 1:N])

  nsweeps = 10
  maxdim = [10, 20, 40, 80, 160]
  cutoff = 1E-12
  E0, ψ = dmrg(H, ψ0; nsweeps, cutoff, maxdim)
  @show E0

  trunc_cutoff = 1E-7
  ψtrunc = truncate(ψ; cutoff=trunc_cutoff)
  @show maxlinkdim(ψtrunc)
  @show 1-inner(ψ, ψtrunc)

  return nothing
end
