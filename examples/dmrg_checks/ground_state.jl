using ITensorMPS

let
  N = 10
  t = 1.0

  h = OpSum()
  for j in 1:(N - 1)
    h += -t,"Cdag",j,"C",j+1
    h += -t,"Cdag",j+1,"C",j
  end
  sites = siteinds("Fermion",N)
  H = MPO(h,sites)

  ψ0 = MPS(sites,[isodd(j) ? "1" : "0" for j=1:N])

  nsweeps = 20
  maxdim = [10,20,40,80,160,320]
  cutoff = 1E-9
  E0, ψ = dmrg(H,ψ0; nsweeps, cutoff, maxdim)
  @show E0

  @show maxlinkdim(ψ)

  return nothing
end
