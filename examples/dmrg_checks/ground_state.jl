# DMRG reference calculation for a spinless fermion tight-binding chain.
# Uses ITensorMPS to find the ground state energy and MPS bond dimension at a given
# truncation cutoff, for comparison against the exact Gaussian-state results.

using ITensorMPS

let
    N = 20
    t = 1.0

    h = OpSum()
    for j in 1:(N - 1)
        h += -t, "Cdag", j, "C", j + 1
        h += -t, "Cdag", j + 1, "C", j
    end
    sites = siteinds("Fermion", N)
    H = MPO(h, sites)

    ψ0 = MPS(sites, [isodd(j) ? "1" : "0" for j in 1:N])

    nsweeps = 20
    maxdim = [10, 20, 40, 80, 160, 320]
    cutoff = 1.0e-12
    E0, ψ = dmrg(H, ψ0; nsweeps, cutoff, maxdim)
    @show E0

    trunc_cutoff = 1.0e-7
    ψtrunc = truncate(ψ; cutoff = trunc_cutoff)
    @show maxlinkdim(ψtrunc)
    @show 1 - inner(ψ, ψtrunc)

    return nothing
end
