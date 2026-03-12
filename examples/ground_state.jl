# Spinless fermion chain ground state example.
# Builds a nearest-neighbor tight-binding chain, finds the half-filled ground state,
# and computes the energy, entanglement entropy, and MPS bond dimension of the left half.

import GaussianFermions as gf

let
    N = 20
    Nf = N ÷ 2
    t = 1.0

    H = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H += -t, "C†", j, "C", j+1
        H += -t, "C†", j+1, "C", j
    end

    E0, ϕ0 = gf.ground_state(H; Nf)
    @show E0
    @show gf.expect(H, ϕ0)
    @show sum(gf.density(ϕ0))

    region_A = 1:(N÷2)

    Sₐ = gf.entanglement(ϕ0; labels = region_A)
    println("Entanglement of region A (=$region_A) is Sₐ = ",Sₐ)

    cutoff = 1E-8
    χ, trunc_error = gf.bond_dimension(ϕ0, region_A, 1E-8)
    println("Truncating to cutoff $cutoff results in bond dimension χ = ",χ)


    return nothing
end
