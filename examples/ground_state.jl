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

    display(gf.matrix_elements(H))

    E0, ϕ0 = gf.ground_state(H; Nf)
    @show E0
    @show gf.expect(H, ϕ0)

    @show gf.entanglement(ϕ0; labels = 1:5)
    @show gf.bond_dimension(ϕ0, 1:5, 1.0e-7)

    return nothing
end
