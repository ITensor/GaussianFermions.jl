# Spin-1/2 fermion chain ground state example.
# Builds a tight-binding chain for both spin species using Up(j)/Dn(j) vertex labels,
# finds the half-filled ground state, and measures energy, entanglement, bond dimension,
# and an off-diagonal correlation function in the up-spin sector.

import GaussianFermions as gf

let
    N = 20
    Nfup = N ÷ 2
    Nfdn = N ÷ 2
    t = 1.0

    ups = [gf.Up(j) for j in 1:N]
    dns = [gf.Dn(j) for j in 1:N]
    verts = vcat(ups, dns)

    H = gf.GaussianOperator(verts)
    for j in 1:(N - 1)
        H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -t)
        H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -t)
    end

    display(gf.matrix_elements(H))

    E0, ϕ0 = gf.ground_state(H; Nf = Nfup + Nfdn)
    @show E0
    @show gf.expect(H, ϕ0)

    left_sites = vcat([gf.Up(j) for j in 1:5], [gf.Dn(j) for j in 1:5])
    @show gf.entanglement(ϕ0; sites = left_sites)
    @show gf.bond_dimension(ϕ0, left_sites, 7.0e-6)

    O = gf.GaussianOperator(verts)
    O = gf.add_cdag_c(O, gf.Up(1), gf.Up(2))
    @show gf.expect(O, ϕ0)

    return nothing
end
