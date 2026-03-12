# Spin-1/2 fermion chain ground state example.
# Builds a tight-binding chain for both spin species using Up(j)/Dn(j) vertex labels,
# finds the half-filled ground state, and measures energy, entanglement, bond dimension,
# and an off-diagonal correlation function in the up-spin sector.

import GaussianFermions as gf
using GaussianFermions: Up, Dn

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
        H += -t, "C†", Up(j),   "C", Up(j+1)
        H += -t, "C†", Up(j+1), "C", Up(j)
        H += -t, "C†", Dn(j),   "C", Dn(j+1)
        H += -t, "C†", Dn(j+1), "C", Dn(j)
    end

    E0, ϕ0 = gf.ground_state(H; Nf = Nfup + Nfdn)
    @show E0
    @show gf.expect(H, ϕ0)

    left_sites = vcat([gf.Up(j) for j in 1:5], [gf.Dn(j) for j in 1:5])
    @show gf.entanglement(ϕ0, left_sites)
    @show gf.bond_dimension(ϕ0, left_sites, 7.0e-6)

    O = gf.GaussianOperator(verts)
    O += "C†", gf.Up(1), "C", gf.Up(2)
    @show gf.expect(O, ϕ0)

    return
end
