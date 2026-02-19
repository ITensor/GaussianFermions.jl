# Spin-1/2 fermion chain time evolution under a Gaussian pulse.
# The bulk chain Hamiltonian H excludes the 1→2 bond; a Gaussian drive adds that bond.
# The state is propagated in the Schrödinger picture and the expectation value of
# A = c†_{up,1} c_{up,2} is recorded along with entanglement and bond dimension over time.

import GaussianFermions as gf
using Printf

include("utilities/write_data.jl")

let
    N = 31
    Nfup = (N - 1) ÷ 2
    Nfdn = (N - 1) ÷ 2
    bond_dim_cutoff = 1.0e-6

    ups = [gf.Up(j) for j in 1:N]
    dns = [gf.Dn(j) for j in 1:N]
    verts = vcat(ups, dns)

    H = gf.GaussianOperator(verts)
    for j in 2:(N - 1)
        H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1)
        H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1)
    end

    B = gf.GaussianOperator(verts)
    B = gf.add_hop(B, gf.Up(1), gf.Up(2), -1)
    B = gf.add_hop(B, gf.Dn(1), gf.Dn(2), -1)

    E0, ψ0 = gf.ground_state(H; Nf = Nfup + Nfdn)
    @show E0
    @show gf.expect(H, ψ0)

    left_sites = vcat([gf.Up(j) for j in 1:(N ÷ 2)], [gf.Dn(j) for j in 1:(N ÷ 2)])
    @show gf.entanglement(ψ0; sites = left_sites)
    @show gf.bond_dimension(ψ0, left_sites, bond_dim_cutoff)

    A = gf.GaussianOperator(verts)
    A = gf.add_cdag_c(A, gf.Up(1), gf.Up(2))

    t0 = 4.0
    σ = 0.2
    coef = 0.1
    field(t) = coef * exp(-(t - t0)^2 / (2σ^2)) / √(2π * σ^2)

    @show σ
    @show field(0)

    dt = 0.02
    T = 1000.0
    time_range = 0:dt:T
    Avals = zeros(ComplexF64, length(time_range))
    entanglement_vals = zeros(length(time_range))
    bond_dimension_vals = zeros(Int, length(time_range))

    ψt = copy(ψ0)
    for (n, t) in enumerate(time_range)
        if mod(t, 10.0) ≈ 0.0
            @printf("  t=%.3f\n", t)
        end
        Ht = H + field(t) * B
        ψt = gf.time_evolve(Ht, dt, ψt)
        Avals[n] = gf.expect(A, ψt)
        entanglement_vals[n] = gf.entanglement(ψt; sites = left_sites)
        bond_dimension_vals[n] = gf.bond_dimension(ψt, left_sites, bond_dim_cutoff)
    end

    times = collect(time_range)

    write_data("output/A_real.dat", times, real(Avals))
    write_data("output/A_imag.dat", times, imag(Avals))
    write_data("output/field.dat", times, field.(times))
    write_data("output/entanglement.dat", times, entanglement_vals)
    write_data("output/bond_dimension.dat", times, bond_dimension_vals)

    return nothing
end
