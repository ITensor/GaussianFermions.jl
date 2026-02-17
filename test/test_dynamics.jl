using Test
import GaussianFermions as gf
using Printf: @printf

include("utilities/hamiltonians.jl")
include("utilities/write_data.jl")

@testset "Spinless Fermion Dynamics" begin
    N = 10
    H1 = fermion_chain_h(N)
    Nf = N ÷ 2
    E0, ϕ0 = gf.ground_state(H1; Nf = Nf - 1)

    # H2 is a perturbation on H1
    H2 = gf.GaussianOperator(N)
    H2 = gf.add_hop(H2, N ÷ 2, N ÷ 2 + 1, -1)

    H = H1 + H2

    densities = Float64[]
    dt = 0.02
    T = 100.0
    time_range = 0:dt:T
    ϕt = copy(ϕ0)
    for (n, t) in enumerate(time_range)
        if mod(t, 10.0) ≈ 0.0
            @printf("  t=%.3f\n", t)
        end
        ϕt = gf.time_evolve(H, dt, ϕt)
        dens = only(real(gf.density(ϕt, sites = (N ÷ 2):(N ÷ 2))))
        push!(densities, dens)
    end
    write_data("output/center_density.dat", time_range, densities)
end
