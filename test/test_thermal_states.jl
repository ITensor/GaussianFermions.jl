import GaussianFermions as gf
using Printf: @printf
using Test

include("utilities/hamiltonians.jl")
include("utilities/write_data.jl")

@testset "Spinless Fermion Chain" begin
    N = 10
    β = 10.0
    H = fermion_chain_h(N)
    # Add a potential
    V(j) = -(j-(N+1)/2)^2
    for j=1:N
        H = gf.add_hop(H,j,j,V(j))
    end
    ϕβ = gf.thermal_state(H, β)
    println("Made thermal state")

    dens = gf.density(ϕβ)
    @show dens
end

