using Test
import GaussianFermions as gf
using Printf: @printf

include("utilities/hamiltonians.jl")
include("utilities/write_data.jl")

@testset "Spinless Fermion Dynamics" begin
    N = 10
    H1 = fermion_chain_h(N)
    Nf = N ÷ 2
    c = N ÷ 2  # center site
    E0, ϕ0 = gf.ground_state(H1; Nf = Nf - 1)

    # H2 is a perturbation on H1
    H2 = gf.GaussianOperator(N)
    H2 = gf.add_hop(H2, c, c + 1, -1)

    H = H1 + H2

    # Evaluate density at site c using Schrodinger picture
    densities = Float64[]
    dt = 0.02
    T = 100.0
    time_range = 0:dt:T
    ϕt = copy(ϕ0)
    for (n, t) in enumerate(time_range)
        dens = only(real(gf.density(ϕt; labels = c:c)))
        push!(densities, dens)

        ϕt = gf.time_evolve(H, dt, ϕt)
    end

    # Evaluate density at site c using Heisenberg picture
    heis_densities = ComplexF64[]
    Nop = gf.GaussianOperator(N)
    Nop += "Cdag",c,"C",c
    for (n, t) in enumerate(time_range)
        Nop_t = gf.time_evolve(H,t,Nop)
        push!(heis_densities, gf.expect(Nop_t, ϕ0))
    end
    @test heis_densities ≈ densities atol=1E-8
end

@testset "Greens Function Consistency" begin
    N = 10
    H = fermion_chain_h(N)
    Nf = N ÷ 2
    E0, ϕg = gf.ground_state(H; Nf)

    # Act with C†₁ |ϕ0⟩
    Cdag = gf.CreationOperator(1:N)
    Cdag += "C†",1
    ϕ0 = gf.apply(Cdag, ϕg)

    # Compute G>(t) with local quench approach
    dt = 0.05
    T = 100.0
    time_range = 0:dt:T

    # Compute G>(t) with local quench approach
    ϕt = gf.time_evolve(H, time_range, ϕ0)
    GG_quench = [-im*gf.inner(ϕ0,ϕt[n])*exp(im*E0*t) for (n,t) in enumerate(time_range)]

    # More explicit version looping over time steps
    GG_quench_loop = zeros(ComplexF64,length(time_range))
    ϕt = copy(ϕ0)
    for (n,t) in enumerate(time_range)
        GG_quench_loop[n] = -im*gf.inner(ϕ0,ϕt)*exp(im*E0*t)
        ϕt = gf.time_evolve(H,dt,ϕt)
    end

    # Compute G>(t) from formula
    GG = gf.greater_greens_function(H, time_range; labels=[1])

    @test GG_quench_loop ≈ GG_quench atol=1E-10
    @test GG_quench ≈ GG[:,1,1] atol=1E-10
end

@testset "Time Range time_evolve Function" begin
    N = 10
    H = fermion_chain_h(N)
    Nf = N ÷ 2
    E0, ϕ0 = gf.ground_state(H; Nf)

    dt = 0.05
    T = 5.0
    time_range = 0:dt:T

    # Evolve step-by-step from ϕ0
    densities_loop = Float64[]
    ϕt = copy(ϕ0)
    for t in time_range
        push!(densities_loop, only(real(gf.density(ϕt; labels = 1:1))))
        ϕt = gf.time_evolve(H, dt, ϕt)
    end

    # Evolve to all time points at once from ϕ0
    ϕs = gf.time_evolve(H, time_range, ϕ0)
    densities_range = [only(real(gf.density(ϕs[j]; labels = 1:1))) for j in 1:length(time_range)]

    @test densities_range ≈ densities_loop atol=1E-10
end
