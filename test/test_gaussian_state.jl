using Test
using LinearAlgebra: norm
import GaussianFermions as gf

include("utilities/hamiltonians.jl")

@testset "Ground State" begin
    # Test ground state of a 1D chain
    N = 10
    H = fermion_chain_h(N)
    Nf = N ÷ 2
    E0, ϕ0 = gf.ground_state(H; Nf)
    @test gf.expect(H, ϕ0) ≈ E0

    # 1D electron chain
    H = electron_chain_h(N)
    Nf = N ÷ 2
    E0, ϕ0 = gf.ground_state(H; Nf)
    @test gf.expect(H, ϕ0) ≈ E0

    # Test for 2D square lattice
    H_graph = square_lattice_h(4)
    Nsites = length(gf.labels(H_graph))
    Nf_graph = Nsites ÷ 2
    E0_graph, ϕ0_graph = gf.ground_state(H_graph; Nf = Nf_graph)
    @test gf.expect(H_graph, ϕ0_graph) ≈ E0_graph
    @test gf.labels(ϕ0_graph) == gf.labels(H_graph)
end

@testset "Entanglement" begin
    N = 10
    H = fermion_chain_h(N)
    E0, ϕ0 = gf.ground_state(H; Nf = N ÷ 2)
    @test gf.entanglement(ϕ0; labels = 1:(N ÷ 2)) > 0
end
