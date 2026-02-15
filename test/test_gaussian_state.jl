using Test
using LinearAlgebra: norm
import GaussianFermions as gf

@testset "Compute Ground State and Properties" begin
    # Construct with integer dimension
    # 1D chain Hamiltonian
    N = 10
    H = gf.GaussianOperator(N)
    for j=1:(N-1)
        H = gf.add_hop(H, j, j+1, -1)
    end

    Nf = N÷2
    E0, ϕ0 = gf.ground_state(H; Nf)
    @test gf.expect(H, ϕ0) ≈ E0

    # Construct with array of vertices
    # 2D square lattice Hamiltonian
    verts = [(i,j) for i=1:N for j=1:N]
    H_graph = gf.GaussianOperator(verts)
    for i=1:N,j=1:N
        (i < N) && (H_graph = gf.add_hop(H_graph, (i,j), (i+1,j), -1))
        (j < N) && (H_graph = gf.add_hop(H_graph, (i,j), (i,j+1), -1))
    end
    Nsites = length(verts)
    Nf_graph = Nsites÷2
    E0_graph, ϕ0_graph = gf.ground_state(H_graph; Nf=Nf_graph)
    @test gf.expect(H_graph, ϕ0_graph) ≈ E0_graph
end
