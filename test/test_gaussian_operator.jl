using Test
using LinearAlgebra: norm, Diagonal
import GaussianFermions as gf
using NamedArrays: NamedArray

@testset "GaussianOperator Constructors" begin
    # Construct with integer dimension
    N = 10
    H = gf.GaussianOperator(N)
    @test length(H) == N
    @test norm(gf.matrix_elements(H)) < 1.0e-12

    # Construct with array of vertices
    verts = [(1, 1), (1, 2), (2, 1), (2, 2)]
    H = gf.GaussianOperator(verts)
    @test length(H) == length(verts)
    @test norm(gf.matrix_elements(H)) < 1.0e-12
end

@testset "Add Hop Function" begin
    # Construct with integer dimension
    N = 4
    H = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H = gf.add_hop(H, j, j + 1, -1)
    end
    h = [
        0 -1  0  0;
        -1  0 -1  0;
        0 -1  0 -1;
        0  0 -1  0
    ]
    @test norm(gf.matrix_elements(H) - h) < 1.0e-12

    # Construct with array of vertices
    # 2D square lattice Hamiltonian
    verts = [(i, j) for i in 1:N for j in 1:N]
    H_graph = gf.GaussianOperator(verts)
    for i in 1:N,j in 1:N
        (i < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i + 1, j), -1))
        (j < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i, j + 1), -1))
    end

    for r in verts, c in verts
        if (r[1] + 1, r[2]) == c || (r[1], r[2] + 1) == c
            @test gf.matrix_elements(H_graph, r, c) == -1
        elseif (c[1] + 1, c[2]) == r || (c[1], c[2] + 1) == r
            @test gf.matrix_elements(H_graph, r, c) == -1
        else
            @test gf.matrix_elements(H_graph, r, c) == 0
        end
    end
end

@testset "Energies and States" begin
    N = 4
    # Construct with array of vertices
    # 2D square lattice Hamiltonian
    verts = [(i, j) for i in 1:N for j in 1:N]
    H_graph = gf.GaussianOperator(verts)
    for i in 1:N,j in 1:N
        (i < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i + 1, j), -1))
        (j < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i, j + 1), -1))
    end
    ϵ, ϕ = gf.energies_states(H_graph)

    @test ϕ isa NamedArray
    @test names(ϕ, 1) == verts
    @test names(ϕ, 2) == 1:length(verts)

    @test ϵ isa Vector{Float64}
    @test length(ϵ) == length(verts)

    h_reconstruct = ϕ * Diagonal(ϵ) * ϕ'
    @test norm(gf.matrix_elements(H_graph) - h_reconstruct) < 1.0e-12
end
