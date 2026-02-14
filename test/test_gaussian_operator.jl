using Test
using LinearAlgebra: norm
import GaussianFermions as gf

@testset "GaussianOperator Constructors" begin
    # Construct with integer dimension
    N = 10
    H = gf.GaussianOperator(N)
    @test length(H) == N
    @test norm(gf.matrix_elements(H)) < 1E-12

    # Construct with array of vertices
    verts = [(1,1),(1,2),(2,1),(2,2)]
    H = gf.GaussianOperator(verts)
    @test length(H) == length(verts)
    @test norm(gf.matrix_elements(H)) < 1E-12
end

@testset "Add Hop Function" begin
    # Construct with integer dimension
    N = 4
    H = gf.GaussianOperator(N)
    for j=1:(N-1)
        H = gf.add_hop(H, j, j+1, -1)
    end
    h = [ 0 -1  0  0;
         -1  0 -1  0;
          0 -1  0 -1;
          0  0 -1  0]
    @test norm(gf.matrix_elements(H) - h) < 1E-12

    # Construct with array of vertices
    verts = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
    H_graph = gf.GaussianOperator(verts)
    for i=1:3,j=1:3
        (i < 3) && (H_graph = gf.add_hop(H_graph, (i,j), (i+1,j), -1))
        (j < 3) && (H_graph = gf.add_hop(H_graph, (i,j), (i,j+1), -1))
    end
    @test gf.matrix_elements(H_graph,(1,1),(2,1)) == -1
    @test gf.matrix_elements(H_graph,(1,1),(1,2)) == -1
    @test gf.matrix_elements(H_graph,(2,1),(3,1)) == -1
    @test gf.matrix_elements(H_graph,(2,1),(2,2)) == -1
    @test gf.matrix_elements(H_graph,(1,1),(3,3)) == 0
end
