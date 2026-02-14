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
