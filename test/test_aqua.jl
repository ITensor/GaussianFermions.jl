using Aqua: Aqua
using GaussianFermions: GaussianFermions
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(GaussianFermions)
end
