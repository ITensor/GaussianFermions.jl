using Test

@testset "GaussianFermions" begin
    include("test_gaussian_operator.jl")
    include("test_gaussian_state.jl")
    include("test_add_remove_orbital.jl")
    include("test_dynamics.jl")
    include("test_exact_dynamics.jl")
    include("mps/test_mps_ground_state.jl")
end
