module GaussianFermions

include("gaussian_operator.jl")
include("spin_gaussian_operator.jl")
include("abstract_gaussian_state.jl")
include("gaussian_state.jl")
include("spin_gaussian_state.jl")

include("state_constructors.jl")
include("observables.jl")

export add_hop

end # module GaussianFermions
