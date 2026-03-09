module GaussianFermions

include("utilities/namedarrays_extensions.jl")

include("spin_types.jl")
include("abstract_gaussian_state.jl")
include("gaussian_state.jl")
include("gaussian_operator.jl")
include("creation_annihilation_operator.jl")

include("state_constructors.jl")
include("state_properties.jl")

include("spinful_state_properties.jl")

end # module GaussianFermions
