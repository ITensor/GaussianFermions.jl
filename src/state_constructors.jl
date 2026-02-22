"""
    ground_state_occupancies(ϵ; Nf = nothing)

Given an array of energies, return a vector of 
occupancies (νⱼ = 0,1) corresponding to the ground
state configuration for those energies.
"""
function ground_state_occupancies(ϵ; Nf = nothing)
    if isnothing(Nf)
        # Occupy all negative-energy levels
        return map(x -> (x < 0 ? 1 : 0), ϵ)
    end
    N = length(ϵ)
    (Nf > N) && error("Number of fermions Nf cannot be greater than system size")
    # Sort levels by energy and fill the Nf lowest ones
    perm = sortperm(ϵ)
    occupancies = vcat(ones(Nf), zeros(N - Nf))
    return occupancies[invperm(perm)]
end

"""
    ground_state(G::GaussianOperator; Nf=nothing)

Compute the ground state of the Gaussian operator `G` with `Nf` fermions.
If `Nf` is not specified, all negative-energy orbitals are filled.

Diagonalizes the hopping matrix to obtain single-particle energies
``\\epsilon_n`` and eigenstates, then fills the `Nf` lowest-energy levels.
The ground state energy is ``E_0 = \\sum_{n=1}^{N_f} \\epsilon_n``.

Returns a tuple `(E0, ϕ0)` where `E0` is the ground state energy and
`ϕ0` is the corresponding [`GaussianState`](@ref).

# Example
```julia
import GaussianFermions as gf

N = 10
H = gf.GaussianOperator(N)
for j in 1:(N - 1)
    H = gf.add_hop(H, j, j + 1, -1.0)
end
E0, ϕ0 = gf.ground_state(H; Nf=N ÷ 2)
```
"""
function ground_state(G::GaussianOperator; Nf = nothing)
    N = length(G)
    ϵ, ϕ = energies_states(G)
    occupancies = ground_state_occupancies(ϵ; Nf)
    Nf = count(==(1), occupancies)
    E = 0.0
    for n in 1:length(occupancies)
        E += ϵ[n] * occupancies[n]
    end
    return E, GaussianState(ϕ, occupancies, Nf)
end
