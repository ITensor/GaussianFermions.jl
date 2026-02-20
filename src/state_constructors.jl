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
    if isnothing(Nf)
        Nf = count(<(0), ϵ)
    else
        (Nf > N) && error("Number of fermions Nf cannot be greater than system size")
    end
    filling = vcat(ones(Nf), zeros(N - Nf))
    E = 0.0
    for n in 1:length(filling)
        E += ϵ[n] * filling[n]
    end
    return E, GaussianState(ϕ, filling, Nf)
end
