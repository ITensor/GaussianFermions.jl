import LinearAlgebra as la
import Base: *, +
using NamedArrays: NamedArray

"""
    GaussianOperator(N::Integer)
    GaussianOperator(labels)

Create a Gaussian (quadratic) fermion operator, initialized to zero. Sites are
labeled `1:N` (integer form) or by the given `labels` (e.g. [`Up`](@ref)/[`Dn`](@ref)
labels for spinful systems).

Build up the operator with [`add_hop`](@ref), [`add_cdag_c`](@ref),
and [`add_c_cdag`](@ref). Operators support scalar multiplication (`*`) and
addition (`+`).

# Examples
```julia
import GaussianFermions as gf

# Spinless 4-site chain
H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end

# Spinful system
ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
for j in 1:3
    H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1.0)
    H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1.0)
end

# 2D system with tuple labels
Lx, Ly = 3, 3
labels = [(x, y) for x in 1:Lx for y in 1:Ly]
H = gf.GaussianOperator(labels)
for x in 1:Lx, y in 1:Ly
    x < Lx && (H = gf.add_hop(H, (x, y), (x + 1, y), -1.0))
    y < Ly && (H = gf.add_hop(H, (x, y), (x, y + 1), -1.0))
end
```
"""
struct GaussianOperator
    matrix_elems::NamedArray
    function GaussianOperator(A::NamedArray)
        return new(A)
    end
end

function GaussianOperator(N::Integer)
    return GaussianOperator(NamedArray(zeros(N, N), (1:N, 1:N), ("Labels", "Labels")))
end

function GaussianOperator(labels)
    N = length(labels)
    return GaussianOperator(NamedArray(zeros(N, N), (labels, labels), ("Labels", "Labels")))
end

Base.copy(G::GaussianOperator) = GaussianOperator(copy(G.matrix_elems))
Base.length(G::GaussianOperator) = size(G.matrix_elems, 1)

"""
    matrix_elements(G::GaussianOperator)
    matrix_elements(G::GaussianOperator, r, c)

Return the matrix elements of `G` as a `NamedArray`, or the element at row `r`
and column `c`. The inputs `r` and `c` can also be ranges or collections of labels.
"""
matrix_elements(G::GaussianOperator, r, c) = G.matrix_elems[r, c]

matrix_elements(G::GaussianOperator) = G.matrix_elems

"""
    labels(G::GaussianOperator)

Return the labels (e.g. site indices or spin-site indices) of the system on which
the operator `G` acts.
"""
labels(G::GaussianOperator) = names(G.matrix_elems, 1)

function (x::Number * G::GaussianOperator)
    return GaussianOperator(x * matrix_elements(G))
end

function (G::GaussianOperator * x::Number)
    return GaussianOperator(x * matrix_elements(G))
end

function (A::GaussianOperator + B::GaussianOperator)
    return GaussianOperator(matrix_elements(A) + matrix_elements(B))
end

"""
    add_cdag_c(G::GaussianOperator, i, j, coef=1.0)

Return a new operator equal to `G` plus `coef * c†_i c_j`.

# Example
```julia
import GaussianFermions as gf

O = gf.GaussianOperator(4)
O = gf.add_cdag_c(O, 1, 2)  # adds c†₁ c₂
```
"""
function add_cdag_c(G::GaussianOperator, i, j, coef::Number = 1.0)
    G = copy(G)
    G.matrix_elems[i, j] += coef
    return G
end

"""
    add_c_cdag(G::GaussianOperator, i, j, coef=1.0)

Return a new operator equal to `G` plus `coef * c_i c†_j`.
Since `c_i c†_j = δ_{ij} - c†_j c_i`, this adds `coef` to the `(j, i)` matrix element.

# Example
```julia
import GaussianFermions as gf

O = gf.GaussianOperator(4)
O = gf.add_c_cdag(O, 1, 2)  # adds c₁ c†₂
```
"""
function add_c_cdag(G::GaussianOperator, i, j, coef::Number = 1.0)
    G = copy(G)
    G.matrix_elems[j, i] += coef
    return G
end

"""
    add_hop(G::GaussianOperator, i, j, coef)

Return a new operator equal to `G` plus a hopping term `coef * (c†_i c_j + c_i c†_j)`.
This is a shorthand for calling both [`add_cdag_c`](@ref) and [`add_c_cdag`](@ref).

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
```
"""
function add_hop(G::GaussianOperator, i, j, coef::Number)
    G = add_cdag_c(G, i, j, coef)
    G = add_c_cdag(G, i, j, coef)
    return G
end

"""
    energies_states(G::GaussianOperator)

Diagonalize the operator `G`, returning a tuple `(ϵ, ϕ)` of eigenvalues and
eigenstates (as a `NamedArray` of column vectors).
"""
function energies_states(G)
    ϵ, ϕ = la.eigen(G.matrix_elems)
    N = length(ϵ)
    ϕ = NamedArray(ϕ, (labels(G), 1:N), ("Labels", "Eigenstates"))
    return ϵ, ϕ
end

"""
    expect(G::GaussianOperator, ψ::GaussianState)

Compute the expectation value ``\\langle ψ | G | ψ \\rangle`` of the operator `G`
in the Gaussian state `ψ`.

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
E0, ϕ = gf.ground_state(H; Nf=2)
gf.expect(H, ϕ) ≈ E0  # true
```
"""
function expect(G::GaussianOperator, ψ::GaussianState)
    return la.tr(matrix_elements(G) * correlation_matrix(ψ))
end

"""
    greens_function(H::GaussianOperator, times; labels = labels(H))

Compute the Greens function ``g(t) = -i e^{-i h t}`` from a GaussianOperator with
hopping matrix h. For positive time values this is identical to the 
retarded Greens function ``G^R(t)``.
Output is a Nt x Nv x Nv complex-valued tensor Gᴿ[t,l1,l2] where the first index 
runs over the time points, and the second two indices run over mode labels.
Optionally passing a subset of mode labels computes ``g(t)`` only on these
labels.
"""
function greens_function(H::GaussianOperator, times; labels = labels(H))
    ϵ, ϕ = energies_states(H)
    G = zeros(ComplexF64, length(times), length(labels), length(labels))
    for j in 1:length(times)
        exp_itϵ = [exp(-im * times[j] * ϵ[n]) for n in 1:length(ϵ)]
        G[j, :, :] = -im * ϕ[labels, :] * la.Diagonal(exp_itϵ) * (ϕ[labels, :])'
    end
    return NamedArray(G,(1:length(times), labels, labels), ("Time Index", "Labels", "Labels"))
end

"""
    lesser_greens_function(H::GaussianOperator, times; labels = labels(H))

Compute the lesser Green's function G<(t) = i⟨c†(0)c(t)⟩ from a GaussianOperator
with hopping matrix h, evaluated in the ground state.
Output is a Nt x Nv x Nv complex-valued tensor G<[t,l1,l2] where the first index 
runs over the time points, and the second two indices run over mode labels.
Optionally passing a subset of mode labels computes G^<(t) only on these labels.
"""
function lesser_greens_function(H::GaussianOperator, times; labels = labels(H))
    ϵ, ϕ = energies_states(H)
    occupancies = ground_state_occupancies(ϵ)
    GL = zeros(ComplexF64, length(times), length(labels), length(labels))
    for j in 1:length(times)
        exp_itϵ = [occupancies[n] * exp(-im * times[j] * ϵ[n]) for n in 1:length(ϵ)]
        GL[j, :, :] = im * ϕ[labels, :] * la.Diagonal(exp_itϵ) * (ϕ[labels, :])'
    end
    return NamedArray(GL,(1:length(times), labels, labels), ("Time Index", "Labels", "Labels"))
end

"""
    greater_greens_function(H::GaussianOperator, times; labels = labels(H))

Compute the greater Green's function G>(t) = -i⟨c(t)c†(0)⟩ from a GaussianOperator
with hopping matrix h, evaluated in the ground state.
Output is a Nt x Nv x Nv complex-valued tensor G>[t,l1,l2] where the first index 
runs over the time points, and the second two indices run over mode labels.
Optionally passing a subset of mode labels computes G^>(t) only on these labels.
"""
function greater_greens_function(H::GaussianOperator, times; labels = labels(H))
    ϵ, ϕ = energies_states(H)
    occupancies = ground_state_occupancies(ϵ)
    GG = zeros(ComplexF64, length(times), length(labels), length(labels))
    for j in 1:length(times)
        exp_itϵ = [(1 - occupancies[n]) * exp(-im * times[j] * ϵ[n]) for n in 1:length(ϵ)]
        GG[j, :, :] = -im * ϕ[labels, :] * la.Diagonal(exp_itϵ) * (ϕ[labels, :])'
    end
    return NamedArray(GG,(1:length(times), labels, labels), ("Time Index", "Labels", "Labels"))
end

greens_function(H, t::Number; kws...) = greens_function(H, [t]; kws...)
lesser_greens_function(H, t::Number; kws...) = lesser_greens_function(H, [t]; kws...)
greater_greens_function(H, t::Number; kws...) = greater_greens_function(H, [t]; kws...)

"""
    time_evolve(H::GaussianOperator, t::Number, ψ::GaussianState)

Evolve the Gaussian state `ψ` forward by time `t` under Hamiltonian `H`
(Schrödinger picture). The time `t` can be of arbitrary size and can be 
real or complex. Returns a new [`GaussianState`](@ref).

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ψ = gf.ground_state(H; Nf=2)
ψt = gf.time_evolve(H, 0.1, ψ)
```
"""
function time_evolve(H::GaussianOperator, t::Number, ψ::GaussianState)
    expHt = im * greens_function(H, t)
    orbs_t = expHt * orbitals(ψ)
    return GaussianState(orbs_t, occupancy(ψ))
end

"""
    time_evolve(H::GaussianOperator, t::Number, O::GaussianOperator)

Evolve the operator `O` forward by time `t` under Hamiltonian `H`
(Heisenberg picture). The time `t` can be of arbitrary size and can be
real or complex. Returns a new [`GaussianOperator`](@ref).

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
A = gf.add_cdag_c(gf.GaussianOperator(4), 1, 2)
At = gf.time_evolve(H, 0.1, A)
```
"""
function time_evolve(H::GaussianOperator, t::Number, O::GaussianOperator)
    expHt = im * greens_function(H, t)
    # TODO: check conjugation convention here:
    matrix_elems_t = expHt' * matrix_elements(O) * expHt
    return GaussianOperator(matrix_elems_t)
end
