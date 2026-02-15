import LinearAlgebra as la
import Base: *, +
using NamedArrays: NamedArray

struct GaussianOperator
    matrix_elems::NamedArray
    function GaussianOperator(A::NamedArray)
        new(A)
    end
end

function GaussianOperator(N::Integer)
    return GaussianOperator(NamedArray(zeros(N, N),(1:N,1:N),("Vertices","Vertices")))
end

function GaussianOperator(vertices)
    N = length(vertices)
    return GaussianOperator(NamedArray(zeros(N, N), (vertices, vertices),("Vertices","Vertices")))
end

Base.copy(G::GaussianOperator) = GaussianOperator(copy(G.matrix_elems))
Base.length(G::GaussianOperator) = size(G.matrix_elems, 1)

matrix_elements(G::GaussianOperator,r,c) = G.matrix_elems[r,c]

matrix_elements(G::GaussianOperator) = G.matrix_elems

vertices(G::GaussianOperator) = names(G.matrix_elems, 1)

function (x::Number * G::GaussianOperator)
    return GaussianOperator(x * matrix_elements(G))
end

function (G::GaussianOperator * x::Number)
    return GaussianOperator(x * matrix_elements(G))
end

function (A::GaussianOperator + B::GaussianOperator)
    return GaussianOperator(matrix_elements(A) + matrix_elements(B))
end

function add_cdag_c(G::GaussianOperator, i, j, coef::Number = 1.0)
    G = copy(G)
    G.matrix_elems[i, j] += coef
    return G
end

function add_c_cdag(G::GaussianOperator, i, j, coef::Number = 1.0)
    G = copy(G)
    G.matrix_elems[j, i] += coef
    return G
end

function add_hop(G::GaussianOperator, i, j, coef::Number)
    G = add_cdag_c(G, i, j, coef)
    G = add_c_cdag(G, i, j, coef)
    return G
end

function energies_states(G) 
    ϵ, ϕ = la.eigen(G.matrix_elems)
    N = length(ϵ)
    ϕ = NamedArray(ϕ,(vertices(G),1:N),("Vertices","Eigenstates"))
    return ϵ, ϕ
end

function expect(G::GaussianOperator, ψ::GaussianState)
    return la.tr(matrix_elements(G) * correlation_matrix(ψ))
end

function greens_function(H::GaussianOperator, t::Number)
    ϵ, ϕ = energies_states(H)
    exp_itϵ = [exp(-im * t * ϵ[n]) for n in 1:length(ϵ)]
    return -im * ϕ * la.Diagonal(exp_itϵ) * ϕ'
end

function lesser_greens_function(H::GaussianOperator, t::Number)
    ϵ, ϕ = energies_states(H)
    exp_itϵ = [filling(H)[n] * exp(-im * t * ϵ[n]) for n in 1:length(ϵ)]
    return im * ϕ * la.Diagonal(exp_itϵ) * ϕ'
end

function greater_greens_function(H::GaussianOperator, t::Number)
    ϵ, ϕ = energies_states(H)
    exp_itϵ = [(1 - filling(H)[n]) * exp(-im * t * ϵ[n]) for n in 1:length(ϵ)]
    return -im * ϕ * la.Diagonal(exp_itϵ) * ϕ'
end

function time_evolve(H::GaussianOperator, t::Number, ψ::GaussianState)
    expHt = im * greens_function(H, t)
    orbs_t = expHt * orbitals(ψ)
    return GaussianState(orbs_t, filling(ψ))
end

function time_evolve(H::GaussianOperator, t::Number, O::GaussianOperator)
    expHt = im * greens_function(H, t)
    # TODO: check conjugation convention here:
    matrix_elems_t = expHt' * matrix_elements(O) * expHt
    return GaussianOperator(matrix_elems_t)
end
