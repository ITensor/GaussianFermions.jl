using NamedArrays: NamedArray
using LinearAlgebra: norm

struct CreationOperator
    orbital::NamedArray
    function CreationOperator(A::NamedArray)
        return new(A)
    end
end

CreationOperator(labels, orbital::Vector) = CreationOperator(NamedArray(orbital,(labels,),("Labels",)))

struct AnnihilationOperator
    orbital::NamedArray
    function AnnihilationOperator(A::NamedArray)
        return new(A)
    end
end

function apply(Cdag::CreationOperator, ψ::GaussianState)
    v = Cdag.orbital
    C = correlation_matrix(ψ)
    v0 = v - C*v
    println("v0 = "); display(v0)
    Cv = norm(v0)^2*C + v0*v0'
    println("Cv = "); display(Cv)
    f, ϕ = la.eigen(Cv)
    println("f = "); display(f)
    println("ϕ = "); display(ϕ)
end

function apply(C::AnnihilationOperator, ψ::GaussianState)
    w = C.orbital
    C = correlation_matrix(ψ)
    Cv = norm(v)^2*C + v*v'
end
