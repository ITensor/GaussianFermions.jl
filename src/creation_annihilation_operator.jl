using NamedArrays: NamedArray
using LinearAlgebra: norm
using Printf: @sprintf

function pause()
    print(stdout, "(Paused) ")
    c = read(stdin, 1)
    c == UInt8[0x71] && exit(0)
    return nothing
end

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

AnnihilationOperator(labels, orbital::Vector) = AnnihilationOperator(NamedArray(orbital,(labels,),("Labels",)))

function apply(Cdag::CreationOperator, ψ::GaussianState)
    # TODO:
    # interesting idea to pursure...
    # precompute trace by doing something like v'*C*v
    # should work for pure and mixed states
    v = Vector(Cdag.orbital)
    C = Matrix(correlation_matrix(ψ))
    v0 = v - C*v
    Cv = norm(v0)^2*C - v0*v0'
    trace = la.tr(Cv)
    if trace < 1E-12
        error(@sprintf("Nearly zero in apply, trace = %.4E\n",trace))
    end
    Cv /= trace
    f, ϕ = la.eigen(Cv)
    ϕ_labeled = NamedArray(ϕ,(labels(ψ), 1:length(f)),("Labels","N. Orbitals"))
    return GaussianState(ϕ_labeled, f, trace)
end

function apply(C::AnnihilationOperator, ψ::GaussianState)
    w = C.orbital
    C = correlation_matrix(ψ)
    Cw = norm(w)^2*C - w*w'
    trace = la.tr(Cw)
    Cw /= trace
    f, ϕ = la.eigen(Cw)
    return GaussianState(ϕ, f, trace)
end
