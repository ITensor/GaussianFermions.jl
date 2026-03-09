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
    #function CreationOperator(A::NamedArray)
    #    return new(A)
    #end
end

CreationOperator(labels, orbital::Vector) = CreationOperator(NamedArray(orbital,(labels,),("Labels",)))

struct AnnihilationOperator
    orbital::NamedArray
    #function AnnihilationOperator(A::NamedArray)
    #    return new(A)
    #end
end

AnnihilationOperator(labels, orbital::Vector) = AnnihilationOperator(NamedArray(orbital,(labels,),("Labels",)))

function apply(Cdag::CreationOperator, ψ::GaussianState)
    v = Vector(Cdag.orbital)
    C = Matrix(correlation_matrix(ψ))
    v0 = v - C*v
    nrm0 = norm(v0)
    trace = nrm0^2
    v0 /= nrm0
    if trace < 1E-12
        error(@sprintf("Nearly zero state in creation apply, trace = %.4E\n",trace))
    end
    # TODO: I think "Using left names" warning is coming from v0*v0' operation...
    Cv = C + v0*v0'
    f, ϕ = la.eigen(Cv)
    ϕ_labeled = NamedArray(ϕ,(labels(ψ), 1:length(f)),("Labels","N. Orbitals"))
    return GaussianState(ϕ_labeled, f, trace)
end

function apply(C::AnnihilationOperator, ψ::GaussianState)
    w = C.orbital
    C = correlation_matrix(ψ)
    w1 = C*w
    nrm1 = norm(w1)
    trace = nrm1^2
    if trace < 1E-12
        error(@sprintf("Nearly zero state in annihilation apply, trace = %.4E\n",trace))
    end
    w1 /= nrm1
    Cw = C - w1*w1'
    f, ϕ = la.eigen(Cw)
    ϕ_labeled = NamedArray(ϕ,(labels(ψ), 1:length(f)),("Labels","N. Orbitals"))
    return GaussianState(ϕ_labeled, f, trace)
end
