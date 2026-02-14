import LinearAlgebra as la

"""
GaussianState

Represents a Gaussian state of a single "species"
(e.g. spinless fermions or fermions of the same spin label).
A pure state has a `filling` vector with entries fₙ=0,1.
A mixed state can have fractional fillings fₙ ∈ [0,1].

The single-particle density matrix or correlation matrix
for such a state is given by
C_ij = ∑ₙ ϕ_in f_n ϕ̄_jn 
"""
struct GaussianState <: AbstractGaussianState
    orbitals::Matrix
    filling::Vector
end

orbitals(ϕ::GaussianState) = ϕ.orbitals
filling(ϕ::GaussianState) = ϕ.filling
Base.length(ϕ::GaussianState) = size(orbitals(ϕ), 1)
Base.copy(ϕ::GaussianState) = GaussianState(orbitals(ϕ), filling(ϕ))

ispure(ϕ::GaussianState) = all(f -> (f == 1.0 || f == 0.0), filling(ϕ))

function correlation_matrix(ϕ::GaussianState; spatial_range = 1:length(ϕ))
    orbs = orbitals(ϕ)[spatial_range, :]
    # TODO: should orbs be complex conjugated here, so conj(orbs)*..*conj(orbs)' ?
    return orbs * la.Diagonal(filling(ϕ)) * orbs'
end

function density(ϕ::GaussianState; spatial_range = 1:length(ϕ))
    return la.diag(correlation_matrix(ϕ))
end

function entanglement(ϕ::GaussianState, range)
    C = correlation_matrix(ϕ; range)
    occs, _ = la.eigen(C)
    Svn = 0.0
    @assert la.norm(imag(occs)) < 1.0e-8
    for ν in real(occs)
        if real(ν) > 0.0
            Svn += -ν * log(ν)
        end
        if real(ν) < 1.0
            Svn += -(1 - ν) * log(1 - ν)
        end
    end
    return Svn
end

inactivity(ν) = abs(2ν - 1)

function bond_dimension(ϕ::GaussianState, range, cutoff::Real)
    C = correlation_matrix(ϕ; range)
    occs, _ = la.eigen(C)
    occs = real(occs)

    n = length(occs)
    inactivities = sort(inactivity.(occs); rev = true)

    # Decimate spectrum by "freezing" inactive modes
    # Each decimation doubles number of discarded eigenvalues
    # (ndiscard is number of discarded modes)
    fidelity = 1.0
    ndisc_modes = 0
    while fidelity * inactivities[ndisc_modes + 1] + cutoff > 1.0
        fidelity *= inactivities[ndisc_modes + 1]
        ndisc_modes += 1
    end

    nactive_modes = n - ndisc_modes
    inactivities = inactivities[(ndisc_modes + 1):n]

    # Explicitly compute remaining eigenvalues
    eigvals = zeros(2^nactive_modes)
    for (w, inds) in enumerate(Iterators.product(fill(0:1, nactive_modes)...))
        eigvals[w] = fidelity # account for modes already discarded
        for j in 1:nactive_modes
            ν, s = occs[j], inds[j]
            eigvals[w] *= ((1 - s) * ν + s * (1 - ν))
        end
    end
    eigvals = sort(eigvals)

    # Discard more eigvalues until infidelity exceeds cutoff
    ndisc_evals = 0
    while fidelity + cutoff - eigvals[ndisc_evals + 1] > 1.0
        fidelity -= eigvals[ndisc_evals + 1]
        ndisc_evals += 1
    end

    return 2^nactive_modes - ndisc_evals
end

# TODO: this method is buggy. It doesn't change the norm
#       of the state even when `orb` is not orthogonal to the filled orbitals
function add_orbital(ψ::GaussianState, orb::Vector)
    ispure(ψ) || error("add_orbital currently defined for pure states only")
    abs(la.norm(orb) - 1) < 1.0e-9 ||
        error("expected orbital `orb` to be normalized in `add_orbital`")
    ϕ, fill = copy(orbitals(ψ)), copy(filling(ψ))
    N, Np = length(ψ), count(>(0), fill)
    ϕ_augmented = zeros(N, N)
    ϕ_augmented[:, 1:Np] = ϕ[:, 1:Np]
    ϕ_augmented[:, Np + 1] = orb
    ϕ_augmented[:, (Np + 2):end] = ϕ[:, (Np + 1):(end - 1)]
    #ϕ_augmented = hcat(ϕ[:,1:Np],orb,ϕ[:,Np+1:end])
    display(ϕ_augmented)
    q, _ = la.qr(ϕ_augmented)
    ϕ = Matrix(q)
    fill[Np + 1] = 1.0
    return GaussianState(ϕ, fill)
end
