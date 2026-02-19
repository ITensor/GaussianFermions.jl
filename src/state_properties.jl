function correlation_matrix(ϕ::GaussianState; sites = vertices(ϕ))
    orbs = orbitals(ϕ)[sites, :]
    return orbs * la.Diagonal(filling(ϕ)) * orbs'
end

function density(ϕ::GaussianState; kws...)
    return la.diag(correlation_matrix(ϕ; kws...))
end

function entanglement(ϕ::GaussianState; sites)
    C = correlation_matrix(ϕ; sites)
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


function bond_dimension(ϕ::GaussianState, range, cutoff::Real)
    C = correlation_matrix(ϕ; sites = range)
    occs, _ = la.eigen(C)
    occs = real(occs)

    inactivity(ν) = abs(2ν - 1)

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
