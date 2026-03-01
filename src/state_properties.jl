"""
    correlation_matrix(ϕ::GaussianState; labels=labels(ϕ))

Compute the single-particle correlation matrix ``C_{ij} = \\langle c^\\dagger_i c_j \\rangle``
for the state `ϕ`, given by ``C_{ij} = \\sum_n \\bar{d}^n_i \\, \\eta_n \\, d^j_n``
where ``d^j_n`` are the orbitals and ``\\eta_n`` the occupancy values.

For a pure state (all ``\\eta_n \\in \\{0,1\\}``), the correlation matrix is a
projector: ``C^2 = C``.

If `labels` is given, only the submatrix for those labels is returned.

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf=2)
C = gf.correlation_matrix(ϕ)
```
"""
function correlation_matrix(ϕ::GaussianState; labels = labels(ϕ))
    orbs = orbitals(ϕ)[labels, :]
    return orbs * la.Diagonal(occupancy(ϕ)) * orbs'
end

"""
    density(ϕ::GaussianState; labels=labels(ϕ))

Return a vector of site occupation numbers ``\\langle n_i \\rangle`` (the diagonal
of the correlation matrix).

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf=2)
gf.density(ϕ)
```
"""
function density(ϕ::GaussianState; kws...)
    return la.diag(correlation_matrix(ϕ; kws...))
end

"""
    entanglement(ϕ::GaussianState; labels)

Compute the von Neumann entanglement entropy of the subsystem defined by `labels`.

The entropy is obtained from the eigenvalues ``\\nu_k`` of the reduced correlation
matrix ``C_{AA}`` (the block of the correlation matrix restricted to `labels`):

```math
S = -\\sum_k \\bigl[\\nu_k \\ln \\nu_k + (1 - \\nu_k) \\ln(1 - \\nu_k)\\bigr]
```

Eigenvalues near 0 or 1 correspond to orbitals fully empty or full within the
subsystem and do not contribute to entanglement.

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(10)
for j in 1:9
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf=5)
gf.entanglement(ϕ; labels=1:5)
```
"""
function entanglement(ϕ::GaussianState; labels)
    C = correlation_matrix(ϕ; labels)
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


"""
    bond_dimension(ϕ::GaussianState, range, cutoff::Real)

Estimate the MPS bond dimension needed to represent the state `ϕ` bipartitioned
at `range` (the labels on one side of the cut) to accuracy `cutoff`.

# Example
```julia
import GaussianFermions as gf

H = gf.GaussianOperator(10)
for j in 1:9
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf=5)
gf.bond_dimension(ϕ, 1:5, 1e-7)
```
"""
function bond_dimension(ϕ::GaussianState, range, cutoff::Real)
    C = correlation_matrix(ϕ; labels = range)
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
