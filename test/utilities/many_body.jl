import LinearAlgebra as la

# Helper functions for exact many-body calculations on small systems.
# Used to validate GaussianFermions results by brute-force comparison.
#
# Conventions:
#  - Basis states s are 0-indexed integers; bit (i-1) of s is occupation of site i.
#  - Sites are 1-indexed.
#  - Fermionic operators obey Jordan-Wigner conventions.

"""
    occupation(s::Int, i::Int) -> Int

Occupation (0 or 1) of site i in basis state s. Sites are 1-indexed.
"""
occupation(s::Int, i::Int) = (s >> (i - 1)) & 1

"""
    parity_below(s::Int, p::Int) -> Int

Number of 1-bits at positions 0, 1, ..., p-1 in s. Used for Jordan-Wigner signs.
"""
parity_below(s::Int, p::Int) = count_ones(s & ((1 << p) - 1))

"""
    apply_cdag_c(s::Int, i::Int, j::Int) -> (Int, Int)

Compute c†_i c_j |s⟩. Returns (s_new, sign) where s_new is the resulting basis
state and sign ∈ {-1, +1}. Returns (0, 0) if the action is zero (forbidden
occupation).

Handles i == j correctly (gives n_i: returns (s, +1) when site i is occupied).
"""
function apply_cdag_c(s::Int, i::Int, j::Int)
    jbit = j - 1
    ibit = i - 1
    # c_j requires site j occupied
    occupation(s, j) == 0 && return (0, 0)
    sign1 = (-1)^parity_below(s, jbit)
    s_mid = s ⊻ (1 << jbit)          # remove particle from j
    # c†_i requires site i empty in s_mid
    (s_mid >> ibit) & 1 == 1 && return (0, 0)
    sign2 = (-1)^parity_below(s_mid, ibit)
    s_new = s_mid | (1 << ibit)       # add particle to i
    return (s_new, sign1 * sign2)
end

"""
    sector_states(N::Int, Nf::Int) -> Vector{Int}

All 0-indexed basis states with exactly Nf particles on N sites.
"""
sector_states(N::Int, Nf::Int) = [s for s in 0:(2^N - 1) if count_ones(s) == Nf]

"""
    build_mb_hamiltonian(N::Int, Nf::Int; hopping=-1.0) -> (Matrix{Float64}, Vector{Int})

Many-body Hamiltonian matrix for a nearest-neighbor spinless fermion chain with N
sites and open boundaries, restricted to the Nf-particle sector. Returns
(H_sector, sector) where sector is the list of basis state indices.
"""
function build_mb_hamiltonian(N::Int, Nf::Int; hopping::Float64 = -1.0)
    sector = sector_states(N, Nf)
    dim = length(sector)
    state_to_idx = Dict(s => idx for (idx, s) in enumerate(sector))
    H = zeros(Float64, dim, dim)
    for (col_idx, s) in enumerate(sector)
        for j in 1:(N - 1)
            for (i_site, j_site) in ((j + 1, j), (j, j + 1))
                s_new, sgn = apply_cdag_c(s, i_site, j_site)
                sgn == 0 && continue
                H[state_to_idx[s_new], col_idx] += hopping * sgn
            end
        end
    end
    return H, sector
end

"""
    build_mb_hamiltonian_from_matrix(N::Int, Nf::Int, h::AbstractMatrix)
        -> (Matrix{ComplexF64}, Vector{Int})

Many-body Hamiltonian matrix from a single-particle hopping matrix h (N×N),
where H = Σ_{ij} h[i,j] c†_i c_j, restricted to the Nf-particle sector.
"""
function build_mb_hamiltonian_from_matrix(N::Int, Nf::Int, h::AbstractMatrix)
    sector = sector_states(N, Nf)
    dim = length(sector)
    state_to_idx = Dict(s => idx for (idx, s) in enumerate(sector))
    H = zeros(ComplexF64, dim, dim)
    for (col_idx, s) in enumerate(sector)
        for j in 1:N, i in 1:N
            h_ij = h[i, j]
            iszero(h_ij) && continue
            s_new, sgn = apply_cdag_c(s, i, j)
            sgn == 0 && continue
            H[state_to_idx[s_new], col_idx] += h_ij * sgn
        end
    end
    return H, sector
end

"""
    exact_ground_state(H_sector::AbstractMatrix) -> (Float64, Vector{ComplexF64})

Lowest eigenvalue and eigenvector of H_sector.
"""
function exact_ground_state(H_sector::AbstractMatrix)
    vals, vecs = la.eigen(H_sector)
    idx = argmin(real(vals))
    return real(vals[idx]), vecs[:, idx]
end

"""
    exact_density(psi, sector, N) -> Vector{Float64}

Site occupations ⟨n_i⟩ = Σ_s |ψ_s|² n_i(s) for state psi in the given sector.
"""
function exact_density(psi::AbstractVector, sector::Vector{Int}, N::Int)
    n = zeros(Float64, N)
    for (idx, s) in enumerate(sector)
        p = abs2(psi[idx])
        iszero(p) && continue
        for i in 1:N
            n[i] += p * occupation(s, i)
        end
    end
    return n
end

"""
    exact_correlation_matrix(psi, sector, N) -> Matrix{ComplexF64}

Single-particle correlation matrix `C[i,j] = ⟨c†_i c_j⟩` for state psi in
the given sector. This matches the GaussianFermions convention:
`C[i,j] = Σ_n conj(Φ[i,n]) η_n Φ[j,n]`.
"""
function exact_correlation_matrix(psi::AbstractVector, sector::Vector{Int}, N::Int)
    state_to_idx = Dict(s => idx for (idx, s) in enumerate(sector))
    C = zeros(ComplexF64, N, N)
    for (col_idx, s) in enumerate(sector)
        iszero(psi[col_idx]) && continue
        for i in 1:N, j in 1:N
            s_new, sgn = apply_cdag_c(s, i, j)   # c†_i c_j |s⟩
            sgn == 0 && continue
            haskey(state_to_idx, s_new) || continue
            C[i, j] += conj(psi[state_to_idx[s_new]]) * psi[col_idx] * sgn
        end
    end
    return C
end

"""
    time_evolve_exact(H_sector, t, psi) -> Vector{ComplexF64}

Evolve psi by time t under H_sector: returns exp(-i t H) psi.
"""
function time_evolve_exact(H_sector::AbstractMatrix, t::Number, psi::AbstractVector)
    return exp(-im * t * H_sector) * psi
end
