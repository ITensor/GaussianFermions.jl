import GaussianFermions as gf
import LinearAlgebra as la
using Test

include("utilities/hamiltonians.jl")
include("utilities/many_body.jl")

"""
    harmonic_trap_chain(N, μ)

Hopping chain in a harmonic trap `V(j) = (j - (N+1)/2)^2`, with the chemical potential
folded into the diagonal as `-μ`. Folding μ into the operator rather than into the
occupancies is exact: `N̂` is the identity in single-particle space, so `H - μN̂` has the
same eigenvectors as `H` and eigenvalues shifted by `-μ`.
"""
function harmonic_trap_chain(N, μ)
    H = fermion_chain_h(N)
    V(j) = (j - (N + 1) / 2)^2
    for j in 1:N
        H = gf.add_cdag_c(H, j, j, V(j) - μ)
    end
    return H
end

"""
    brute_force_thermal(N, h, β)

Grand-canonical `(density, energy)` for `H = Σ_ij h[i,j] c†_i c_j` by summing over every
state of the full `2^N`-dimensional Fock space. Independent of the Gaussian machinery --
it never assumes Wick's theorem -- so it tests the factorized form itself, not just the
occupancy formula.
"""
function brute_force_thermal(N, h, β)
    energies = Float64[]
    densities = Vector{Float64}[]
    for Nf in 0:N
        H_sector, sector = build_mb_hamiltonian_from_matrix(N, Nf, h)
        vals, vecs = la.eigen(la.Hermitian(H_sector))
        for k in eachindex(vals)
            push!(energies, real(vals[k]))
            push!(densities, exact_density(vecs[:, k], sector, N))
        end
    end
    # Shifted by the ground state energy so the weights cannot overflow at large β.
    weights = exp.(-β .* (energies .- minimum(energies)))
    Z = sum(weights)
    return sum(weights .* densities) / Z, sum(weights .* energies) / Z
end

@testset "Spinless Fermion Chain" begin
    N = 20
    β = 50
    μ = N^2 / 16

    H = harmonic_trap_chain(N, μ)
    ϕβ = gf.thermal_state(H, β)

    dens = gf.density(ϕβ)
    @test sum(dens) ≈ N / 2
end

@testset "High Temperature" begin
    N = 10

    # At β = 0 every orbital is half filled whatever the Hamiltonian, so the state is
    # maximally mixed and the density is 1/2 on every site. This is the case the
    # purification ancilla construction starts from.
    for H in (fermion_chain_h(N), harmonic_trap_chain(N, 3.0), electron_chain_h(N ÷ 2))
        ϕ0 = gf.thermal_state(H, 0.0)
        @test all(≈(1 / 2), gf.occupancy(ϕ0))
        @test all(≈(1 / 2), real(gf.density(ϕ0)))
        # ⟨H⟩ = tr(h)/2, and tr(h) is the sum of the single-particle energies. An absolute
        # tolerance because a bare hopping chain is traceless, so both sides are zero and a
        # relative comparison would be against numerical noise.
        ϵ, _ = gf.energies_states(H)
        @test isapprox(real(gf.expect(H, ϕ0)), sum(ϵ) / 2; atol = 1.0e-10)
    end

    # Leading high-temperature behaviour. Expanding the Fermi function,
    #   f(ϵ) = 1/2 - βϵ/4 + O(β³)
    # gives ⟨H⟩ = tr(h)/2 - β·tr(h²)/4 and ⟨N⟩ = N/2 - β·tr(h)/4, both with O(β³) error.
    # Test is of the scaling of the residual rather than its size.
    H = harmonic_trap_chain(N, 3.0)
    ϵ, _ = gf.energies_states(H)
    errors = map((0.02, 0.01)) do β
        ϕ = gf.thermal_state(H, β)
        predicted = sum(ϵ) / 2 - β * sum(abs2, ϵ) / 4
        return abs(real(gf.expect(H, ϕ)) - predicted)
    end
    @test errors[1] / errors[2] ≈ 8 rtol = 0.05
end

@testset "Low Temperature" begin
    N = 10
    β = 200.0

    # As β → ∞ the Fermi function becomes a step at zero, so the thermal state must
    # converge to the ground state that fills every negative-energy orbital -- which is
    # what `ground_state` returns when no `Nf` is given. This is the check that pins the
    # sign of the exponent: the thermal state of -H would fill the *positive* orbitals.
    H = harmonic_trap_chain(N, 3.0)
    E0, ϕ0 = gf.ground_state(H)
    ϕβ = gf.thermal_state(H, β)
    @test real(gf.density(ϕβ)) ≈ real(gf.density(ϕ0))
    @test real(gf.expect(H, ϕβ)) ≈ E0
    @test gf.occupancy(ϕβ) ≈ round.(gf.occupancy(ϕβ))   # every orbital is 0 or 1

    # The filling is set by how many orbitals lie below μ, so raising μ fills the trap.
    ϵ, _ = gf.energies_states(fermion_chain_h(N))
    counts = map((-1.0, 0.0, 1.0)) do μ
        H = fermion_chain_h(N)
        for j in 1:N
            H = gf.add_cdag_c(H, j, j, -μ)
        end
        n = sum(real(gf.density(gf.thermal_state(H, β))))
        @test n ≈ count(<(μ), ϵ)
        return n
    end
    @test issorted(counts)
end

@testset "Exact Many-Body Comparison" begin
    # Small enough to diagonalize every particle-number sector of the full Fock space.
    N = 5
    H = harmonic_trap_chain(N, 1.5)
    h = Matrix(gf.matrix_elements(H))

    for β in (0.5, 2.0, 8.0)
        ϕβ = gf.thermal_state(H, β)
        dens_exact, energy_exact = brute_force_thermal(N, h, β)
        @test real(gf.density(ϕβ)) ≈ dens_exact
        @test real(gf.expect(H, ϕβ)) ≈ energy_exact
    end
end
