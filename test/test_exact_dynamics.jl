import GaussianFermions as gf
using LinearAlgebra: norm
using Test

include("utilities/hamiltonians.jl")
include("utilities/many_body.jl")

# Tests that compare GaussianFermions results against exact
# diagonalization on N=10 sites (Hilbert space dimension 2^10 = 1024).
# The many-body sector with Nf=5 particles has dimension C(10,5)=252.

@testset "Ground State vs Exact Diagonalization" begin
    N = 10
    Nf = 5

    H_gf = fermion_chain_h(N)
    E0_gf, ϕ0_gf = gf.ground_state(H_gf; Nf)

    H_mb, sector = build_mb_hamiltonian(N, Nf)
    E0_exact, psi0 = exact_ground_state(H_mb)

    # Ground state energy
    @test E0_gf ≈ E0_exact atol = 1.0e-10

    # Site densities
    n_gf = real(gf.density(ϕ0_gf))
    n_exact = exact_density(psi0, sector, N)
    @test n_gf ≈ n_exact atol = 1.0e-10

    # Full correlation matrix C[i,j] = ⟨c†_i c_j⟩
    C_gf = Array(gf.correlation_matrix(ϕ0_gf))
    C_exact = exact_correlation_matrix(psi0, sector, N)
    @test real(C_gf) ≈ real(C_exact) atol = 1.0e-10
    @test norm(imag(C_gf)) < 1.0e-10
    @test norm(imag(C_exact)) < 1.0e-10

    # Sanity: correlation matrix is a projector for a pure state (C² = C)
    @test real(C_gf)^2 ≈ real(C_gf) atol = 1.0e-10
end

@testset "Time Evolution of Ground State vs Exact" begin
    N = 10
    Nf = 5

    H_gf = fermion_chain_h(N)
    _, ϕ0_gf = gf.ground_state(H_gf; Nf)

    H_mb, sector = build_mb_hamiltonian(N, Nf)
    _, psi0 = exact_ground_state(H_mb)

    # Compare densities and correlation matrices at several time points
    for t in (0.5, 1.0, 2.0, 5.0)
        ϕt_gf = gf.time_evolve(H_gf, t, ϕ0_gf)
        psi_t = time_evolve_exact(H_mb, t, psi0)

        n_gf = real(gf.density(ϕt_gf))
        n_exact = exact_density(psi_t, sector, N)
        @test n_gf ≈ n_exact atol = 1.0e-8
    end

    # Full correlation matrix at one time point
    t = 1.5
    ϕt_gf = gf.time_evolve(H_gf, t, ϕ0_gf)
    psi_t = time_evolve_exact(H_mb, t, psi0)
    C_gf = Array(gf.correlation_matrix(ϕt_gf))
    C_exact = exact_correlation_matrix(psi_t, sector, N)
    @test C_gf ≈ C_exact atol = 1.0e-8
end

@testset "Quench Dynamics vs Exact" begin
    # Prepare ground state of a standard chain, then evolve under a
    # different Hamiltonian (quench: add on-site potential at the center site).
    N = 10
    Nf = 5
    center = N ÷ 2   # site 5

    H1_gf = fermion_chain_h(N)
    _, ϕ0_gf = gf.ground_state(H1_gf; Nf)

    # Quench Hamiltonian: chain + on-site energy μ=1 at site `center`
    H_quench_gf = H1_gf + ("Cdag", center, "C", center)

    # Extract the single-particle matrix and build the many-body version
    h_quench = real(Array(gf.matrix_elements(H_quench_gf)))
    H1_mb, sector = build_mb_hamiltonian(N, Nf)
    H_quench_mb, _ = build_mb_hamiltonian_from_matrix(N, Nf, h_quench)
    _, psi0 = exact_ground_state(H1_mb)

    for t in (0.5, 1.0, 2.0, 5.0)
        ϕt_gf = gf.time_evolve(H_quench_gf, t, ϕ0_gf)
        psi_t = time_evolve_exact(H_quench_mb, t, psi0)

        n_gf = real(gf.density(ϕt_gf))
        n_exact = exact_density(psi_t, sector, N)
        @test n_gf ≈ n_exact atol = 1.0e-8
    end

    # Correlation matrix at one time point
    t = 2.0
    ϕt_gf = gf.time_evolve(H_quench_gf, t, ϕ0_gf)
    psi_t = time_evolve_exact(H_quench_mb, t, psi0)
    C_gf = Array(gf.correlation_matrix(ϕt_gf))
    C_exact = exact_correlation_matrix(psi_t, sector, N)
    @test C_gf ≈ C_exact atol = 1.0e-8
end

@testset "Physical Consistency of Time Evolution" begin
    # Check properties that must hold for any correct time evolution:
    # (1) Particle number is conserved.
    # (2) The correlation matrix remains a projector: C(t)² ≈ C(t).
    # (3) Energy is conserved under evolution by H itself.
    N = 10
    Nf = 5

    H_gf = fermion_chain_h(N)
    _, ϕ0 = gf.ground_state(H_gf; Nf)
    E0 = gf.expect(H_gf, ϕ0)

    for t in (1.0, 3.0, 7.0)
        ϕt = gf.time_evolve(H_gf, t, ϕ0)

        # (1) Particle number
        @test sum(real(gf.density(ϕt))) ≈ Nf atol = 1.0e-10

        # (2) Projector property C² = C
        C = Array(gf.correlation_matrix(ϕt))
        @test C * C ≈ C atol = 1.0e-10

        # (3) Energy conservation
        @test real(gf.expect(H_gf, ϕt)) ≈ E0 atol = 1.0e-10
    end

    # Same checks for quench evolution (different Hamiltonian preserves Nf but not energy of H1)
    c = N ÷ 2
    H_quench = H_gf + ("Cdag", c, "C", c)
    E_quench0 = gf.expect(H_quench, ϕ0)
    @test E_quench0 > E0

    for t in (1.0, 3.0, 7.0)
        ϕt = gf.time_evolve(H_quench, t, ϕ0)

        # Particle number conserved
        @test sum(real(gf.density(ϕt))) ≈ Nf atol = 1.0e-10

        # Projector property
        C = Array(gf.correlation_matrix(ϕt))
        @test C * C ≈ C atol = 1.0e-10

        # Energy under H_quench is conserved
        @test real(gf.expect(H_quench, ϕt)) ≈ E_quench0 atol = 1.0e-10
    end
end
