using Test
using ITensors
using ITensorMPS
import GaussianFermions as gf
using GaussianFermions: Up, Dn

@testset "MPS vs GaussianFermions: spinless fermion chain" begin
    outputlevel = 0
    N = 20
    Nf = N ÷ 2
    t = 1.0

    #
    # GaussianFermions: ground state energy and entanglement
    #
    H_gf = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H_gf = gf.add_hop(H_gf, j, j + 1, -t)
    end
    E0_gf, ϕ0 = gf.ground_state(H_gf; Nf)

    S_gf = [gf.entanglement(ϕ0; sites = 1:b) for b in 1:(N - 1)]

    #
    # ITensorMPS / DMRG: ground state energy and entanglement
    #
    sites = siteinds("Fermion", N; conserve_qns = true)

    os = OpSum()
    for j in 1:(N - 1)
        os += -t, "Cdag", j, "C", j + 1
        os += -t, "Cdag", j + 1, "C", j
    end
    H_mps = MPO(os, sites)

    state = [n <= Nf ? "Occ" : "Emp" for n in 1:N]
    psi0 = MPS(sites, state)

    nsweeps = 10
    maxdim = [10, 20, 40, 80, 100]
    cutoff = 1.0e-12

    (outputlevel > 1) && println("Running DMRG for fermion chain")
    E0_mps, psi = dmrg(H_mps, psi0; nsweeps, maxdim, cutoff, outputlevel)
    (outputlevel > 1) && println("Done running DMRG for fermion chain")

    # Compute entanglement entropy at each bond from MPS
    S_mps = Float64[]
    for b in 1:(N - 1)
        orthogonalize!(psi, b)
        linds = [siteind(psi, b)]
        if b > 1
            push!(linds, linkind(psi, b - 1))
        end
        _, S, _ = svd(psi[b], linds)
        SvN = 0.0
        for n in 1:dim(S, 1)
            p = S[n, n]^2
            if p > 1.0e-14
                SvN -= p * log(p)
            end
        end
        push!(S_mps, SvN)
    end

    #
    # Density on each site
    #
    n_gf = gf.density(ϕ0)
    n_mps = expect(psi, "N")

    #
    # Correlation matrix <c†_i c_j>
    #
    C_gf = gf.correlation_matrix(ϕ0)
    C_mps = correlation_matrix(psi, "Cdag", "C")

    #
    # Compare results
    #
    @testset "Ground state energy" begin
        @test E0_gf ≈ E0_mps atol = 1.0e-8
    end

    @testset "Bond entanglement entropies" begin
        for b in 1:(N - 1)
            @test S_gf[b] ≈ S_mps[b] atol = 1.0e-6
        end
    end

    @testset "Site densities" begin
        for j in 1:N
            @test n_gf[j] ≈ n_mps[j] atol = 1.0e-8
        end
    end

    @testset "Correlation matrix" begin
        for i in 1:N, j in 1:N
            @test C_gf[i, j] ≈ C_mps[i, j] atol = 1.0e-6
        end
    end
end

@testset "MPS vs GaussianFermions: electron chain" begin
    outputlevel = 0
    N = 10
    Nup = N ÷ 2
    Ndn = N ÷ 2
    Nf = Nup + Ndn
    t = 1.0

    #
    # GaussianFermions: electron chain using Up/Dn spin wrapper types
    #
    ups = [Up(j) for j in 1:N]
    dns = [Dn(j) for j in 1:N]
    verts = vcat(ups, dns)
    H_gf = gf.GaussianOperator(verts)
    for j in 1:(N - 1)
        H_gf = gf.add_hop(H_gf, Up(j), Up(j + 1), -t)
        H_gf = gf.add_hop(H_gf, Dn(j), Dn(j + 1), -t)
    end
    E0_gf, ϕ0 = gf.ground_state(H_gf; Nf)

    # Entanglement at bond b: include both spins for sites 1:b
    # Vertex ordering is [up1,...,upN, dn1,...,dnN]
    S_gf = [gf.entanglement(ϕ0; sites = [1:b; (N + 1):(N + b)]) for b in 1:(N - 1)]

    # Densities per spin
    n_all = gf.density(ϕ0)
    nup_gf = n_all[1:N]
    ndn_gf = n_all[(N + 1):(2N)]

    # Correlation matrices per spin
    C_full = gf.correlation_matrix(ϕ0)
    Cup_gf = C_full[1:N, 1:N]
    Cdn_gf = C_full[(N + 1):(2N), (N + 1):(2N)]

    #
    # ITensorMPS / DMRG: Electron sites
    #
    sites = siteinds("Electron", N; conserve_qns = true)

    os = OpSum()
    for j in 1:(N - 1)
        os += -t, "Cdagup", j, "Cup", j + 1
        os += -t, "Cdagup", j + 1, "Cup", j
        os += -t, "Cdagdn", j, "Cdn", j + 1
        os += -t, "Cdagdn", j + 1, "Cdn", j
    end
    H_mps = MPO(os, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = MPS(sites, state)

    nsweeps = 6
    maxdim = [20, 50, 100, 200, 400]
    cutoff = 1.0e-12

    (outputlevel > 1) && println("Running DMRG for electron chain")
    E0_mps, psi = dmrg(H_mps, psi0; nsweeps, maxdim, cutoff, outputlevel)
    (outputlevel > 1) && println("Done running DMRG for electron chain")

    # Entanglement entropy at each bond from MPS
    S_mps = Float64[]
    for b in 1:(N - 1)
        orthogonalize!(psi, b)
        linds = [siteind(psi, b)]
        if b > 1
            push!(linds, linkind(psi, b - 1))
        end
        _, S, _ = svd(psi[b], linds)
        SvN = 0.0
        for n in 1:dim(S, 1)
            p = S[n, n]^2
            if p > 1.0e-14
                SvN -= p * log(p)
            end
        end
        push!(S_mps, SvN)
    end

    nup_mps = expect(psi, "Nup")
    ndn_mps = expect(psi, "Ndn")

    Cup_mps = correlation_matrix(psi, "Cdagup", "Cup")
    Cdn_mps = correlation_matrix(psi, "Cdagdn", "Cdn")

    #
    # Compare results
    #
    @testset "Ground state energy" begin
        @test E0_gf ≈ E0_mps atol = 1.0e-8
    end

    @testset "Bond entanglement entropies" begin
        for b in 1:(N - 1)
            @test S_gf[b] ≈ S_mps[b] atol = 1.0e-6
        end
    end

    @testset "Site densities" begin
        for j in 1:N
            @test nup_gf[j] ≈ nup_mps[j] atol = 1.0e-8
            @test ndn_gf[j] ≈ ndn_mps[j] atol = 1.0e-8
        end
    end

    @testset "Correlation matrices" begin
        for i in 1:N, j in 1:N
            @test Cup_gf[i, j] ≈ Cup_mps[i, j] atol = 1.0e-6
            @test Cdn_gf[i, j] ≈ Cdn_mps[i, j] atol = 1.0e-6
        end
    end
end

@testset "MPS vs GaussianFermions: 2D spinless fermion ladder" begin
    outputlevel = 0
    Lx = 10
    Ly = 4
    N = Lx * Ly
    Nf = N ÷ 2
    t = 1.0
    V = 2.0  # staggered on-site potential to open a gap

    # Use ITensorMPS square_lattice to define bonds (periodic in y)
    lattice = square_lattice(Lx, Ly; yperiodic = true)

    #
    # GaussianFermions: 2D ladder with staggered potential
    # Site numbering: n = (x-1)*Ly + y, matching square_lattice convention
    #
    H_gf = gf.GaussianOperator(N)
    for b in lattice
        H_gf = gf.add_hop(H_gf, b.s1, b.s2, -t)
    end
    for n in 1:N
        x = (n - 1) ÷ Ly + 1
        y = (n - 1) % Ly + 1
        H_gf = gf.add_cdag_c(H_gf, n, n, V * (-1)^(x + y))
    end
    E0_gf, ϕ0 = gf.ground_state(H_gf; Nf)

    #
    # ITensorMPS / DMRG: 2D ladder
    #
    sites = siteinds("Fermion", N; conserve_qns = true)

    os = OpSum()
    for b in lattice
        os += -t, "Cdag", b.s1, "C", b.s2
        os += -t, "Cdag", b.s2, "C", b.s1
    end
    for n in 1:N
        x = (n - 1) ÷ Ly + 1
        y = (n - 1) % Ly + 1
        os += V * (-1)^(x + y), "N", n
    end
    H_mps = MPO(os, sites)

    # Initial state: occupy low-energy sites of staggered potential
    state = String[]
    for n in 1:N
        x = (n - 1) ÷ Ly + 1
        y = (n - 1) % Ly + 1
        push!(state, isodd(x + y) ? "Occ" : "Emp")
    end
    psi0 = MPS(sites, state)

    nsweeps = 10
    maxdim = [10, 20, 40, 80, 160, 200]
    cutoff = 1.0e-12

    (outputlevel > 1) && println("Running DMRG for 2D system")
    E0_mps, psi = dmrg(H_mps, psi0; nsweeps, maxdim, cutoff, outputlevel)
    (outputlevel > 1) && println("Done running DMRG for 2D system")

    #
    # Density on each site
    #
    n_gf = gf.density(ϕ0)
    n_mps = expect(psi, "N")

    #
    # Correlation matrix <c†_i c_j>
    #
    C_gf = gf.correlation_matrix(ϕ0)
    C_mps = correlation_matrix(psi, "Cdag", "C")

    #
    # Compare results
    #
    @testset "Ground state energy" begin
        @test E0_gf ≈ E0_mps atol = 1.0e-6
    end

    @testset "Site densities" begin
        for j in 1:N
            @test n_gf[j] ≈ n_mps[j] atol = 1.0e-6
        end
    end

    @testset "Correlation matrix" begin
        for i in 1:N, j in 1:N
            @test C_gf[i, j] ≈ C_mps[i, j] atol = 1.0e-4
        end
    end
end
