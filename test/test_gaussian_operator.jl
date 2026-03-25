import GaussianFermions as gf
using LinearAlgebra: Diagonal, norm
using NamedArrays: NamedArray
using Test

include("utilities/hamiltonians.jl")

@testset "GaussianOperator Constructors" begin
    # Construct with integer dimension
    N = 10
    H = gf.GaussianOperator(N)
    @test length(H) == N
    @test norm(gf.matrix_elements(H)) < 1.0e-12

    # Construct with array of vertices
    sites = [(1, 1), (1, 2), (2, 1), (2, 2)]
    H = gf.GaussianOperator(sites)
    @test length(H) == length(sites)
    @test norm(gf.matrix_elements(H)) < 1.0e-12
end

@testset "Add Hop Function" begin
    # Construct with integer dimension
    N = 4
    H = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H = gf.add_hop(H, j, j + 1, -1)
    end
    h = [
        0 -1 0 0;
        -1 0 -1 0;
        0 -1 0 -1;
        0 0 -1 0
    ]
    @test norm(gf.matrix_elements(H) - h) < 1.0e-12

    # Construct with array of vertices
    # 2D square lattice Hamiltonian
    sites = [(i, j) for i in 1:N for j in 1:N]
    H_graph = gf.GaussianOperator(sites)
    for i in 1:N, j in 1:N
        (i < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i + 1, j), -1))
        (j < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i, j + 1), -1))
    end

    for r in sites, c in sites
        if (r[1] + 1, r[2]) == c || (r[1], r[2] + 1) == c
            @test gf.matrix_elements(H_graph, r, c) == -1
        elseif (c[1] + 1, c[2]) == r || (c[1], c[2] + 1) == r
            @test gf.matrix_elements(H_graph, r, c) == -1
        else
            @test gf.matrix_elements(H_graph, r, c) == 0
        end
    end
end

@testset "Tuple Sum Syntax" begin
    N = 4

    # Build a chain Hamiltonian using += tuple syntax (C†/C style)
    H_new = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H_new += -1, "C†", j, "C", j + 1
        H_new += -1, "C†", j + 1, "C", j
    end

    # Compare against the same Hamiltonian built with add_hop
    H_old = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H_old = gf.add_hop(H_old, j, j + 1, -1)
    end

    @test norm(gf.matrix_elements(H_new) - gf.matrix_elements(H_old)) < 1.0e-12

    # "Cdag" string is also accepted as an alias for "C†"
    H_cdag = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H_cdag += -1, "Cdag", j, "C", j + 1
        H_cdag += -1, "Cdag", j + 1, "C", j
    end
    @test norm(gf.matrix_elements(H_cdag) - gf.matrix_elements(H_old)) < 1.0e-12

    # Coefficient-free form (defaults to 1.0)
    O = gf.GaussianOperator(N)
    O += "C†", 1, "C", 2
    @test gf.matrix_elements(O, 1, 2) ≈ 1.0
    @test gf.matrix_elements(O, 2, 1) ≈ 0.0

    # add_cdag_c and add_c_cdag produce the same result
    O2 = gf.GaussianOperator(N)
    O2 = gf.add_cdag_c(O2, 1, 2)
    @test norm(gf.matrix_elements(O) - gf.matrix_elements(O2)) < 1.0e-12

    # Anticommutation: "C",j,"C†",i gives the negative of "C†",i,"C",j
    A = gf.GaussianOperator(N)
    A += 2.0, "C†", 1, "C", 3

    B = gf.GaussianOperator(N)
    B += 2.0, "C", 3, "C†", 1

    @test norm(gf.matrix_elements(A) + gf.matrix_elements(B)) < 1.0e-12

    # General quadratic form built from a matrix
    h = [Float64(i + j) for i in 1:N, j in 1:N]
    H_mat = gf.GaussianOperator(N)
    for i in 1:N, j in 1:N
        H_mat += h[i, j], "C†", i, "C", j
    end
    @test norm(gf.matrix_elements(H_mat) - h) < 1.0e-12

    # Invalid label check
    @test_throws ErrorException (gf.GaussianOperator(N) + ("C†", N + 1, "C", 1))
    @test_throws ErrorException (gf.GaussianOperator(N) + ("C†", 1, "C", N + 1))

    # Invalid operator names
    @test_throws ErrorException (gf.GaussianOperator(N) + ("X", 1, "C", 2))
    @test_throws ErrorException (gf.GaussianOperator(N) + ("C†", 1, "C†", 2))
end

@testset "Tuple Sum with Spinful Labels" begin
    N = 3
    ups = [gf.Up(j) for j in 1:N]
    dns = [gf.Dn(j) for j in 1:N]
    verts = vcat(ups, dns)

    H_new = gf.GaussianOperator(verts)
    for j in 1:(N - 1)
        H_new += -1, "C†", gf.Up(j), "C", gf.Up(j + 1)
        H_new += -1, "C†", gf.Up(j + 1), "C", gf.Up(j)
        H_new += -1, "C†", gf.Dn(j), "C", gf.Dn(j + 1)
        H_new += -1, "C†", gf.Dn(j + 1), "C", gf.Dn(j)
    end

    H_old = gf.GaussianOperator(verts)
    for j in 1:(N - 1)
        H_old = gf.add_hop(H_old, gf.Up(j), gf.Up(j + 1), -1)
        H_old = gf.add_hop(H_old, gf.Dn(j), gf.Dn(j + 1), -1)
    end

    @test norm(gf.matrix_elements(H_new) - gf.matrix_elements(H_old)) < 1.0e-12

    # Invalid spin label check
    @test_throws ErrorException (
        gf.GaussianOperator(verts) + ("C†", gf.Up(N + 1), "C", gf.Up(1))
    )
end

@testset "Energies and States" begin
    N = 4
    # Construct with array of vertices
    # 2D square lattice Hamiltonian
    sites = [(i, j) for i in 1:N for j in 1:N]
    H_graph = gf.GaussianOperator(sites)
    for i in 1:N, j in 1:N
        (i < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i + 1, j), -1))
        (j < N) && (H_graph = gf.add_hop(H_graph, (i, j), (i, j + 1), -1))
    end
    ϵ, ϕ = gf.energies_states(H_graph)

    @test ϕ isa NamedArray
    @test names(ϕ, 1) == sites
    @test names(ϕ, 2) == 1:length(sites)

    @test ϵ isa Vector{Float64}
    @test length(ϵ) == length(sites)

    h_reconstruct = ϕ * Diagonal(ϵ) * ϕ'
    @test norm(gf.matrix_elements(H_graph) - h_reconstruct) < 1.0e-12
end

@testset "Greens Functions" begin
    N = 10
    H = electron_chain_h(N)
    times = 0:0.1:6

    G = gf.greens_function(H, times)
    GG = gf.greater_greens_function(H, times)
    GL = gf.lesser_greens_function(H, times)
    @test norm(G - (GG - GL)) < 1.0e-10
    @test size(G) == (length(times), 2N, 2N)
    @test size(GG) == (length(times), 2N, 2N)
    @test size(GL) == (length(times), 2N, 2N)
    @test G isa NamedArray
    @test GL isa NamedArray
    @test GG isa NamedArray

    # Test pre-selecting labels
    # for computing G, GG, GL
    labels = [2, 4]
    G = gf.greens_function(H, times; labels)
    GG = gf.greater_greens_function(H, times; labels)
    GL = gf.lesser_greens_function(H, times; labels)
    @test norm(G - (GG - GL)) < 1.0e-10
    @test size(G) == (length(times), 2, 2)
    @test size(GG) == (length(times), 2, 2)
    @test size(GL) == (length(times), 2, 2)
    @test G isa NamedArray
    @test GL isa NamedArray
    @test GG isa NamedArray
end

@testset "Expected Value" begin
    N = 10
    H = electron_chain_h(N)
    E0, ϕ0 = gf.ground_state(H)
    @test gf.expect(H, ϕ0) ≈ E0

    a = 0.2
    ϕ0a = ϕ0 * a
    @test gf.expect(H, ϕ0a) ≈ E0 * a^2
end
