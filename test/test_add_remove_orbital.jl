import GaussianFermions as gf
import LinearAlgebra as la
using Test

include("utilities/hamiltonians.jl")

@testset "Creation and Annihilation Operator Tests" begin

    # ──────────────────────────────────────────────────────────────────────────────
    # Setup: 10-site chain ground state used across all testsets
    # ──────────────────────────────────────────────────────────────────────────────
    N = 10
    Nf = N ÷ 2
    H = fermion_chain_h(N)
    _, ψ0 = gf.ground_state(H; Nf)
    C0 = Matrix(gf.correlation_matrix(ψ0))  # C_ij = ⟨ψ₀|c†_i c_j|ψ₀⟩

    # ──────────────────────────────────────────────────────────────────────────────
    # AnnihilationOperator construction
    # ──────────────────────────────────────────────────────────────────────────────

    @testset "AnnihilationOperator construction methods" begin
        u = randn(N)

        # Method 1: direct vector
        Cu_direct = gf.AnnihilationOperator(1:N, u)

        # Method 2: summation interface — start from zero and accumulate terms
        Cu_sum = gf.AnnihilationOperator(1:N)
        for j in 1:N
            Cu_sum += u[j], "C", j
        end

        # Method 3: single-site annihilation operator with implicit coefficient 1
        C3 = gf.AnnihilationOperator(1:N)
        C3 += "C", 3

        # All construction paths must yield identical states when applied
        ψ_direct = gf.apply(Cu_direct, ψ0)
        ψ_sum = gf.apply(Cu_sum, ψ0)

        @test gf.correlation_matrix(ψ_direct) ≈ gf.correlation_matrix(ψ_sum) atol = 1.0e-10
        @test gf.trace(ψ_direct) ≈ gf.trace(ψ_sum) atol = 1.0e-10
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # CreationOperator construction
    # ──────────────────────────────────────────────────────────────────────────────

    @testset "CreationOperator construction methods" begin
        v = randn(N)

        # Method 1: direct vector
        Cdagv_direct = gf.CreationOperator(1:N, v)

        # Method 2: summation interface with "C†" symbol
        Cdagv_sum = gf.CreationOperator(1:N)
        for j in 1:N
            Cdagv_sum += v[j], "C†", j
        end

        # Method 3: summation interface with "Cdag" symbol (alternate accepted name)
        Cdagv_alt = gf.CreationOperator(1:N)
        for j in 1:N
            Cdagv_alt += v[j], "Cdag", j
        end

        # Method 4: single-site creation with implicit coefficient 1
        Cdag1 = gf.CreationOperator(1:N)
        Cdag1 += "C†", 1

        ψ_direct = gf.apply(Cdagv_direct, ψ0)
        ψ_sum = gf.apply(Cdagv_sum, ψ0)
        ψ_alt = gf.apply(Cdagv_alt, ψ0)

        @test gf.correlation_matrix(ψ_direct) ≈ gf.correlation_matrix(ψ_sum) atol = 1.0e-10
        @test gf.correlation_matrix(ψ_direct) ≈ gf.correlation_matrix(ψ_alt) atol = 1.0e-10
        @test gf.trace(ψ_direct) ≈ gf.trace(ψ_sum) atol = 1.0e-10
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Physical identity for annihilation:
    #   ⟨(∑_j u_j c_j)ψ₀ | (∑_k v_k c_k)ψ₀⟩ = u† C₀ v
    #
    # Proof: expanding gives ∑_{jk} ū_j v_k ⟨ψ₀|c†_j c_k|ψ₀⟩ = u†C₀v.
    #
    # Verify this using the polarization identity:
    #   ⟨ψu|ψv⟩ = (‖ψu+ψv‖² - ‖ψu‖² - ‖ψv‖²) / 2
    # Since A_{u+v}|ψ0⟩ = A_u|ψ0⟩ + A_v|ψ0⟩, we have ‖A_{u+v}|ψ0⟩‖² = gf.trace(ψuv).
    # ──────────────────────────────────────────────────────────────────────────────

    @testset "Annihilation: ⟨ψu|ψv⟩ = u†C₀v" begin
        u = randn(N)
        v = randn(N)

        # Use complementary construction methods to also exercise both interfaces
        Cu = gf.AnnihilationOperator(1:N, u)     # direct vector
        Cv = gf.AnnihilationOperator(1:N)         # summation interface
        for j in 1:N
            Cv += v[j], "C", j
        end
        Cupv = gf.AnnihilationOperator(1:N, u + v)  # direct, for polarization

        ψu = gf.apply(Cu, ψ0)
        ψv = gf.apply(Cv, ψ0)
        ψuv = gf.apply(Cupv, ψ0)

        # Self-overlaps: ‖A_u|ψ0⟩‖² = u†C₀u
        @test gf.trace(ψu) ≈ real(u' * C0 * u) atol = 1.0e-10
        @test gf.trace(ψv) ≈ real(v' * C0 * v) atol = 1.0e-10

        # Cross term via polarization: ⟨ψu|ψv⟩ = u†C₀v
        inner_uv = (gf.trace(ψuv) - gf.trace(ψu) - gf.trace(ψv)) / 2
        @test inner_uv ≈ real(u' * C0 * v) atol = 1.0e-10
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Physical identity for creation:
    #   ⟨(∑_j u_j c†_j)ψ₀ | (∑_k v_k c†_k)ψ₀⟩ = u† (I − C₀) v
    #
    # From c_j c†_k = δ_{jk} − c†_k c_j:
    #   ∑_{jk} ū_j v_k ⟨ψ₀|c_j c†_k|ψ₀⟩ = ∑_{jk} ū_j v_k (δ_{jk} − C₀_{kj}) = u†(I−C₀)v
    # ──────────────────────────────────────────────────────────────────────────────

    @testset "Creation: ⟨ψu†|ψv†⟩ = u†(I−C₀)v" begin
        u = randn(N)
        v = randn(N)

        Cdagu = gf.CreationOperator(1:N)          # summation interface
        for j in 1:N
            Cdagu += u[j], "C†", j
        end
        Cdagv = gf.CreationOperator(1:N, v)       # direct vector
        Cdagupv = gf.CreationOperator(1:N, u + v)   # direct, for polarization

        ψu = gf.apply(Cdagu, ψ0)
        ψv = gf.apply(Cdagv, ψ0)
        ψuv = gf.apply(Cdagupv, ψ0)

        IC0 = la.I - C0

        # Self-overlaps: ‖A†_u|ψ0⟩‖² = u†(I−C₀)u
        @test gf.trace(ψu) ≈ real(u' * IC0 * u) atol = 1.0e-12
        @test gf.trace(ψv) ≈ real(v' * IC0 * v) atol = 1.0e-12

        # Cross term via polarization: ⟨ψu†|ψv†⟩ = u†(I−C₀)v
        inner_uv = (gf.trace(ψuv) - gf.trace(ψu) - gf.trace(ψv)) / 2
        @test inner_uv ≈ real(u' * IC0 * v) atol = 1.0e-12
    end
end
