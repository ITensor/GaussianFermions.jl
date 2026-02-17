using Test
import GaussianFermions as gf

@testset "Add and Remove Orbitals" begin
    N = 20
    Nf = N ÷ 2
    H = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H = gf.add_hop(H, j, j + 1, -1.0)
    end
    E0, ϕ0 = gf.ground_state(H; Nf)
    @show gf.density(ϕ0)
    @show sum(gf.density(ϕ0))

    #orb = fill(0.,N)
    #orb[1] = 1.0
    ##orb[2] = 1.0/√2
    #ψ = gf.add_orbital(ϕ0, orb)
    #@show gf.density(ψ)
    #@show sum(gf.density(ψ))
end
