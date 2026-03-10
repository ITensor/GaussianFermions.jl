import GaussianFermions as gf
import LinearAlgebra: norm

let
    N = 20
    Nf = N ÷ 2
    t = 1.0

    labels = 1:N

    H = gf.GaussianOperator(labels)
    for j in 1:(N - 1)
        H += -t, "C†", j, "C", j+1
        H += -t, "C†", j+1, "C", j
    end
    #display(gf.matrix_elements(H))

    E0, ϕ0 = gf.ground_state(H; Nf)
    n0 = gf.density(ϕ0)
    #display(n0)

    v = rand(N)
    v /= norm(v)
    Cdag = gf.CreationOperator(labels, v)

    Cdag_ϕ0 = gf.apply(Cdag, ϕ0)

    @show gf.density(Cdag_ϕ0)
    @show gf.occupancy(Cdag_ϕ0)
    #@show gf.nparticles(Cdag_ϕ0)

    return
end
