import GaussianFermions as gf
import LinearAlgebra: norm

let
    N = 20
    Nf = N ÷ 2
    t = 1.0

    labels = 1:N

    H = gf.GaussianOperator(labels)
    for j in 1:(N - 1)
        H = gf.add_hop(H, j, j + 1, -t)
    end

    display(gf.matrix_elements(H))

    E0, ϕ0 = gf.ground_state(H; Nf)
    n0 = gf.density(ϕ0)
    display(n0)

    v = rand(N)
    v /= norm(v)
    Cdag = gf.CreationOperator(labels, v)

    gf.apply(Cdag, ϕ0)

    return
end
