# Spin-1/2 fermion chain driven by a time-dependent second-neighbor hopping.
# The state starts as the ground state of a nearest-neighbor chain H1, then is
# time-evolved under H1 + f(t)*H2 where H2 contains next-nearest-neighbor hops
# and f(t) is a smooth ramp function.

import GaussianFermions as gf
using GaussianFermions: Dn, Up

let
    N = 10
    Nfup = N ÷ 2
    Nfdn = N ÷ 2
    t1 = 1.0
    t2 = 0.5

    dt = 0.05
    T = 10.0
    time_range = 0.0:dt:10

    f(t) = sin(π * t / T)   # smooth ramp from 0 to 0 over total time

    ups = [gf.Up(j) for j in 1:N]
    dns = [gf.Dn(j) for j in 1:N]
    labels = vcat(ups, dns)

    H1 = gf.GaussianOperator(labels)
    for j in 1:(N - 1)
        H1 += -t1, "C†", Up(j), "C", Up(j + 1)
        H1 += -t1, "C†", Up(j + 1), "C", Up(j)
        H1 += -t1, "C†", Dn(j), "C", Dn(j + 1)
        H1 += -t1, "C†", Dn(j + 1), "C", Dn(j)
    end

    H2 = gf.GaussianOperator(labels)
    for j in 1:(N - 2)
        H2 += -t2, "C†", Up(j), "C", Up(j + 2)
        H2 += -t2, "C†", Up(j + 2), "C", Up(j)
        H2 += -t2, "C†", Dn(j), "C", Dn(j + 2)
        H2 += -t2, "C†", Dn(j + 2), "C", Dn(j)
    end

    E0, ϕ0 = gf.ground_state(H1; Nf = Nfup + Nfdn)
    @show E0

    ϕt = copy(ϕ0)
    for t in time_range
        ϕt = gf.time_evolve(H1 + f(t) * H2, dt, ϕt)
    end

    @show gf.expect(H1, ϕt)

    return nothing
end
