import GaussianFermions as gf

include("utilities/write_data.jl")

let
    N = 7
    Nf = (N - 1) ÷ 2

    H = gf.GaussianOperator(N)
    for j in 2:(N - 1)
        H = gf.add_hop(H, j, j + 1, -1)
    end

    B = gf.GaussianOperator(N)
    B = gf.add_hop(H, 1, 2, -1)

    E0, ψ0 = gf.ground_state(H; Nf)
    @show E0
    @show gf.expect(H, ψ0)
    @show gf.entanglement(ψ0; sites = 1:(N ÷ 2))
    #@show gf.bond_dimension(ψ0; 1:N÷2, 1E-7)

    A = gf.GaussianOperator(N)
    A = gf.add_cdag_c(A, 1, 2)

    t0 = 2.0
    σ = 0.2
    coef = 0.1
    field(t) = coef * exp(-(t - t0)^2 / (2σ^2)) / √(2π * σ^2)

    dt = 0.02
    T = 30.0
    time_range = 0:dt:T
    At = copy(A)
    Avals = zeros(ComplexF64, length(time_range))
    for (n, t) in enumerate(time_range)
        Ht = H + field(t) * B
        At = gf.time_evolve(Ht, dt, At)
        Avals[n] = gf.expect(At, ψ0)
    end

    times = collect(time_range)

    write_data("A_real.dat", times, real(Avals))
    write_data("A_imag.dat", times, imag(Avals))
    write_data("field.dat", times, field.(times))

    return nothing
end
