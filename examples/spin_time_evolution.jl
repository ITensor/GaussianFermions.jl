import GaussianFermions as gf
using Printf

include("utilities/write_data.jl")

let
    N = 31
    Nfup = (N - 1) ÷ 2
    Nfdn = (N - 1) ÷ 2
    bond_dim_cutoff = 1.0e-6

    H = gf.SpinGaussianOperator(N)
    for j in 2:(N - 1)
        H = gf.add_hop(H, j, j + 1, -1)
    end

    B = gf.SpinGaussianOperator(N)
    B = gf.add_hop(H, 1, 2, -1)

    E0, ψ0 = gf.ground_state(H; Nfup, Nfdn)
    @show E0
    @show gf.expect(H, ψ0)
    @show gf.entanglement(ψ0, 1:(N ÷ 2))
    @show gf.bond_dimension(ψ0, 1:(N ÷ 2), bond_dim_cutoff)

    Ht = gf.time_evolve(H, 0.4, H)
    @show gf.expect(Ht, ψ0)

    A = gf.SpinGaussianOperator(N)
    A = gf.add_cdag_c(A, 1, 2; spin = "up")

    t0 = 4.0
    σ = 0.2
    coef = 0.1
    field(t) = coef * exp(-(t - t0)^2 / (2σ^2)) / √(2π * σ^2)

    @show σ
    @show field(0)

    dt = 0.02
    T = 1000.0
    time_range = 0:dt:T
    Avals = zeros(ComplexF64, length(time_range))
    entanglement = zeros(length(time_range))
    bond_dimension = zeros(Int, length(time_range))

    ψt = copy(ψ0)
    for (n, t) in enumerate(time_range)
        if mod(t, 10.0) ≈ 0.0
            @printf("  t=%.3f\n", t)
        end
        Ht = H + field(t) * B
        ψt = gf.time_evolve(Ht, dt, ψt)
        Avals[n] = gf.expect(A, ψt)
        entanglement[n] = gf.entanglement(ψt, 1:(N ÷ 2))
        bond_dimension[n] = gf.bond_dimension(ψt, 1:(N ÷ 2), bond_dim_cutoff)
    end

    # TODO: why doesn't this version work?
    #At = copy(A)
    #for (n, t) in enumerate(time_range)
    #  Ht = H + field(t)*B
    #  At = gf.time_evolve(Ht, dt, At)
    #  Avals[n] = gf.expect(At, ψ0)
    #end

    times = collect(time_range)

    write_data("A_real.dat", times, real(Avals))
    write_data("A_imag.dat", times, imag(Avals))
    write_data("field.dat", times, field.(times))
    write_data("entanglement.dat", times, entanglement)
    write_data("bond_dimension.dat", times, bond_dimension)

    #
    # Check: convolve G<(t) with field(t)
    #
    # TODO: update this code copied from ImpuritySolving.jl
    #prefactor = 1/2
    #dt = 0.05
    #conv_t_max = 30
    #convolved_G = zeros(ComplexF64, length(times))
    #for (j, t1) in enumerate(times)
    #  for t2 in (-conv_t_max):dt:conv_t_max
    #    convolved_G[j] +=
    #      prefactor *
    #      dt *
    #      field(t2) *
    # TODO: should be G<(t), not GR(t)
    #      retarded_green_function(t1-t2, ϕ, ϵ; sites=impurity_site)
    #  end
    #end
    #convolved_G = -im*convolved_G
    #write_data("convolved_greens_function_real.dat",times,real(convolved_G))
    #write_data("convolved_greens_function_imag.dat",times,imag(convolved_G))


    return nothing
end
