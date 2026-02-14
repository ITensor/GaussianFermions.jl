#
# TODO:
# Possibly redefine these so input in an arbitrary GaussianState ρ.
#
# Return e.g. G>(t) = Tr[ρ cᵢ(t) c†ⱼ] = Tr[ρ exp(+iHt) cᵢ exp(-iHt) c†ⱼ]
#
# Is there a nice formula for this for arbitrary ρ, not necessarily
# defined in eigenbasis of H?
#

function _compute_correlator(N::Int, f::Function; sites = 1:N, kwargs...)
    site_range = (sites isa AbstractRange) ? sites : collect(sites)
    Ns = length(site_range)

    C = zeros(ComplexF64, Ns, Ns)
    for (ni, i) in enumerate(site_range), (nj, j) in enumerate(site_range)
        C[ni, nj] += f(i, j)
    end

    if sites isa Number
        return C[1, 1]
    end
    return C
end

"""
    retarded_green_function(t, H; kwargs...)

Compute the retarded single-particle Green function
GR(t)_ij = -i θ(t) <ϕ|{cᵢ(t), c†ⱼ(0)}|ϕ>
at time t where |ϕ> is the ground state of the operator H.
"""
function retarded_green_function(
        t::Number, ϕ, ϵ::Vector{Float64}, Nfill = count(<(0), ϵ); kwargs...
    )
    if t < 0.0
        return 0.0
    end
    N = length(ϵ)
    @assert size(ϕ) == (N, N)
    function compute_GR(i, j)
        gr = 0.0im
        for n in 1:N
            gr += -im * ϕ[i, n] * conj(ϕ[j, n]) * exp(-im * ϵ[n] * t)
        end
        return gr
    end
    return _compute_correlator(N, compute_GR; kwargs...)
end

"""
    greater_green_function(t, H; kwargs...)
Compute the greater single-particle Green function
G>(t)_ij = -i <ϕ|cᵢ(t) c†ⱼ(0)|ϕ>
at time t where |ϕ> is the ground state of the operator H.
"""
function greater_green_function(t::Number, H::GaussianOperator; kwargs...)
    N = length(H)
    ϵ, ϕ = energies_states(H)
    Nfill = count(<(0), ϵ)

    function compute_GG(i, j)
        gg = 0.0im
        for n in (Nfill + 1):N
            gg += -im * ϕ[i, n] * conj(ϕ[j, n]) * exp(-im * ϵ[n] * t)
        end
        return gg
    end
    return _compute_correlator(N, compute_GG; kwargs...)
end

"""
    lesser_green_function(t, H; kwargs...)
Compute the lesser single-particle Green function
G<(t)_ij = +i <ϕ|c†ᵢ(0) cⱼ(t)|ϕ>
at time t where |ϕ> is the ground state of H.
"""
function lesser_green_function(
        t::Number,
        ϕ,
        ϵ::Vector{Float64},
        Nfill = count(<(0), ϵ);
        gaussian_broadening = 0.0,
        broadening = 0.0,
        kwargs...,
    )
    N = length(ϵ)
    @assert size(ϕ) == (N, N)
    gauss_b = (gaussian_broadening * t)^2 / 2
    lorentz_b = abs(broadening * t)

    function compute_GL(i, j)
        gl = 0.0im
        for n in 1:Nfill
            gl += +im * ϕ[i, n] * conj(ϕ[j, n]) * exp(-im * ϵ[n] * t - lorentz_b - gauss_b)
        end
        return gl
    end
    return _compute_correlator(N, compute_GL; kwargs...)
end

function spectral_function(
        ω::Number,
        ϕ,
        ϵ::Vector{Float64};
        empirical_omega_shift = 0.0,
        gaussian_broadening = 0.0,
        broadening = 0.0,
        maximum_time = 0.0,
        kwargs...,
    )
    N = length(ϵ)
    @assert size(ϕ) == (N, N)

    delt(ω) = 1.0
    if !iszero(gaussian_broadening)
        η = gaussian_broadening
        delt_gaussian(ω) = √(1 / (2π * η^2)) * exp(-(ω / η)^2 / 2)
        delt = delt_gaussian
    elseif !iszero(broadening)
        delt_b(ω) = (broadening / π) / (broadening^2 + abs(ω)^2)
        delt = delt_b
    elseif !iszero(maximum_time)
        delt_tmax(ω) = (sin(maximum_time * ω) / (π * ω))
        delt = delt_tmax
    else
        error("Must specify either broadening or maximum_time keywords in spectral_function")
    end

    function compute_rho(i, j)
        rho = 0.0im
        for n in 1:N
            rho += ϕ[i, n] * conj(ϕ[j, n]) * delt(ω - ϵ[n] - empirical_omega_shift)
        end
        return rho
    end

    return _compute_correlator(N, compute_rho; kwargs...)
end

function band_function(
        ω::Number, broadening::Number, ϕ, ϵ::Vector{Float64}; fac = 1.0, kwargs...
    )
    N = length(ϵ)
    @assert size(ϕ) == (N, N)
    b = broadening

    function compute_rho(i, j)
        rho = 0.0im
        for n in 1:N
            rho += fac * (b / π) / (b^2 + (ω - ϵ[n])^2)
        end
        return rho
    end

    return _compute_correlator(N, compute_rho; kwargs...)
end
