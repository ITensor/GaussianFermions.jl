
# TODO: this method is buggy. It doesn't change the norm
#       of the state even when `orb` is not orthogonal to the filled orbitals
function add_orbital(ψ::GaussianState, orb::Vector)
    ispure(ψ) || error("add_orbital currently defined for pure states only")
    abs(la.norm(orb) - 1) < 1.0e-9 ||
        error("expected orbital `orb` to be normalized in `add_orbital`")
    ϕ, fill = copy(orbitals(ψ)), copy(filling(ψ))
    N, Np = length(ψ), count(>(0), fill)
    ϕ_augmented = zeros(N, N)
    ϕ_augmented[:, 1:Np] = ϕ[:, 1:Np]
    ϕ_augmented[:, Np + 1] = orb
    ϕ_augmented[:, (Np + 2):end] = ϕ[:, (Np + 1):(end - 1)]
    #ϕ_augmented = hcat(ϕ[:,1:Np],orb,ϕ[:,Np+1:end])
    display(ϕ_augmented)
    q, _ = la.qr(ϕ_augmented)
    ϕ = Matrix(q)
    fill[Np + 1] = 1.0
    return GaussianState(ϕ, fill)
end
