using LinearAlgebra: norm
using NamedArrays: NamedArray
using Printf: @sprintf

function pause()
    print(stdout, "(Paused) ")
    c = read(stdin, 1)
    c == UInt8[0x71] && exit(0)
    return nothing
end

"""
    CreationOperator(labels, orbital::Vector)

Represents a creation operator ``\\hat{v}^\\dagger = \\sum_j v_j \\, \\hat{c}^\\dagger_j``
which is a linear combination of single-site creation operators ``\\hat{c}^\\dagger_j``
with coefficients given by the vector `orbital`.

The `labels` argument specifies the mode labels for each coefficient and must
match the labels of any [`GaussianState`](@ref GaussianFermions.GaussianState) the
operator will be applied to.

Use [`apply`](@ref apply(::CreationOperator, ::GaussianState)) to act with this
operator on a Gaussian state.

# Examples

```julia
import GaussianFermions as gf

# Create c†₁ (creation operator on site 1)
v = zeros(4);
v[1] = 1.0
Cd = gf.CreationOperator(1:4, v)

# Create a superposition of creation operators
v = normalize(randn(4))
Cd = gf.CreationOperator(1:4, v)

# Spinful system
using GaussianFermions: Up, Dn
labels = [Up(1), Up(2), Dn(1), Dn(2)]
v = zeros(4);
v[1] = 1.0  # c†_{1,↑}
Cd = gf.CreationOperator(labels, v)
```
"""
struct CreationOperator
    orbital::NamedArray
    function CreationOperator(orbital::NamedArray)
        return new(orbital)
    end
end

function CreationOperator(labels, orbital = zeros(length(labels)))
    return CreationOperator(NamedArray(orbital, (labels,), ("Labels",)))
end

labels(Cdag::CreationOperator) = names(Cdag.orbital, 1)

Base.copy(C::CreationOperator) = CreationOperator(C.orbital)

function (Cdag::CreationOperator + t::Tuple)
    coef, label = process_tuple(t; opnames = ("Cdag", "C†"))
    if !(label in labels(Cdag))
        error("Label $label is invalid in creation operator sum")
    end
    Cdag = copy(Cdag)
    Cdag.orbital[label] += coef
    return Cdag
end

"""
    AnnihilationOperator(labels, orbital::Vector)

Represents an annihilation operator ``\\hat{w} = \\sum_j \\bar{w}_j \\, \\hat{c}_j``
which is a linear combination of single-site annihilation operators ``\\hat{c}_j``
with coefficients given by the vector `orbital`.

The `labels` argument specifies the mode labels for each coefficient and must
match the labels of any [`GaussianState`](@ref GaussianFermions.GaussianState) the
operator will be applied to.

Use [`apply`](@ref apply(::AnnihilationOperator, ::GaussianState)) to act with this
operator on a Gaussian state.

# Examples

```julia
import GaussianFermions as gf

# Create c₃ (annihilation operator on site 3)
w = zeros(4);
w[3] = 1.0
C = gf.AnnihilationOperator(1:4, w)

# Create a superposition of annihilation operators
w = normalize(randn(4))
C = gf.AnnihilationOperator(1:4, w)
```
"""
struct AnnihilationOperator
    orbital::NamedArray
    function AnnihilationOperator(orbital::NamedArray)
        return new(orbital)
    end
end

function AnnihilationOperator(labels, orbital = zeros(length(labels)))
    return AnnihilationOperator(NamedArray(orbital, (labels,), ("Labels",)))
end

labels(C::AnnihilationOperator) = names(C.orbital, 1)

Base.copy(C::AnnihilationOperator) = AnnihilationOperator(C.orbital)

function (C::AnnihilationOperator + t::Tuple)
    coef, label = process_tuple(t; opnames = ("C",))
    if !(label in labels(C))
        error("Label $label is invalid in annihilation operator sum")
    end
    C = copy(C)
    C.orbital[label] += coef
    return C
end

"""
    apply(Cdag::CreationOperator, ψ::GaussianState) -> GaussianState

Act with the creation operator `Cdag` on the Gaussian state `ψ`, returning the
resulting Gaussian state ``|\\tilde{\\psi}\\rangle = \\hat{v}^\\dagger |\\psi\\rangle``.

The returned state has particle number ``N_f + 1`` and its correlation matrix is

```math
C_{\\tilde{\\psi}} = \\|v_0\\|^2 \\, C_\\psi + v_0 v_0^\\dagger
```

where ``v_0 = (1 - C_\\psi) v`` is the projection of the orbital vector ``v`` onto
the unoccupied subspace of ``C_\\psi``.

The returned state is generally not normalized. Its `trace`
gives ``\\langle\\tilde{\\psi}|\\tilde{\\psi}\\rangle = \\|v_0\\|^2``, which equals the
probability of successfully creating the particle.

Throws an error if the resulting state has near-zero norm (i.e. the orbital is
already fully occupied).

# Example

```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j+1, -1.0)
end
_, ψ = gf.ground_state(H; Nf = 2)

v = zeros(4);
v[1] = 1.0
Cd = gf.CreationOperator(1:4, v)
ψ_new = gf.apply(Cd, ψ)  # ψ_new = c†₁ |ψ⟩
```
"""
function apply(Cdag::CreationOperator, ψ::GaussianState)
    v = Vector(Cdag.orbital)
    Cm = Matrix(correlation_matrix(ψ))
    v0 = v - Cm * v
    nrm0 = norm(v0)
    trace = nrm0^2
    if trace < 1.0e-12
        error(@sprintf("Nearly zero state in creation apply, trace = %.4E\n", trace))
    end
    v0 /= nrm0

    tol = 1.0e-6
    f = occupancy(ψ)
    occ = findall(ν -> isapprox(1.0, ν; atol = tol), f)
    unocc = findall(ν -> isapprox(0.0, ν; atol = tol), f)
    Φ = Matrix(orbitals(ψ))
    Φ_occ = Φ[:, occ]
    Φ_unocc = Φ[:, unocc]
    N_unocc = length(unocc)

    if N_unocc == 1
        # v0 is the only unoccupied orbital; all modes become occupied
        Φ_new = [Φ_occ v0]
        f_new = ones(length(f))
    else
        # Project v0 out of each unoccupied orbital and drop the most-aligned one
        drop = argmax(abs.(Φ_unocc' * v0))
        keep = deleteat!(collect(1:N_unocc), drop)
        P = Φ_unocc[:, keep] .- v0 .* (v0' * Φ_unocc[:, keep])

        # Thin QR of projected remaining unoccupied orbitals
        Q_raw = Matrix(la.qr(P).Q)[:, 1:(N_unocc - 1)]

        # v0 is uniquely determined (no sign ambiguity); Q_raw is unoccupied
        # so its sign does not affect inner products or observables
        Φ_new = [Φ_occ Q_raw v0]
        f_new = [ones(length(occ)); zeros(N_unocc - 1); 1.0]
    end

    ϕ_labeled = NamedArray(Φ_new, (labels(ψ), 1:length(f_new)), ("Labels", "N. Orbitals"))
    return GaussianState(ϕ_labeled, f_new, trace)
end

"""
    apply(C::AnnihilationOperator, ψ::GaussianState) -> GaussianState

Act with the annihilation operator `C` on the Gaussian state `ψ`, returning the
resulting Gaussian state ``|\\tilde{\\eta}\\rangle = \\hat{w} |\\psi\\rangle``.

The returned state has particle number ``N_f - 1`` and its correlation matrix is

```math
C_{\\tilde{\\eta}} = \\|w_1\\|^2 \\, C_\\psi - w_1 w_1^\\dagger
```

where ``w_1 = C_\\psi w`` is the projection of the orbital vector ``w`` onto
the occupied subspace of ``C_\\psi``.

The returned state is generally not normalized. Its `trace`
gives ``\\langle\\tilde{\\eta}|\\tilde{\\eta}\\rangle = \\|w_1\\|^2``, which equals the
probability of successfully annihilating the particle.

Throws an error if the resulting state has near-zero norm (i.e. the orbital is
unoccupied).

# Example

```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j+1, -1.0)
end
_, ψ = gf.ground_state(H; Nf = 2)

w = zeros(4);
w[1] = 1.0
C = gf.AnnihilationOperator(1:4, w)
ψ_new = gf.apply(C, ψ)  # ψ_new = c₁ |ψ⟩
```
"""
function apply(C::AnnihilationOperator, ψ::GaussianState)
    w = Vector(C.orbital)
    Cm = Matrix(correlation_matrix(ψ))
    w1 = Cm * w
    nrm1 = norm(w1)
    trace = nrm1^2
    if trace < 1.0e-12
        error(@sprintf("Nearly zero state in annihilation apply, trace = %.4E\n", trace))
    end
    w1 /= nrm1

    tol = 1.0e-6
    f = occupancy(ψ)
    occ = findall(ν -> isapprox(1.0, ν; atol = tol), f)
    unocc = findall(ν -> isapprox(0.0, ν; atol = tol), f)
    Φ = Matrix(orbitals(ψ))
    Φ_occ = Φ[:, occ]
    Φ_unocc = Φ[:, unocc]
    Nf = length(occ)

    if Nf == 1
        # After annihilation: 0 occupied orbitals; w1 becomes unoccupied
        Φ_new = [Φ_unocc w1]
        f_new = zeros(length(f))
    else
        # Project w1 out of each occupied orbital and drop the most-aligned one
        drop = argmax(abs.(Φ_occ' * w1))
        keep = deleteat!(collect(1:Nf), drop)
        P = Φ_occ[:, keep] .- w1 .* (w1' * Φ_occ[:, keep])

        # Thin QR of projected remaining occupied orbitals
        Q_raw = Matrix(la.qr(P).Q)[:, 1:(Nf - 1)]

        # Sign fix: ensure det([α̂ | Φ_occ' Q_raw]) = +1
        # This guarantees inner(c_w ψ, c_v ψ) = w† C_ψ v for all w, v
        α_hat = Φ_occ' * w1
        α_hat ./= norm(α_hat)
        M = [α_hat Φ_occ' * Q_raw]
        s = sign(real(la.det(M)))
        Q_raw[:, 1] .*= s

        Φ_new = [Φ_unocc w1 Q_raw]
        f_new = [zeros(length(unocc) + 1); ones(Nf - 1)]
    end

    ϕ_labeled = NamedArray(Φ_new, (labels(ψ), 1:length(f_new)), ("Labels", "N. Orbitals"))
    return GaussianState(ϕ_labeled, f_new, trace)
end
