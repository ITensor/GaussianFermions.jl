function site_labels(ϕ)
    up_labels = filter(v -> (v isa Up), labels(ϕ))
    if length(up_labels) != length(labels(ϕ)) ÷ 2
        error("Expected equal number of Up and Dn labels")
    end
    return map(v -> (v.site), up_labels)
end

"""
    up_correlation_matrix(ϕ::GaussianState; labels=labels(ϕ))

Compute the spin-up correlation matrix ``\\langle c^\\dagger_{\\uparrow,i} c_{\\uparrow,j} \\rangle``
for a spinful state `ϕ`.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
for j in 1:3
    H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1.0)
    H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1.0)
end
_, ϕ = gf.ground_state(H; Nf=4)
gf.up_correlation_matrix(ϕ)
```
"""
function up_correlation_matrix(ϕ::GaussianState; labels = site_labels(ϕ))
    return correlation_matrix(ϕ; labels = [Up(s) for s in labels])
end

"""
    dn_correlation_matrix(ϕ::GaussianState; labels=labels(ϕ))

Compute the spin-down correlation matrix ``\\langle c^\\dagger_{\\downarrow,i} c_{\\downarrow,j} \\rangle``
for a spinful state `ϕ`.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
for j in 1:3
    H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1.0)
    H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1.0)
end
_, ϕ = gf.ground_state(H; Nf=4)
gf.dn_correlation_matrix(ϕ)
```
"""
function dn_correlation_matrix(ϕ::GaussianState; labels = site_labels(ϕ))
    return correlation_matrix(ϕ; labels = [Dn(s) for s in labels])
end

"""
    up_density(ϕ::GaussianState; labels=labels(ϕ))

Return a vector of spin-up occupation numbers ``\\langle n_{\\uparrow,i} \\rangle``.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
for j in 1:3
    H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1.0)
    H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1.0)
end
_, ϕ = gf.ground_state(H; Nf=4)
gf.up_density(ϕ)
```
"""
function up_density(ϕ::GaussianState; kws...)
    return la.diag(up_correlation_matrix(ϕ; kws...))
end

"""
    dn_density(ϕ::GaussianState; labels=labels(ϕ))

Return a vector of spin-down occupation numbers ``\\langle n_{\\downarrow,i} \\rangle``.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
for j in 1:3
    H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1.0)
    H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1.0)
end
_, ϕ = gf.ground_state(H; Nf=4)
gf.dn_density(ϕ)
```
"""
function dn_density(ϕ::GaussianState; kws...)
    return la.diag(dn_correlation_matrix(ϕ; kws...))
end

"""
    total_density(ϕ::GaussianState; labels=labels(ϕ))

Return a vector of total occupation numbers ``\\langle n_{\\uparrow,i} \\rangle + \\langle n_{\\downarrow,i} \\rangle``.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
for j in 1:3
    H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1.0)
    H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1.0)
end
_, ϕ = gf.ground_state(H; Nf=4)
gf.total_density(ϕ)
```
"""
function total_density(ϕ::GaussianState; kws...)
    return up_density(ϕ; kws...) + dn_density(ϕ; kws...)
end
