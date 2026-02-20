abstract type Spin end

"""
    Up(site)

Spin-up vertex label wrapping a site index. Used with [`Dn`](@ref) to label
orbitals in spinful fermion systems.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
```
"""
struct Up{T} <: Spin
    site::T
end

"""
    Dn(site)

Spin-down vertex label wrapping a site index. Used with [`Up`](@ref) to label
orbitals in spinful fermion systems.

# Example
```julia
import GaussianFermions as gf

ups = [gf.Up(j) for j in 1:4]
dns = [gf.Dn(j) for j in 1:4]
H = gf.GaussianOperator(vcat(ups, dns))
```
"""
struct Dn{T} <: Spin
    site::T
end

"""
    site(s::Spin)

Return the site index wrapped by the spin label `s`.

# Example
```julia
import GaussianFermions as gf

gf.site(gf.Up(3))  # returns 3
```
"""
site(s::Spin) = s.site

Base.show(io::IO, s::Up) = print(io, "Up(", site(s), ")")
Base.show(io::IO, s::Dn) = print(io, "Dn(", site(s), ")")
