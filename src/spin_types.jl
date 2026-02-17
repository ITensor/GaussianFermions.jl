abstract type Spin end

struct Up{T} <: Spin
    site::T
end

struct Dn{T} <: Spin
    site::T
end

site(s::Spin) = s.site

Base.show(io::IO, s::Up) = print(io, "Up(", site(s), ")")
Base.show(io::IO, s::Dn) = print(io, "Dn(", site(s), ")")
