function fermion_chain_h(N)
    H = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H = gf.add_hop(H, j, j + 1, -1)
    end
    return H
end

function electron_chain_h(N)
    ups = [gf.Up(j) for j in 1:N]
    dns = [gf.Dn(j) for j in 1:N]
    verts = vcat(ups, dns)
    H = gf.GaussianOperator(verts)
    for j in 1:(N - 1)
        H = gf.add_hop(H, gf.Up(j), gf.Up(j + 1), -1)
        H = gf.add_hop(H, gf.Dn(j), gf.Dn(j + 1), -1)
    end
    return H
end

function square_lattice_h(N)
    verts = [(i, j) for i in 1:N for j in 1:N]
    H = gf.GaussianOperator(verts)
    for i in 1:N,j in 1:N
        (i < N) && (H = gf.add_hop(H, (i, j), (i + 1, j), -1))
        (j < N) && (H = gf.add_hop(H, (i, j), (i, j + 1), -1))
    end
    return H
end
