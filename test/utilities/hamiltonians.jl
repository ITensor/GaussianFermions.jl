
function fermion_chain_h(N)
    H = gf.GaussianOperator(N)
    for j=1:(N-1)
        H = gf.add_hop(H, j, j+1, -1)
    end
    return H
end

function electron_chain_h(N)
    ups = [(j,"↑") for j=1:N]
    dns = [(j,"↓") for j=1:N]
    verts = vcat(ups,dns)
    H = gf.GaussianOperator(verts)
    for j=1:(N-1)
        H = gf.add_hop(H, (j,"↑"), (j+1,"↑"), -1)
        H = gf.add_hop(H, (j,"↓"), (j+1,"↓"), -1)
    end
    return H
end

function square_lattice_h(N)
    verts = [(i,j) for i=1:N for j=1:N]
    H = gf.GaussianOperator(verts)
    for i=1:N,j=1:N
        (i < N) && (H = gf.add_hop(H, (i,j), (i+1,j), -1))
        (j < N) && (H = gf.add_hop(H, (i,j), (i,j+1), -1))
    end
    return H
end
