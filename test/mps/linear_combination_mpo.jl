using ITensors
using ITensorMPS

"""
    linear_combination_mpo(sites, w::Vector, opname::String) -> MPO

Construct a bond-dimension-2 MPO representing the operator

    Σ_j w[j] * opname_j

with Jordan-Wigner string ("F") operators included.

# Arguments
- `sites`: vector of site indices (from `siteinds`)
- `w`: coefficient vector (length must match number of sites)
- `opname`: operator name string. For "Fermion" sites: "C" or "Cdag".
  For "Electron" sites: "Cup", "Cdn", "Cdagup", or "Cdagdn".
"""
function linear_combination_mpo(sites, w::Vector{<:Number}, opname::String)
    N = length(sites)
    @assert length(w) == N

    if N == 1
        return MPO([w[1] * op(opname, sites[1])])
    end

    # Create link indices (bond dimension 2)
    links = Vector{Index}(undef, N - 1)
    if hasqns(sites[1])
        # State 1 QN = operator flux (active JW string, operator not yet applied)
        # State 2 QN = zero (operator already applied, identity from here)
        op_qn = flux(op(opname, sites[1]))
        zero_qn = qn(sites[1], 1)
        for b in 1:(N - 1)
            links[b] = Index([op_qn => 1, zero_qn => 1], "Link,l=$b")
        end
    else
        for b in 1:(N - 1)
            links[b] = Index(2, "Link,l=$b")
        end
    end

    tensors = Vector{ITensor}(undef, N)

    for j in 1:N
        s = sites[j]
        F_j = op("F", s)
        I_j = op("Id", s)
        c_j = op(opname, s)

        if j == 1
            # Row vector: [F, w*c]
            lr = links[1]
            tensors[j] = F_j * onehot(lr => 1) + w[j] * c_j * onehot(lr => 2)
        elseif j == N
            # Column vector: [w*c; I]
            ll = dag(links[N - 1])
            tensors[j] = w[j] * c_j * onehot(ll => 1) + I_j * onehot(ll => 2)
        else
            # Matrix: |F   w*c|
            #         |0    I |
            ll = dag(links[j - 1])
            lr = links[j]
            tensors[j] = F_j * onehot(ll => 1) * onehot(lr => 1) +
                         w[j] * c_j * onehot(ll => 1) * onehot(lr => 2) +
                         I_j * onehot(ll => 2) * onehot(lr => 2)
        end
    end

    return MPO(tensors)
end
