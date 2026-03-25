function process_tuple(t::Tuple; opnames::Tuple)
    coef = 1.0
    n = 1
    if first(t) isa Number
        coef = first(t)
        n += 1
    end
    valid_operator = (t[n] in opnames)
    n += 1
    if n != length(t) || !valid_operator
        println("Valid opnames: ", opnames)
        error(
            "Terms must be of the form c,\"Opname\",l or \"Opname\",l when adding to CreationOperator"
        )
    end
    return coef, t[n]
end

function process_gaussian_tuple(t::Tuple)
    coef = 1.0
    n = 1
    if first(t) isa Number
        coef = first(t)
        n += 1
    end
    if length(t) - n + 1 != 4
        error(
            "Terms must be of the form (coef,)\"Op1\",l1,\"Op2\",l2 when adding to GaussianOperator"
        )
    end
    op1 = t[n]
    n += 1
    label1 = t[n]
    n += 1
    op2 = t[n]
    n += 1
    label2 = t[n]

    is_creation(op) = op in ("Cdag", "C†")
    is_annihilation(op) = op == "C"

    if is_creation(op1) && is_annihilation(op2)
        return coef, :cdag_c, label1, label2
    elseif is_annihilation(op1) && is_creation(op2)
        return coef, :c_cdag, label1, label2
    else
        error(
            "Operator pair ($op1, $op2) is not valid. Must be (\"Cdag\"/\"C†\", \"C\") or (\"C\", \"Cdag\"/\"C†\")"
        )
    end
end
