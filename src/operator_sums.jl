
function process_tuple(t::Tuple; opnames::Tuple)
    coef = 1.0
    n = 1
    if first(t) isa Number
        coef = first(t)
        n += 1
    end
    valid_operator =  (t[n] in opnames)
    n += 1
    if n != length(t) || !valid_operator
        println("Valid opnames: ", opnames)
        error("Terms must be of the form c,\"Opname\",l or \"Opname\",l when adding to CreationOperator")
    end
    return coef, t[n]
end
