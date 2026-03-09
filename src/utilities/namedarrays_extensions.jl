
function named_vector(names; dimname="")
    N = length(names)
    return NamedArray(zeros(N); names=(names,), dimnames=(dimname,))
end

function named_matrix(rownames, colnames; rowdimname="", coldimname="")
    Nr = length(rownames)
    Nc = length(colnames)
    return NamedArray(zeros(Nr,Nc); names=(rownames,colnames), dimnames=(rowdimname,coldimname))
end
