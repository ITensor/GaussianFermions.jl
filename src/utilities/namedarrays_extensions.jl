"""
    named_vector(names; dimname="")

Create a zero-initialized `NamedArray` vector of length `length(names)` with
the given `names` as its axis labels and optional dimension name `dimname`.
"""
function named_vector(names; dimname="")
    N = length(names)
    return NamedArray(zeros(N); names=(names,), dimnames=(dimname,))
end

"""
    named_matrix(rownames, colnames; rowdimname="", coldimname="")

Create a zero-initialized `NamedArray` matrix of size `length(rownames) × length(colnames)`
with the given row and column names and optional dimension names.
"""
function named_matrix(rownames, colnames; rowdimname="", coldimname="")
    Nr = length(rownames)
    Nc = length(colnames)
    return NamedArray(zeros(Nr,Nc); names=(rownames,colnames), dimnames=(rowdimname,coldimname))
end
