# Utility for writing two-column data files (x, y) to disk.
# Supports integer and Float64 values; each row is space-separated.

using Printf: @sprintf

format_string(i::Integer) = @sprintf("%d", i)
format_string(x::Float64) = @sprintf("%.14f", x)
function format_string(x)
    return error("Writing to data file of type $(typeof(x)) not currently supported")
end

function write_data(fname::AbstractString, xv::Vector, yv::Vector)
    f = open(fname, "w")
    for (x, y) in zip(xv, yv)
        write(f, "$(format_string(x)) $(format_string(y))\n")
    end
    return close(f)
end
