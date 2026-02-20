include("make.jl")

using LiveServer
serve(; dir=joinpath(@__DIR__, "build"), launch_browser=true)
