using GaussianFermions
using Documenter

makedocs(;
  modules=[GaussianFermions],
  sitename="GaussianFermions.jl",
  warnonly=[:missing_docs],
  pages=[
    "Home" => "index.md",
  ],
)
