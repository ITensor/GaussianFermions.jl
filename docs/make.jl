using GaussianFermions
using Documenter

makedocs(;
  modules=[GaussianFermions],
  sitename="GaussianFermions.jl",
  warnonly=[:missing_docs, :cross_references],
  pages=[
    "Home" => "index.md",
    "Background" => "theory.md",
    "Operators" => "operators.md",
    "States" => "states.md",
    "Creation & Annihilation" => "creation_annihilation.md",
    "Measurements" => "measurements.md",
    "Dynamics" => "dynamics.md",
    "Spinful Systems" => "spinful.md",
  ],
)
