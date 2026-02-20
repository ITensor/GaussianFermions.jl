using GaussianFermions
using Documenter

makedocs(;
  modules=[GaussianFermions],
  sitename="GaussianFermions.jl",
  warnonly=[:missing_docs],
  pages=[
    "Home" => "index.md",
    "Background" => "theory.md",
    "Operators" => "operators.md",
    "States" => "states.md",
    "Measurements" => "measurements.md",
    "Time Evolution" => "time_evolution.md",
    "Spinful Systems" => "spinful.md",
  ],
)
