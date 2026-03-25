using Documenter: Documenter, DocMeta, deploydocs, makedocs
using GaussianFermions: GaussianFermions
using ITensorFormatter: ITensorFormatter

DocMeta.setdocmeta!(
    GaussianFermions, :DocTestSetup, :(using GaussianFermions); recursive = true
)

ITensorFormatter.make_index!(pkgdir(GaussianFermions))

makedocs(;
    modules = [GaussianFermions],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "GaussianFermions.jl",
    warnonly = [:missing_docs, :cross_references],
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/GaussianFermions.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = [
        "Home" => "index.md",
        "Background" => "theory.md",
        "Operators" => "operators.md",
        "States" => "states.md",
        "Creation & Annihilation" => "creation_annihilation.md",
        "Measurements" => "measurements.md",
        "Dynamics" => "dynamics.md",
        "Spinful Systems" => "spinful.md",
    ]
)
