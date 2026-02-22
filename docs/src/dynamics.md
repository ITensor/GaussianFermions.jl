# Dynamics

## Greens Functions

```@docs
GaussianFermions.greens_function(::GaussianFermions.GaussianOperator, times; kws...)
GaussianFermions.lesser_greens_function(::GaussianFermions.GaussianOperator, times; kws...)
GaussianFermions.greater_greens_function(::GaussianFermions.GaussianOperator, times; kws...)
```

## Time Evolution

```@docs
GaussianFermions.time_evolve(::GaussianFermions.GaussianOperator, ::Number, ::GaussianFermions.GaussianState)
GaussianFermions.time_evolve(::GaussianFermions.GaussianOperator, ::Number, ::GaussianFermions.GaussianOperator)
```
