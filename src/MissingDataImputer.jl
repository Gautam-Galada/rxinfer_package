module MissingDataImputer

# External dependencies
using DataFrames, Distributions, CSV, RxInfer, Plots, Statistics, StatsBase

# Include submodules
include("ImputationStrategies.jl")
include("PlotUtils.jl")

# Re-export specific functionality (CRUCIAL FIX)
using .ImputationStrategies: impute_missing, ImputationStrategy, Standard, Benford
using .PlotUtils: plot_imputation  # If you have plotting functions

# Explicit exports list
export impute_missing,
       ImputationStrategy,
       Standard,
       Benford,
       plot_imputation  # Export plot function if exists

end # module