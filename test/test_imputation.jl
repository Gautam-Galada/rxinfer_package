using CSV, DataFrames

# First ensure the module is loaded properly
include("../src/MissingDataImputer.jl")

# Now use the module explicitly with all necessary imports
using .MissingDataImputer
using .MissingDataImputer.ImputationStrategies
using .MissingDataImputer.PlotUtils
using StatsBase  # Add this explicit import just in case

# Load your test data
df = CSV.read("data/test_data.csv", DataFrame)
println("Original data:")
println(df)

# Use fully qualified names for the enum values
imputed_standard = ImputationStrategies.impute_missing(df, ImputationStrategies.Standard)
println("Standard Strategy:")
println(imputed_standard)

imputed_benford = ImputationStrategies.impute_missing(df, ImputationStrategies.Benford)
println("Benford Strategy:")
println(imputed_benford)

# Convert Dict to DataFrame
imputed_standard_df = DataFrame(imputed_standard)
imputed_benford_df = DataFrame(imputed_benford)

# Visualize the results
MissingDataImputer.PlotUtils.plot_imputation(df, imputed_standard_df)
MissingDataImputer.PlotUtils.plot_imputation(df, imputed_benford_df)