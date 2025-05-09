module PlotUtils

using Plots, DataFrames

export plot_imputation


function plot_imputation(original::DataFrame, imputed::DataFrame)
    plt = plot(layout=(length(names(original)), 1), size=(800, 600))
    for (i, col) in enumerate(names(original))
        if eltype(original[!, col]) <: Union{Missing, Real}
            # Extract valid (non-missing) indices and values
            orig_valid = collect(skipmissing(original[!, col]))
            orig_indices = findall(.!ismissing.(original[!, col]))
            
            imp_valid = collect(skipmissing(imputed[!, col]))
            imp_indices = findall(.!ismissing.(imputed[!, col]))
            
            # Plot only valid points
            scatter!(plt[i], orig_indices, orig_valid, label="Original: $col", color=:blue)
            scatter!(plt[i], imp_indices, imp_valid, label="Imputed: $col", color=:red)
        end
    end
    savefig(plt, "imputation_plot.png")
    closeall()
end
end # module