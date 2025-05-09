module ImputationStrategies

using DataFrames
using Distributions
using Statistics
using StatsBase

export impute_missing, ImputationStrategy, Standard, Benford

@enum ImputationStrategy begin
    Standard
    Benford
end

function impute_missing(df::DataFrame, strategy::ImputationStrategy)
    
    if strategy == Standard
        return standard_imputation(df)
    elseif strategy == Benford
        return benford_imputation(df)
    else
        error("Unknown strategy")
    end
end

function standard_imputation(df::DataFrame)
    imputed = deepcopy(df)
    for col in names(df)
        col_data = df[!, col]
        mask = ismissing.(col_data)
        present_data = skipmissing(col_data)
        μ, σ² = mean(present_data), var(present_data)
        
        for i in eachindex(col_data)
            if mask[i]
                imputed[i, col] = rand(Normal(μ, sqrt(σ²)))
            end
        end
    end
    # Convert to dictionary for Python compatibility
    # return Dict(names(imputed) .=> [Vector(imputed[!, col]) for col in names(imputed)])
    return imputed
end

function benford_imputation(df::DataFrame)
    imputed = deepcopy(df)
    for col in names(df)
        if eltype(df[!, col]) <: Union{Missing, Real}
            mask = ismissing.(df[!, col])
            present = skipmissing(df[!, col])
            
            valid_values = filter(x -> x > 0, collect(present))
            isempty(valid_values) && continue
            
            digits = first_digit_frequencies(valid_values)
            for i in eachindex(df[!, col])
                if mask[i]
                    d = sample_from_benford(digits)
                    value = d * 10.0^rand(0:3)
                    imputed[i, col] = value
                end
            end
        end
    end
    # Convert to dictionary for Python compatibility
    # return Dict(names(imputed) .=> [Vector(imputed[!, col]) for col in names(imputed)])
    return imputed
end

function first_digit_frequencies(values)
    freqs = Dict(d => 0.0 for d in 1:9)
    for v in values
        # Skip zeros and negative values (Benford's Law applies to positive numbers)
        v <= 0 && continue
        
        # Handle values between 0 and 1 (e.g., 0.05 → first digit 5)
        s = string(abs(v))
        if startswith(s, "0.")
            # Find first non-zero digit after the decimal
            digits = split(s, ".")[2]
            first_d = first([d for d in digits if d != '0'])
        else
            first_d = first(s)
        end
        
        d = parse(Int, first_d)
        freqs[d] += 1
    end
    return normalize!(freqs)
end

function normalize!(dict::Dict{Int, Float64})
    total = sum(values(dict))
    for k in keys(dict)
        dict[k] /= total
    end
    return dict
end

function sample_from_benford(freqs::Dict{Int, Float64})
    p = [freqs[d] for d in 1:9]
    return sample(1:9, Weights(p))
end

end # module