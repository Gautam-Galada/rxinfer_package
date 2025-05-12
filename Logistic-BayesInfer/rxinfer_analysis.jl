using CSV, DataFrames, RxInfer, JSON3, CategoricalArrays, LinearAlgebra

function preprocess_data(df)
    df.Gender = (df.Gender .== "male") .|> Int
    df.Education = categorical(df.Education, levels=["highschool", "bachelor", "master"])
    df = transform(df, :Education => ByRow(levelcode) => :Education_code)
    features = select(df, [:Age, :Income, :Gender, :Education_code])
    target = df.Target
    return Matrix(features), target
end

@model function bayesian_logistic_regression(n_features)
    # Data variables with explicit size constraints
    X = datavar(Matrix{Float64}, (:, n_features))  # Flexible rows, fixed features
    y = datavar(Vector{Int}, size(X, 1))           # Targets match row count
    
    # Priors
    weights ~ MvNormal(mean = zeros(n_features), cov = 100.0 * I)
    bias ~ Normal(mean = 0.0, var = 10.0)
    
    # Likelihood using manual element-wise operations
    for i in 1:length(y)
        logit = sum(X[i,j] * weights[j] for j in 1:n_features) + bias
        y[i] ~ Bernoulli(logistic(logit))
    end
end

function run_analysis(X, y)
    n_features = size(X, 2)
    
    # Create model with concrete dimensions
    model = bayesian_logistic_regression(n_features)
    
    # Perform inference
    result = infer(
        model = model,
        data = (X = X, y = y),
        iterations = 5,
        returnvars = (weights = KeepLast(), bias = KeepLast())
    )
    return result
end

function main()
    df = CSV.read("synthetic_classification_data.csv", DataFrame)
    X, y = preprocess_data(df)
    result = run_analysis(X, y)
    
    # Extract results
    weights = mean.(result.posteriors[:weights])
    bias = mean(result.posteriors[:bias])
    
    # Prepare output
    feature_names = names(df[:, [:Age, :Income, :Gender, :Education_code]])
    output = Dict(
        "bias" => bias,
        "weights" => Dict(zip(feature_names, weights))
    )
    
    # Save results
    JSON3.write("feature_weights.json", output)
    println("Analysis completed successfully!")
end

# Execute
main()