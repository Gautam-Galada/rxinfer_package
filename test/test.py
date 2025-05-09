import julia
julia.install()
from julia import Main
import pandas as pd
import numpy as np

# Pre-load Julia packages and modules
Main.eval('using DataFrames, Distributions, Plots, StatsBase')
Main.include("src/ImputationStrategies.jl")
Main.include("src/PlotUtils.jl")
Main.eval('using .ImputationStrategies, .PlotUtils')

def impute_missing(df: pd.DataFrame, strategy: str):
    clean_df = df.replace({np.nan: None})
    data = clean_df.to_dict(orient='list')
    jl_df = Main.DataFrame(data)
    strategy_enum = getattr(Main.ImputationStrategies, strategy.capitalize())
    imputed_dict = Main.ImputationStrategies.impute_missing(jl_df, strategy_enum)
    return pd.DataFrame({k: list(v) for k, v in imputed_dict.items()})

def plot_results(original_df: pd.DataFrame, imputed_df: pd.DataFrame):
    # Convert to Julia-compatible format
    original_data = original_df.astype(object).where(pd.notnull(original_df), None).to_dict(orient='list')
    imputed_data = imputed_df.astype(object).where(pd.notnull(imputed_df), None).to_dict(orient='list')
    
    # Create Julia DataFrames
    jl_original = Main.DataFrame(original_data)
    jl_imputed = Main.DataFrame(imputed_data)
    
    # Call Julia plotting function
    Main.PlotUtils.plot_imputation(jl_original, jl_imputed)  # Use Julia function
    
    from PIL import Image
    return Image.open("imputation_plot.png")

if __name__ == "__main__":
    df = pd.DataFrame({
        'age': [25, None, 30, None, 45],
        'income': [None, 50000, 75000, None, 120000]
    }).astype({'age': 'float64', 'income': 'float64'})
    
    benford_df = impute_missing(df, "Benford")
    print("Imputed Data:\n", benford_df)
    
    plot_image = plot_results(df, benford_df)
    plot_image.show()