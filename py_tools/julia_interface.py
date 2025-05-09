


import julia
import pandas as pd
import numpy as np
julia.install()  # Critical first step
from julia import Main


# Initialize with compiled modules OFF
julia.Julia(compiled_modules=False)

# Now load your modules
Main.include("src/MissingDataImputer.jl")
Main.include("src/ImputationStrategies.jl")
Main.include("src/PlotUtils.jl")

class JuliaImputer:
    def __init__(self):
        self.Strategies = Main.Standard
        self.PlotUtils = Main.PlotUtils
    
    def impute_missing(df: pd.DataFrame, strategy: str):
        """Python-friendly wrapper for Julia imputation"""
        clean_df = df.replace({np.nan: None})
        data = clean_df.to_dict(orient='records')
        jl_df = Main.DataFrame(data)
        
        strategy_enum = getattr(Main, strategy.capitalize())
        imputed_jl = Main.impute_missing(jl_df, strategy_enum)
        
        # Convert Julia DataFrame to pandas using Tables.jl interface
        return pd.DataFrame({
            col: [x if not Main.ismissing(x) else None 
                for x in Main.getproperty(imputed_jl, col)]
            for col in df.columns
        })
    
    def plot_results(self, original_df: pd.DataFrame, imputed_df: pd.DataFrame):
        """Wrapper for Julia plotting"""
        jl_original = Main.DataFrame(original_df)
        jl_imputed = Main.DataFrame(imputed_df)
        self.PlotUtils.plot_imputation(jl_original, jl_imputed)