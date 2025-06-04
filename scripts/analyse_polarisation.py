import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_weighted_std(group):
    """
    Calculates the vote-share weighted standard deviation of 'rile' scores.
    'pervote' is used as the weight.
    """
    # Drop rows where 'rile' or 'pervote' is NaN, or pervote is zero or less
    # as they cannot contribute to a weighted standard deviation.
    valid_data = group.dropna(subset=['rile', 'pervote'])
    valid_data = valid_data[valid_data['pervote'] > 0]

    # Need at least two parties with valid rile and positive vote share 
    # to calculate standard deviation.
    if len(valid_data) < 2:
        return np.nan

    # Ensure weights are not all zero (though filtered above)
    if valid_data['pervote'].sum() == 0:
        return np.nan

    try:
        # Using statsmodels DescrStatsW for weighted statistics
        weighted_stats = DescrStatsW(valid_data['rile'], weights=valid_data['pervote'], ddof=0) # ddof=0 for population-like std
        return weighted_stats.std
    except Exception as e:
        # print(f"Could not calculate weighted_std for group: {group.name}. Error: {e}")
        return np.nan

def main():
    """
    Main function to perform the analysis.
    """
    # --- 1. Load Data ---
    # Setup file paths (assuming script is in 'scripts', data in 'output' which are siblings)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    output_folder_name = "output" 
    output_dir = os.path.join(project_base_dir, output_folder_name)
    
    merged_data_filename = "merged_political_oecd_data.csv"
    merged_data_path = os.path.join(output_dir, merged_data_filename)

    if not os.path.exists(merged_data_path):
        print(f"Error: Merged data file not found at {merged_data_path}")
        print("Please ensure you have run the data merging script first and the file is in the 'output' directory.")
        return

    print(f"Loading merged data from: {merged_data_path}")
    # Added low_memory=False to address DtypeWarning and help with type inference.
    df = pd.read_csv(merged_data_path, low_memory=False)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

    # --- Debugging oecdmember column ---
    if 'oecdmember' in df.columns:
        print("\n--- Debugging 'oecdmember' column ---")
        print(f"Data type of 'oecdmember' column: {df['oecdmember'].dtype}")
        print(f"Unique values in 'oecdmember' column: {df['oecdmember'].unique()}")
        print("Value counts for 'oecdmember':")
        print(df['oecdmember'].value_counts(dropna=False)) # include NaNs in counts
        print("First 5 values of 'oecdmember' column:")
        print(df['oecdmember'].head())
        print("-------------------------------------\n")
    else:
        print("Error: 'oecdmember' column not found in the loaded dataframe. Cannot proceed with OECD filtering.")
        return
        
    # --- 2. Filter for OECD Countries ---
    # The 'oecdmember' column should indicate OECD membership.
    # Based on the debugging output, OECD members are coded as 10.
    # Convert to numeric, coercing errors, then check for 10.
    df['oecdmember_numeric'] = pd.to_numeric(df['oecdmember'], errors='coerce')
    # Corrected filtering condition to check for 10
    df_oecd = df[df['oecdmember_numeric'] == 10].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    print(f"Filtered for OECD countries (where 'oecdmember_numeric' == 10): {df_oecd.shape[0]} rows remaining.")
    if df_oecd.empty:
        print("No OECD member data found after filtering with 'oecdmember_numeric' == 10.")
        # If still empty, provide more info from the original df before filtering
        if 'oecdmember' in df.columns:
             print("Re-checking original 'oecdmember' column unique values before numeric conversion:")
             print(f"Unique values: {df['oecdmember'].unique()}")
             print(f"Data type: {df['oecdmember'].dtype}")
        return

    # --- 3. Calculate Political Polarization Score ---
    print("\nCalculating Political Polarization Score (vote-share weighted std dev of rile)...")
    
    # Group by country and year, then apply the function
    # This calculates polarization for each country-year combination.
    polarization_scores = df_oecd.groupby(['countryname', 'year']).apply(calculate_weighted_std)
    polarization_df = polarization_scores.reset_index(name='PolarizationScore')
    
    print("Polarization scores calculated. Sample:")
    print(polarization_df.head())
    print(f"Number of country-years with polarization scores: {len(polarization_df.dropna(subset=['PolarizationScore']))}")


    # --- 4. Prepare Final Dataset (Country-Year Level) ---
    # We need GINI and CPI at the country-year level.
    # Since GINI and CPI were merged from OECD data, they should be unique per country-year in the merged_df.
    # We can aggregate the original df_oecd (or df) to get these values or merge polarization_df back.
    
    # Select relevant columns and drop duplicates at country-year level for GINI and CPI
    country_year_oecd_data = df_oecd[['countryname', 'year', 'GINI', 'CPI']].drop_duplicates(subset=['countryname', 'year'])
    
    # Merge polarization scores with GINI and CPI
    analysis_df = pd.merge(country_year_oecd_data, polarization_df, on=['countryname', 'year'], how='left')
    
    print("\nFinal country-year level dataset for analysis (before dropping NaNs for analysis):")
    print(analysis_df.head())
    print(f"Shape of analysis_df: {analysis_df.shape}")

    # --- 5. Handle Missing Data for Analysis ---
    # For correlation and regression, we need complete cases for the variables involved.
    analysis_df_complete = analysis_df.dropna(subset=['PolarizationScore', 'GINI', 'CPI'])
    print(f"\nShape of analysis_df after dropping NaNs in PolarizationScore, GINI, CPI: {analysis_df_complete.shape}")
    
    if analysis_df_complete.empty:
        print("No complete data rows available for analysis after handling missing values. Exiting.")
        return
    if len(analysis_df_complete) < 2:
        print("Too few complete data rows (<2) available for correlation/regression. Exiting.")
        return

    # --- 6. Correlation Analysis ---
    print("\n--- Correlation Analysis ---")
    correlation_matrix = analysis_df_complete[['PolarizationScore', 'GINI', 'CPI']].corr(method='pearson')
    print("Pearson Correlation Matrix:")
    print(correlation_matrix)

    # Visualize relationships using scatter plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=analysis_df_complete, x='GINI', y='PolarizationScore')
    plt.title('Polarization Score vs. GINI Coefficient')
    plt.xlabel('GINI Coefficient (Income Inequality)')
    plt.ylabel('Political Polarization Score')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=analysis_df_complete, x='CPI', y='PolarizationScore')
    plt.title('Polarization Score vs. CPI')
    plt.xlabel('Consumer Price Index (CPI)')
    plt.ylabel('Political Polarization Score')
    plt.grid(True)

    plt.tight_layout()
    # Save the figure
    correlation_plot_path = os.path.join(output_dir, "correlation_plots.png")
    try:
        plt.savefig(correlation_plot_path)
        print(f"\nCorrelation plots saved to: {correlation_plot_path}")
    except Exception as e:
        print(f"Could not save correlation plots: {e}")
    plt.show()


    # --- 7. Regression Analysis ---
    print("\n--- Regression Analysis ---")

    # Ensure no NaN/inf values in the subset used for regression
    regression_data = analysis_df_complete[['PolarizationScore', 'GINI', 'CPI']].copy()
    regression_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    regression_data.dropna(inplace=True)

    if regression_data.empty or len(regression_data) < 2 : # Need at least 2 points for regression
        print("Not enough valid data points for regression after handling inf/NaN. Skipping regression.")
        return

    # Simple Linear Regression: PolarizationScore ~ GINI
    print("\n1. Simple Linear Regression: PolarizationScore ~ GINI")
    try:
        model_gini = smf.ols('PolarizationScore ~ GINI', data=regression_data).fit()
        print(model_gini.summary())
        
        # Plotting regression for GINI
        plt.figure(figsize=(8, 6))
        sns.regplot(x='GINI', y='PolarizationScore', data=regression_data, ci=95, line_kws={'color':'red'})
        plt.title('Regression: Polarization Score vs. GINI')
        plt.xlabel('GINI Coefficient')
        plt.ylabel('Political Polarization Score')
        plt.grid(True)
        reg_gini_plot_path = os.path.join(output_dir, "regression_gini_plot.png")
        try:
            plt.savefig(reg_gini_plot_path)
            print(f"GINI regression plot saved to: {reg_gini_plot_path}")
        except Exception as e:
            print(f"Could not save GINI regression plot: {e}")
        plt.show()

    except Exception as e:
        print(f"Error during GINI regression: {e}")


    # Simple Linear Regression: PolarizationScore ~ CPI
    print("\n2. Simple Linear Regression: PolarizationScore ~ CPI")
    try:
        model_cpi = smf.ols('PolarizationScore ~ CPI', data=regression_data).fit()
        print(model_cpi.summary())

        # Plotting regression for CPI
        plt.figure(figsize=(8, 6))
        sns.regplot(x='CPI', y='PolarizationScore', data=regression_data, ci=95, line_kws={'color':'red'})
        plt.title('Regression: Polarization Score vs. CPI')
        plt.xlabel('Consumer Price Index (CPI)')
        plt.ylabel('Political Polarization Score')
        plt.grid(True)
        reg_cpi_plot_path = os.path.join(output_dir, "regression_cpi_plot.png")
        try:
            plt.savefig(reg_cpi_plot_path)
            print(f"CPI regression plot saved to: {reg_cpi_plot_path}")
        except Exception as e:
            print(f"Could not save CPI regression plot: {e}")
        plt.show()

    except Exception as e:
        print(f"Error during CPI regression: {e}")


    # Multiple Linear Regression: PolarizationScore ~ GINI + CPI
    print("\n3. Multiple Linear Regression: PolarizationScore ~ GINI + CPI")
    try:
        # Check if both GINI and CPI have enough variance and are not perfectly collinear
        # (though statsmodels handles perfect collinearity by dropping one variable)
        if regression_data['GINI'].nunique() > 1 and regression_data['CPI'].nunique() > 1:
            model_multiple = smf.ols('PolarizationScore ~ GINI + CPI', data=regression_data).fit()
            print(model_multiple.summary())
        else:
            print("Skipping multiple regression due to insufficient variance in GINI or CPI.")
            
    except Exception as e:
        print(f"Error during Multiple regression: {e}")
        
    print("\nAnalysis complete.")
    # Creating .csv file for PowerBI analysis
    analysis_df_complete.to_csv(os.path.join(output_dir, "analysis_ready_data.csv"), index=False)
    print(f"\nSaved analysis-ready data (102 observations) to: {os.path.join(output_dir, 'analysis_ready_data.csv')}")

if __name__ == '__main__':
    main()
