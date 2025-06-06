import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

def calculate_weighted_std(group):
    """
    Calculates the vote-share weighted standard deviation of 'rile' scores.
    'pervote' is used as the weight.
    """
    valid_data = group.dropna(subset=['rile', 'pervote'])
    valid_data['rile'] = pd.to_numeric(valid_data['rile'], errors='coerce')
    valid_data['pervote'] = pd.to_numeric(valid_data['pervote'], errors='coerce')
    valid_data = valid_data.dropna(subset=['rile', 'pervote'])
    valid_data = valid_data[valid_data['pervote'] > 0]

    if len(valid_data) < 2:
        return np.nan
    if valid_data['pervote'].sum() == 0:
        return np.nan
    try:
        weighted_stats = DescrStatsW(valid_data['rile'], weights=valid_data['pervote'], ddof=0)
        return weighted_stats.std
    except Exception:
        return np.nan

def create_lagged_features(df, group_cols, target_cols, lags=[1]):
    """
    Creates lagged features for specified columns, grouped by group_cols.
    """
    df_lagged = df.copy()
    for lag in lags:
        for col in target_cols:
            new_col_name = f"{col}_lag{lag}"
            df_lagged[new_col_name] = df_lagged.groupby(group_cols)[col].shift(lag)
    return df_lagged

def main():
    """
    Main function to perform the analysis.
    """
    # --- 1. Load Data ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    output_folder_name = "output" 
    output_dir = os.path.join(project_base_dir, output_folder_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    merged_data_filename = "merged_political_oecd_data.csv"
    merged_data_path = os.path.join(output_dir, merged_data_filename)

    if not os.path.exists(merged_data_path):
        print(f"Error: Merged data file not found at {merged_data_path}")
        return
    print(f"Loading merged data from: {merged_data_path}")
    df = pd.read_csv(merged_data_path, low_memory=False)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

    if 'oecdmember' not in df.columns:
        print("Error: 'oecdmember' column not found. Cannot proceed.")
        return
    
    df['oecdmember_numeric'] = pd.to_numeric(df['oecdmember'], errors='coerce')
    df_oecd = df[df['oecdmember_numeric'] == 10].copy()
    print(f"\nFiltered for OECD countries ('oecdmember_numeric' == 10): {df_oecd.shape[0]} rows remaining.")
    if df_oecd.empty:
        print("No OECD member data found after filtering. Exiting.")
        return

    print("\nCalculating Political Polarization Score...")
    polarization_scores = df_oecd.groupby(['countryname', 'year']).apply(calculate_weighted_std)
    polarization_df = polarization_scores.reset_index(name='PolarizationScore')
    print(f"Polarization scores calculated. {len(polarization_df.dropna(subset=['PolarizationScore']))} country-years with scores.")

    country_year_oecd_data = df_oecd[['countryname', 'year', 'GINI', 'CPI']].drop_duplicates(subset=['countryname', 'year'])
    analysis_df = pd.merge(country_year_oecd_data, polarization_df, on=['countryname', 'year'], how='left')

    print("\nCreating lagged features for GINI and CPI (1-year lag)...")
    analysis_df = analysis_df.sort_values(by=['countryname', 'year'])
    analysis_df_lagged = create_lagged_features(analysis_df, 
                                                group_cols=['countryname'], 
                                                target_cols=['GINI', 'CPI'], 
                                                lags=[1])

    power_bi_data_filename = "country_year_full_analysis_data.csv"
    power_bi_data_path = os.path.join(output_dir, power_bi_data_filename)
    try:
        analysis_df_lagged.to_csv(power_bi_data_path, index=False)
        print(f"\nComprehensive data for Power BI saved to: {power_bi_data_path}")
    except Exception as e:
        print(f"Error saving data for Power BI: {e}")

    cols_for_current_analysis = ['PolarizationScore', 'GINI', 'CPI']
    cols_for_lagged_analysis = ['PolarizationScore', 'GINI_lag1', 'CPI_lag1']
    
    analysis_df_current_complete = analysis_df_lagged.dropna(subset=cols_for_current_analysis)
    print(f"\nShape for current vars OLS/Pooled analysis: {analysis_df_current_complete.shape}")

    analysis_df_lagged_complete = analysis_df_lagged.dropna(subset=cols_for_lagged_analysis)
    print(f"Shape for lagged vars OLS/Pooled analysis: {analysis_df_lagged_complete.shape}")
    
    # --- Pooled OLS: Correlation Analysis (Current and Lagged) ---
    if not analysis_df_current_complete.empty and len(analysis_df_current_complete) >= 2:
        print("\n--- Pooled OLS: Correlation Analysis (Current Variables) ---")
        correlation_matrix_current = analysis_df_current_complete[cols_for_current_analysis].corr(method='pearson')
        print("Pearson Correlation Matrix (Current):")
        print(correlation_matrix_current)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); sns.scatterplot(data=analysis_df_current_complete, x='GINI', y='PolarizationScore'); plt.title('Polarization Score vs. Current GINI'); plt.grid(True)
        plt.subplot(1, 2, 2); sns.scatterplot(data=analysis_df_current_complete, x='CPI', y='PolarizationScore'); plt.title('Polarization Score vs. Current CPI'); plt.grid(True)
        plt.tight_layout()
        corr_current_plot_path = os.path.join(output_dir, "correlation_plots_current.png")
        try: plt.savefig(corr_current_plot_path); print(f"Current correlation plots saved to: {corr_current_plot_path}")
        except Exception as e: print(f"Could not save current correlation plots: {e}")
        plt.show()

    if not analysis_df_lagged_complete.empty and len(analysis_df_lagged_complete) >= 2:
        print("\n--- Pooled OLS: Correlation Analysis (Lagged Variables) ---")
        cols_for_lagged_corr_matrix = ['PolarizationScore', 'GINI', 'CPI', 'GINI_lag1', 'CPI_lag1']
        cols_present_for_lagged_corr = [col for col in cols_for_lagged_corr_matrix if col in analysis_df_lagged_complete.columns]
        correlation_matrix_lagged = analysis_df_lagged_complete[cols_present_for_lagged_corr].corr(method='pearson')
        print("Pearson Correlation Matrix (Including Lags where available):")
        print(correlation_matrix_lagged)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); sns.scatterplot(data=analysis_df_lagged_complete, x='GINI_lag1', y='PolarizationScore'); plt.title('Polarization Score vs. GINI (Lagged 1 Year)'); plt.grid(True)
        plt.subplot(1, 2, 2); sns.scatterplot(data=analysis_df_lagged_complete, x='CPI_lag1', y='PolarizationScore'); plt.title('Polarization Score vs. CPI (Lagged 1 Year)'); plt.grid(True)
        plt.tight_layout()
        corr_lagged_plot_path = os.path.join(output_dir, "correlation_plots_lagged.png")
        try: plt.savefig(corr_lagged_plot_path); print(f"Lagged correlation plots saved to: {corr_lagged_plot_path}")
        except Exception as e: print(f"Could not save lagged correlation plots: {e}")
        plt.show()

    # --- Pooled OLS: Regression Analysis (Current and Lagged) ---
    if not analysis_df_current_complete.empty and len(analysis_df_current_complete) >= 2:
        print("\n--- Pooled OLS: Regression Analysis (Current Variables) ---")
        regression_data_current = analysis_df_current_complete[cols_for_current_analysis].copy()
        regression_data_current.replace([np.inf, -np.inf], np.nan, inplace=True); regression_data_current.dropna(inplace=True)

        if not regression_data_current.empty and len(regression_data_current) >=2:
            print(f"Observations for current OLS regression: {len(regression_data_current)}")
            try:
                print("\nOLS Model 1a: PolarizationScore ~ GINI")
                model_gini_current = smf.ols('PolarizationScore ~ GINI', data=regression_data_current).fit()
                print(model_gini_current.summary())
                fig, ax = plt.subplots(figsize=(8,6)); sns.regplot(x='GINI', y='PolarizationScore', data=regression_data_current, ax=ax, ci=95, line_kws={'color':'red'}); ax.set_title('OLS: Polarization Score vs. Current GINI'); plt.grid(True); fig.tight_layout()
                path = os.path.join(output_dir, "ols_regression_gini_current.png"); plt.savefig(path); plt.show(); print(f"Plot saved: {path}")
            except Exception as e: print(f"Error in OLS GINI current regression: {e}")
            
            try:
                print("\nOLS Model 2a: PolarizationScore ~ CPI")
                model_cpi_current = smf.ols('PolarizationScore ~ CPI', data=regression_data_current).fit()
                print(model_cpi_current.summary())
                fig, ax = plt.subplots(figsize=(8,6)); sns.regplot(x='CPI', y='PolarizationScore', data=regression_data_current, ax=ax, ci=95, line_kws={'color':'red'}); ax.set_title('OLS: Polarization Score vs. Current CPI'); plt.grid(True); fig.tight_layout()
                path = os.path.join(output_dir, "ols_regression_cpi_current.png"); plt.savefig(path); plt.show(); print(f"Plot saved: {path}")
            except Exception as e: print(f"Error in OLS CPI current regression: {e}")

            try:
                print("\nOLS Model 3a: PolarizationScore ~ GINI + CPI")
                if regression_data_current['GINI'].nunique() > 1 and regression_data_current['CPI'].nunique() > 1:
                    model_multiple_current = smf.ols('PolarizationScore ~ GINI + CPI', data=regression_data_current).fit()
                    print(model_multiple_current.summary())
                else: print("Skipping current OLS multiple regression due to insufficient variance.")
            except Exception as e: print(f"Error in OLS Multiple current regression: {e}")
        else: print("Not enough data for current OLS regressions after final cleaning.")
            
    if not analysis_df_lagged_complete.empty and len(analysis_df_lagged_complete) >= 2:
        print("\n--- Pooled OLS: Regression Analysis (Lagged Variables) ---")
        regression_data_lagged = analysis_df_lagged_complete[['PolarizationScore', 'GINI_lag1', 'CPI_lag1']].copy()
        regression_data_lagged.replace([np.inf, -np.inf], np.nan, inplace=True); regression_data_lagged.dropna(inplace=True)

        if not regression_data_lagged.empty and len(regression_data_lagged) >=2:
            print(f"Observations for lagged OLS regression: {len(regression_data_lagged)}")
            try:
                print("\nOLS Model 1b: PolarizationScore ~ GINI_lag1")
                model_gini_lagged = smf.ols('PolarizationScore ~ GINI_lag1', data=regression_data_lagged).fit()
                print(model_gini_lagged.summary())
                fig, ax = plt.subplots(figsize=(8,6)); sns.regplot(x='GINI_lag1', y='PolarizationScore', data=regression_data_lagged, ax=ax, ci=95, line_kws={'color':'blue'}); ax.set_title('OLS: Polarization Score vs. GINI (Lagged 1 Year)'); plt.grid(True); fig.tight_layout()
                path = os.path.join(output_dir, "ols_regression_gini_lagged.png"); plt.savefig(path); plt.show(); print(f"Plot saved: {path}")
            except Exception as e: print(f"Error in OLS GINI_lag1 regression: {e}")

            try:
                print("\nOLS Model 2b: PolarizationScore ~ CPI_lag1")
                model_cpi_lagged = smf.ols('PolarizationScore ~ CPI_lag1', data=regression_data_lagged).fit()
                print(model_cpi_lagged.summary())
                fig, ax = plt.subplots(figsize=(8,6)); sns.regplot(x='CPI_lag1', y='PolarizationScore', data=regression_data_lagged, ax=ax, ci=95, line_kws={'color':'blue'}); ax.set_title('OLS: Polarization Score vs. CPI (Lagged 1 Year)'); plt.grid(True); fig.tight_layout()
                path = os.path.join(output_dir, "ols_regression_cpi_lagged.png"); plt.savefig(path); plt.show(); print(f"Plot saved: {path}")
            except Exception as e: print(f"Error in OLS CPI_lag1 regression: {e}")

            try:
                print("\nOLS Model 3b: PolarizationScore ~ GINI_lag1 + CPI_lag1")
                if regression_data_lagged['GINI_lag1'].nunique() > 1 and regression_data_lagged['CPI_lag1'].nunique() > 1:
                    model_multiple_lagged = smf.ols('PolarizationScore ~ GINI_lag1 + CPI_lag1', data=regression_data_lagged).fit()
                    print(model_multiple_lagged.summary())
                else: print("Skipping OLS lagged multiple regression due to insufficient variance.")
            except Exception as e: print(f"Error in OLS Multiple lagged regression: {e}")
        else: print("Not enough data for OLS lagged regressions after final cleaning.")

    # --- Panel Data Regression (Fixed Effects) ---
    print("\n--- Panel Data Regression (Country Fixed Effects) ---")
    print("Note: You may need to install 'linearmodels' (pip install linearmodels)")

    panel_data = analysis_df_lagged.set_index(['countryname', 'year'])

    print("\nPanel Model 1: PolarizationScore ~ GINI + CPI (Country Fixed Effects)")
    try:
        model1_data = panel_data[['PolarizationScore', 'GINI', 'CPI']].dropna()
        if len(model1_data) > (model1_data.index.get_level_values('countryname').nunique() + 2):
            formula1 = 'PolarizationScore ~ 1 + GINI + CPI + EntityEffects'
            mod1 = PanelOLS.from_formula(formula1, data=model1_data)
            results1 = mod1.fit(cov_type='robust') # Using robust standard errors
            print(results1)
        else: print("Not enough observations for Panel Model 1.")
    except Exception as e: print(f"Error in Panel Model 1: {e}")

    print("\nPanel Model 2: PolarizationScore ~ GINI_lag1 + CPI_lag1 (Country Fixed Effects)")
    try:
        model2_data = panel_data[['PolarizationScore', 'GINI_lag1', 'CPI_lag1']].dropna()
        if len(model2_data) > (model2_data.index.get_level_values('countryname').nunique() + 2):
            formula2 = 'PolarizationScore ~ 1 + GINI_lag1 + CPI_lag1 + EntityEffects'
            mod2 = PanelOLS.from_formula(formula2, data=model2_data)
            results2 = mod2.fit(cov_type='robust') # Using robust standard errors
            print(results2)
        else: print("Not enough observations for Panel Model 2.")
    except Exception as e: print(f"Error in Panel Model 2: {e}")

    print("\nPanel Model 3: PolarizationScore ~ CPI + CPI_lag1 (Country Fixed Effects)")
    try:
        model3_data = panel_data[['PolarizationScore', 'CPI', 'CPI_lag1']].dropna()
        if len(model3_data) > (model3_data.index.get_level_values('countryname').nunique() + 2):
            formula3 = 'PolarizationScore ~ 1 + CPI + CPI_lag1 + EntityEffects'
            mod3 = PanelOLS.from_formula(formula3, data=model3_data)
            results3 = mod3.fit(cov_type='robust') # Using robust standard errors
            print(results3)
        else: print("Not enough observations for Panel Model 3.")
    except Exception as e: print(f"Error in Panel Model 3: {e}")
        
    print("\nPanel Model 4: PolarizationScore ~ GINI + CPI + GINI_lag1 + CPI_lag1 (Country Fixed Effects)")
    try:
        model4_data = panel_data[['PolarizationScore', 'GINI', 'CPI', 'GINI_lag1', 'CPI_lag1']].dropna()
        if len(model4_data) > (model4_data.index.get_level_values('countryname').nunique() + 4): 
            formula4 = 'PolarizationScore ~ 1 + GINI + CPI + GINI_lag1 + CPI_lag1 + EntityEffects'
            mod4 = PanelOLS.from_formula(formula4, data=model4_data)
            results4 = mod4.fit(cov_type='robust') # Using robust standard errors
            print(results4)
        else: print("Not enough observations for Panel Model 4.")
    except Exception as e: print(f"Error in Panel Model 4: {e}")

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
