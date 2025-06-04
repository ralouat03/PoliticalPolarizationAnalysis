import pandas as pd
import sqlite3
import os

# Configuration for file paths

script_dir = os.path.dirname(os.path.abspath(__file__))
project_base_dir = os.path.dirname(script_dir)

# Define the names of your data and output folders
data_folder_name = "data"
output_folder_name = "output" # Or you could use data_folder_name if you prefer outputs there

# Construct paths to the data and output folders
data_dir = os.path.join(project_base_dir, data_folder_name)
output_dir = os.path.join(project_base_dir, output_folder_name)

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Define input file names
mp_dataset_filename = "MPDataset_MPDS2024a.csv"
oecd_dataset_filename = "OECD_gini_cpi_data.csv"

# Define output file names
sqlite_db_name = "merged_data.sqlite"
csv_output_filename = "merged_political_oecd_data.csv"

# Construct full paths to input and output files
mp_dataset_path = os.path.join(data_dir, mp_dataset_filename)
oecd_dataset_path = os.path.join(data_dir, oecd_dataset_filename)

sqlite_db_path = os.path.join(output_dir, sqlite_db_name)
csv_output_path = os.path.join(output_dir, csv_output_filename)


# Load the datasets
try:
    print(f"Loading MP Dataset from: {mp_dataset_path}")
    df_mp = pd.read_csv(mp_dataset_path)
    print(f"Loading OECD Dataset from: {oecd_dataset_path}")
    df_oecd = pd.read_csv(oecd_dataset_path)

    # Preparing MP Dataset
    print("\n--- MP Dataset (Initial) ---")
    print(df_mp[['date', 'countryname']].head(2))
    print(f"Initial MP rows: {len(df_mp)}")
    
    df_mp['year'] = df_mp['date'] // 100
    
    print("\n--- MP Dataset (with 'year' column added) ---")
    print(df_mp[['date', 'year', 'countryname']].head(2))
    print(f"Data type of 'year' in df_mp: {df_mp['year'].dtype}")

    # Preparing OECD Dataset
    print("\n--- OECD Dataset (Initial) ---")
    print(df_oecd[['Reference area', 'TIME_PERIOD', 'MEASURE', 'OBS_VALUE']].head(2))
    print(f"Initial OECD rows: {len(df_oecd)}")
        
    df_oecd_selected = df_oecd[['Reference area', 'TIME_PERIOD', 'MEASURE', 'OBS_VALUE']].copy()
    
    df_oecd_renamed = df_oecd_selected.rename(columns={
        'Reference area': 'countryname',
        'TIME_PERIOD': 'year', 
        'MEASURE': 'variable',
        'OBS_VALUE': 'value'
    })

    print("\n--- OECD Dataset After Selection and Rename ---")
    print(df_oecd_renamed.head(2))
    print(f"Data type of 'year' in df_oecd_renamed: {df_oecd_renamed['year'].dtype}")
    
    df_oecd_pivot = df_oecd_renamed.pivot_table(
        index=['countryname', 'year'], 
        columns='variable', 
        values='value'
    ).reset_index()
    
    if 'INC_DISP_GINI' in df_oecd_pivot.columns:
        df_oecd_pivot.rename(columns={'INC_DISP_GINI': 'GINI'}, inplace=True)
    
    print("\n--- OECD Dataset After Pivot and Renaming GINI column ---")
    print(df_oecd_pivot.head(2))
    print(f"Columns in pivoted OECD data: {df_oecd_pivot.columns.tolist()}")
    print(f"Data type of 'year' in df_oecd_pivot: {df_oecd_pivot['year'].dtype}")

    # Merging Datasets
    merged_df = pd.merge(df_mp, df_oecd_pivot, on=['countryname', 'year'], how='left')

    print("\n--- Merged Dataset ---")
    print(merged_df[['countryname', 'year', 'partyname', 'rile', 'CPI', 'GINI']].head())
    print(f"\nTotal rows in merged_df: {len(merged_df)}")
    if 'GINI' in merged_df.columns:
        print(f"Number of rows with GINI data: {merged_df['GINI'].notna().sum()}")
    if 'CPI' in merged_df.columns:
        print(f"Number of rows with CPI data: {merged_df['CPI'].notna().sum()}")
    
    # Storing Merged Data
    table_name = "merged_political_oecd_data" 
    
    print(f"\nConnecting to SQLite database: {sqlite_db_path}")
    conn = sqlite3.connect(sqlite_db_path)
    merged_df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Successfully merged data and stored in SQLite database '{sqlite_db_path}' in table '{table_name}'")
    
    print(f"\nSaving merged data to CSV file: {csv_output_path}")
    merged_df.to_csv(csv_output_path, index=False)
    print(f"Merged data also saved to CSV file: '{csv_output_path}'")

except FileNotFoundError as e:
    print(f"Error: A CSV file was not found. Please check the file paths and names.")
    print(f"Details: {e}")
    print(f"Attempted to load MP data from: {mp_dataset_path}")
    print(f"Attempted to load OECD data from: {oecd_dataset_path}")
except KeyError as e:
    print(f"KeyError during processing: {e}. This often means a column name expected in the script was not found in the CSV files, or there was an issue with merge keys.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")