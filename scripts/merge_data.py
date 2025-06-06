import pandas as pd
import sqlite3
import os

def main():
    # --- Configuration for file paths ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_base_dir = os.path.dirname(script_dir)
    except NameError:
        project_base_dir = os.getcwd()
        print(f"Warning: __file__ not defined, using current working directory as project base: {project_base_dir}")

    data_folder_name = "data"
    output_folder_name = "output"
    
    data_dir = os.path.join(project_base_dir, data_folder_name)
    output_dir = os.path.join(project_base_dir, output_folder_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Define input file names
    mp_dataset_filename = "MPDataset_MPDS2024a.csv"
    worldbank_gini_filename = "worldbank_gini_data.csv"
    oecd_cpi_filename = "OECD_cpi_data.csv"

    # Define output file names
    sqlite_db_name = "merged_data.sqlite"
    csv_output_filename = "merged_political_oecd_data.csv"

    mp_dataset_path = os.path.join(data_dir, mp_dataset_filename)
    worldbank_gini_path = os.path.join(data_dir, worldbank_gini_filename)
    oecd_cpi_path = os.path.join(data_dir, oecd_cpi_filename)
    
    sqlite_db_path = os.path.join(output_dir, sqlite_db_name)
    csv_output_path = os.path.join(output_dir, csv_output_filename)

    try:
        # --- 1. Load and Prepare Manifesto Project Data ---
        print(f"Loading Manifesto Project Dataset from: {mp_dataset_path}")
        df_mp = pd.read_csv(mp_dataset_path, low_memory=False)
        print(f"Initial MP rows: {len(df_mp)}")
        df_mp['year'] = df_mp['date'] // 100 # Extract year
        print("Manifesto data loaded and 'year' column created.")

        # --- 2. Load and Prepare World Bank GINI Data ---
        print(f"\nLoading World Bank GINI Data from: {worldbank_gini_path}")
        df_gini_wb = pd.read_csv(worldbank_gini_path, skiprows=4)
        print(f"Initial World Bank GINI rows: {len(df_gini_wb)}")
        
        # Keep only relevant columns: Country Name and year columns
        # Identify year columns (those that can be converted to numeric, typically from 1960 onwards)
        year_cols = [col for col in df_gini_wb.columns if col.isdigit() and 1900 <= int(col) <= 2100]
        if not year_cols:
            raise ValueError("No year columns found in World Bank GINI data. Check column names.")
            
        df_gini_wb_selected = df_gini_wb[['Country Name'] + year_cols]
        
        # Melt the DataFrame to long format
        df_gini_long = df_gini_wb_selected.melt(
            id_vars=['Country Name'],
            value_vars=year_cols,
            var_name='year',
            value_name='GINI'
        )
        
        df_gini_long.rename(columns={'Country Name': 'countryname'}, inplace=True)
        df_gini_long['year'] = pd.to_numeric(df_gini_long['year'], errors='coerce').astype('Int64') # Keep as nullable Int
        df_gini_long['GINI'] = pd.to_numeric(df_gini_long['GINI'], errors='coerce')
        
        # Drop rows where GINI is NaN after melting, as these represent years without data for a country
        df_gini_processed = df_gini_long.dropna(subset=['GINI', 'year'])
        print(f"Processed World Bank GINI data. Shape: {df_gini_processed.shape}")
        print("Sample of processed GINI data:")
        print(df_gini_processed.head())

        # --- 3. Load and Prepare OECD CPI Data ---
        print(f"\nLoading OECD CPI Data from: {oecd_cpi_path}")
        df_cpi_oecd = pd.read_csv(oecd_cpi_path, low_memory=False)
        print(f"Initial OECD CPI rows: {len(df_cpi_oecd)}")

        # Filter for relevant CPI data if necessary
        # Based on inspection, 'MEASURE' == 'CPI' and 'FREQUENCY'=='A' (Annual) are good filters
        # Check if 'FREQUENCY' column exists, otherwise adapt
        if 'FREQUENCY' in df_cpi_oecd.columns:
            df_cpi_filtered = df_cpi_oecd[
                (df_cpi_oecd['MEASURE'] == 'CPI') & 
                (df_cpi_oecd['FREQUENCY'] == 'A') # Assuming 'A' is for Annual
            ]
        elif 'Frequency of observation' in df_cpi_oecd.columns: # Fallback to verbose name
             df_cpi_filtered = df_cpi_oecd[
                (df_cpi_oecd['MEASURE'] == 'CPI') & 
                (df_cpi_oecd['Frequency of observation'] == 'Annual')
            ]
        else: # If no frequency column, just filter by MEASURE
            print("Warning: Frequency column not found in OECD CPI data. Filtering only by MEASURE='CPI'.")
            df_cpi_filtered = df_cpi_oecd[df_cpi_oecd['MEASURE'] == 'CPI']

        if df_cpi_filtered.empty:
            print("Warning: No CPI data found after filtering in OECD CPI dataset. CPI will be all NaN.")
            df_cpi_processed = pd.DataFrame(columns=['countryname', 'year', 'CPI'])
        else:
            df_cpi_processed = df_cpi_filtered[['Reference area', 'TIME_PERIOD', 'OBS_VALUE']].copy()
            df_cpi_processed.rename(columns={
                'Reference area': 'countryname',
                'TIME_PERIOD': 'year',
                'OBS_VALUE': 'CPI'
            }, inplace=True)
            df_cpi_processed['year'] = pd.to_numeric(df_cpi_processed['year'], errors='coerce').astype('Int64')
            df_cpi_processed['CPI'] = pd.to_numeric(df_cpi_processed['CPI'], errors='coerce')
            df_cpi_processed = df_cpi_processed.dropna(subset=['CPI', 'year'])
        
        print(f"Processed OECD CPI data. Shape: {df_cpi_processed.shape}")
        print("Sample of processed CPI data:")
        print(df_cpi_processed.head())

        # --- 4. Merge Processed GINI and CPI Data ---
        print("\nMerging processed GINI and CPI data...")
        if df_gini_processed.empty and df_cpi_processed.empty:
            print("Both GINI and CPI processed data are empty. Merged economic data will be empty.")
            df_economic = pd.DataFrame(columns=['countryname', 'year', 'GINI', 'CPI'])
        elif df_gini_processed.empty:
            print("GINI data is empty. Merged economic data will only contain CPI.")
            df_economic = df_cpi_processed
        elif df_cpi_processed.empty:
            print("CPI data is empty. Merged economic data will only contain GINI.")
            df_economic = df_gini_processed
        else:
            df_economic = pd.merge(df_gini_processed, df_cpi_processed, on=['countryname', 'year'], how='outer')
        
        print(f"Merged economic data (GINI & CPI). Shape: {df_economic.shape}")
        print("Sample of merged economic data:")
        print(df_economic.head())
        # Handle potential duplicate country-year entries if any source had them (though processing should minimize this)
        df_economic = df_economic.groupby(['countryname', 'year']).agg({'GINI':'first', 'CPI':'first'}).reset_index()
        print(f"Shape of economic data after ensuring unique country-year: {df_economic.shape}")


        # --- 5. Merge Economic Data with Manifesto Data ---
        print("\nMerging economic data with Manifesto Project data...")
        # Ensure 'year' in df_mp is also Int64 for consistent merge, or both are int64 without NaNs
        df_mp['year'] = df_mp['year'].astype('Int64')
        
        merged_df = pd.merge(df_mp, df_economic, on=['countryname', 'year'], how='left')
        print(f"Final merged DataFrame shape: {merged_df.shape}")
        print("Sample of final merged data (selected columns):")
        print(merged_df[['countryname', 'year', 'partyname', 'rile', 'GINI', 'CPI']].head())
        
        # Check how many rows have GINI and CPI data
        if 'GINI' in merged_df.columns:
            print(f"Number of rows with GINI data in final merge: {merged_df['GINI'].notna().sum()} out of {len(merged_df)}")
        if 'CPI' in merged_df.columns:
            print(f"Number of rows with CPI data in final merge: {merged_df['CPI'].notna().sum()} out of {len(merged_df)}")

        # --- 6. Store the Merged Data ---
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
    except ValueError as e:
        print(f"ValueError during processing: {e}")
    except KeyError as e:
        print(f"KeyError during processing: {e}. This often means a column name expected was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
