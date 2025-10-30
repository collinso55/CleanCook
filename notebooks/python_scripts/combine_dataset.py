# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os

# def inspect_dataframe(df, name):
#     """Inspect a dataframe to understand its structure"""
#     print(f"\n🔍 {name} DATAFRAME INSPECTION:")
#     print(f"Shape: {df.shape}")
#     print(f"Columns: {list(df.columns)}")
#     print(f"First 3 rows:")
#     print(df.head(3))
#     return df

# def load_and_clean_weather_data(weather_file):
#     """Load and clean the weather data with flexible column handling"""
#     print("🌤️ Loading weather data...")
#     try:
#         weather_df = pd.read_csv(weather_file)
#         weather_df = inspect_dataframe(weather_df, "WEATHER")
        
#         # Convert date column
#         date_col = None
#         for col in ['date', 'Date', 'DATE', 'datetime']:
#             if col in weather_df.columns:
#                 date_col = col
#                 break
#         if date_col:
#             weather_df['date'] = pd.to_datetime(weather_df[date_col])
#         else:
#             weather_df['date'] = pd.date_range(start='2010-01-01', periods=len(weather_df), freq='D')
        
#         weather_df.set_index('date', inplace=True)
        
#         # Select and rename weather columns
#         weather_columns = {}
#         for target, possible_names in [
#             ('temperature_avg', ['temperature_avg', 'temp_avg', 'temperature', 'temp']),
#             ('rainfall_mm', ['rainfall_mm', 'rainfall', 'precipitation', 'rain']),
#             ('humidity_avg', ['humidity_avg', 'humidity', 'humidity_percent']),
#             ('wind_speed_kmh', ['wind_speed_kmh', 'wind_speed', 'wind']),
#             ('season', ['season', 'Season'])
#         ]:
#             for col in possible_names:
#                 if col in weather_df.columns:
#                     weather_columns[target] = col
#                     break
        
#         available_cols = [v for v in weather_columns.values()]
#         weather_essential = weather_df[available_cols].copy()
        
#         # Rename to standard names
#         rename_dict = {v: k for k, v in weather_columns.items()}
#         weather_essential.rename(columns=rename_dict, inplace=True)
        
#         print(f"✅ Weather data loaded: {len(weather_essential)} days")
#         return weather_essential
        
#     except Exception as e:
#         print(f"❌ Error loading weather data: {e}")
#         return None

# def load_and_clean_fao_data(fao_file):
#     """Load and clean FAO wood fuel data with flexible column handling"""
#     print("🌳 Loading FAO wood fuel data...")
#     try:
#         fao_df = pd.read_csv(fao_file)
#         fao_df = inspect_dataframe(fao_df, "FAO")
        
#         # Convert date column
#         date_col = None
#         for col in ['date', 'Date', 'DATE', 'datetime']:
#             if col in fao_df.columns:
#                 date_col = col
#                 break
#         if date_col:
#             fao_df['date'] = pd.to_datetime(fao_df[date_col])
#         else:
#             fao_df['date'] = pd.date_range(start='2010-01-01', periods=len(fao_df), freq='D')
        
#         fao_df.set_index('date', inplace=True)
        
#         # Find production/value column
#         production_col = None
#         for col in ['daily_production_m3', 'production_m3', 'value', 'Value', 'woodfuel_production_m3', 'daily_value', 'Value']:
#             if col in fao_df.columns:
#                 production_col = col
#                 break
        
#         if production_col:
#             fao_essential = fao_df[[production_col]].copy()
#             fao_essential.rename(columns={production_col: 'woodfuel_production_m3'}, inplace=True)
#             print(f"✅ FAO data loaded: {len(fao_essential)} days")
#             return fao_essential
#         else:
#             print("❌ No production column found in FAO data")
#             return None
            
#     except Exception as e:
#         print(f"❌ Error loading FAO data: {e}")
#         return None

# def load_and_clean_worldbank_data(wb_file):
#     """Load and clean World Bank data with flexible column handling"""
#     print("🏦 Loading World Bank clean cooking data...")
#     try:
#         wb_df = pd.read_csv(wb_file)
#         wb_df = inspect_dataframe(wb_df, "WORLD BANK")
        
#         # Convert date column
#         date_col = None
#         for col in ['date', 'Date', 'DATE', 'datetime']:
#             if col in wb_df.columns:
#                 date_col = col
#                 break
#         if date_col:
#             wb_df['date'] = pd.to_datetime(wb_df[date_col])
#         else:
#             wb_df['date'] = pd.date_range(start='2010-01-01', periods=len(wb_df), freq='D')
        
#         wb_df.set_index('date', inplace=True)
        
#         # Find clean fuel access column
#         fuel_col = None
#         for col in ['clean_fuel_access', 'clean_fuel', 'access_clean_fuels', 'value', 'Value', 'percentage']:
#             if col in wb_df.columns:
#                 fuel_col = col
#                 break
        
#         if fuel_col:
#             wb_essential = wb_df[[fuel_col]].copy()
#             wb_essential.rename(columns={fuel_col: 'clean_fuel_access'}, inplace=True)
#             print(f"✅ World Bank data loaded: {len(wb_essential)} days")
#             return wb_essential
#         else:
#             print("❌ No clean fuel access column found")
#             return None
            
#     except Exception as e:
#         print(f"❌ Error loading World Bank data: {e}")
#         return None

# def create_master_dataset(weather_df=None, fao_df=None, wb_df=None, start_date='2010-01-01', end_date='2024-12-31'):
#     """Merge all available datasets into a master dataset"""
#     print("\n🔗 Merging datasets...")
    
#     # Create a complete date index
#     full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
#     master_df = pd.DataFrame(index=full_date_range)
#     master_df.index.name = 'date'
    
#     print(f"📅 Master date range: {len(master_df)} days")
    
#     # Merge available datasets
#     datasets_to_merge = []
#     if weather_df is not None:
#         datasets_to_merge.append(('Weather', weather_df))
#     if fao_df is not None:
#         datasets_to_merge.append(('FAO', fao_df))
#     if wb_df is not None:
#         datasets_to_merge.append(('World Bank', wb_df))
    
#     for name, df in datasets_to_merge:
#         # Ensure the dataframe has the same index type
#         df_index = df.index
#         if not isinstance(df_index, pd.DatetimeIndex):
#             df.index = pd.to_datetime(df.index)
        
#         master_df = master_df.merge(df, left_index=True, right_index=True, how='left')
#         non_na_count = df.notna().any(axis=1).sum() if len(df.columns) > 0 else 0
#         print(f"✅ {name} data merged: {non_na_count} days with data")
    
#     # Add temporal features
#     master_df['year'] = master_df.index.year
#     master_df['month'] = master_df.index.month
#     master_df['quarter'] = master_df.index.quarter
#     master_df['day_of_year'] = master_df.index.dayofyear
#     master_df['day_of_week'] = master_df.index.dayofweek
#     master_df['is_weekend'] = (master_df['day_of_week'] >= 5).astype(int)
    
#     # Create season column if it doesn't exist
#     if 'season' not in master_df.columns:
#         master_df['season'] = master_df['month'].apply(lambda x: 'Dry' if x in [11, 12, 1, 2] else 'Rainy')
    
#     # Calculate derived features
#     master_df = calculate_derived_features(master_df)
    
#     # Handle missing values
#     master_df = handle_missing_values(master_df)
    
#     return master_df

# def calculate_derived_features(df):
#     """Calculate additional features for time series forecasting"""
#     print("📊 Calculating derived features...")
    
#     # Only calculate for columns that exist
#     if 'temperature_avg' in df.columns:
#         df['temp_7d_avg'] = df['temperature_avg'].rolling(7, center=True, min_periods=1).mean()
#         df['temp_30d_avg'] = df['temperature_avg'].rolling(30, center=True, min_periods=1).mean()
#         df['temperature_lag1'] = df['temperature_avg'].shift(1)
    
#     if 'rainfall_mm' in df.columns:
#         df['rainfall_7d_avg'] = df['rainfall_mm'].rolling(7, center=True, min_periods=1).mean()
#         df['rainfall_30d_cumulative'] = df['rainfall_mm'].rolling(30, min_periods=1).sum()
#         df['rainfall_lag1'] = df['rainfall_mm'].shift(1)
    
#     if 'woodfuel_production_m3' in df.columns:
#         df['woodfuel_7d_avg'] = df['woodfuel_production_m3'].rolling(7, center=True, min_periods=1).mean()
#         df['woodfuel_30d_avg'] = df['woodfuel_production_m3'].rolling(30, center=True, min_periods=1).mean()
#         df['woodfuel_lag1'] = df['woodfuel_production_m3'].shift(1)
#         df['woodfuel_lag7'] = df['woodfuel_production_m3'].shift(7)
    
#     if 'clean_fuel_access' in df.columns:
#         df['clean_fuel_30d_avg'] = df['clean_fuel_access'].rolling(30, center=True, min_periods=1).mean()
    
#     # Seasonal indicators
#     df['is_dry_season'] = (df['month'].isin([11, 12, 1, 2])).astype(int)
#     df['is_rainy_season'] = (df['month'].isin([3, 4, 5, 6, 7, 8, 9, 10])).astype(int)
    
#     return df

# def handle_missing_values(df):
#     """Handle missing values in the merged dataset"""
#     print("🔧 Handling missing values...")
    
#     missing_before = df.isnull().sum().sum()
    
#     # Fill missing values
#     for column in df.select_dtypes(include=[np.number]).columns:
#         if df[column].isnull().any():
#             df[column].interpolate(method='linear', inplace=True)
#             df[column].fillna(method='ffill', inplace=True)
#             df[column].fillna(method='bfill', inplace=True)
    
#     # For non-numeric columns
#     for column in df.select_dtypes(exclude=[np.number]).columns:
#         if df[column].isnull().any():
#             df[column].fillna(method='ffill', inplace=True)
#             df[column].fillna(method='bfill', inplace=True)
    
#     missing_after = df.isnull().sum().sum()
#     print(f"✅ Missing values treated: {missing_before} -> {missing_after}")
    
#     return df

# def create_fallback_data():
#     """Create fallback data if real data is unavailable"""
#     print("🔄 Creating fallback data...")
    
#     dates = pd.date_range('2010-01-01', '2024-12-31', freq='D')
#     fallback_df = pd.DataFrame(index=dates)
    
#     # Add basic weather data
#     fallback_df['temperature_avg'] = 20 + 5 * np.sin(2 * np.pi * (fallback_df.index.dayofyear - 80) / 365)
#     fallback_df['rainfall_mm'] = np.where(fallback_df.index.month.isin([11, 12, 1, 2]), 
#                                          np.random.exponential(1), 
#                                          np.random.exponential(5))
#     fallback_df['humidity_avg'] = 70 + 10 * np.sin(2 * np.pi * (fallback_df.index.dayofyear - 180) / 365)
    
#     # Add wood fuel data (growing over time)
#     years = (fallback_df.index.year - 2010).values
#     fallback_df['woodfuel_production_m3'] = 25000 + (years * 100) + np.random.normal(0, 2000, len(dates))
    
#     # Add clean fuel access (growing over time)
#     fallback_df['clean_fuel_access'] = 2.1 + (years * 0.3) + np.random.normal(0, 0.1, len(dates))
    
#     return fallback_df

# def main():
#     """Main function to combine all datasets"""
#     print("🚀 COMBINING WEATHER, FAO, AND WORLD BANK DATASETS")
#     print("=" * 60)
    
#     # Define file paths - UPDATE THESE WITH YOUR ACTUAL FILE NAMES
#     weather_file = "kumbo_daily_weather_2010_2025_e.csv"
#     fao_file = "cameroon_woodfuel_daily_2010_2024.csv" 
#     wb_file = "cameroon_clean_cooking_simple_2024.csv"
    
#     # Load available datasets
#     weather_df = load_and_clean_weather_data(weather_file) if os.path.exists(weather_file) else None
#     fao_df = load_and_clean_fao_data(fao_file) if os.path.exists(fao_file) else None
#     wb_df = load_and_clean_worldbank_data(wb_file) if os.path.exists(wb_file) else None
    
#     # Check what we have
#     available_datasets = []
#     if weather_df is not None:
#         available_datasets.append("Weather")
#     if fao_df is not None:
#         available_datasets.append("FAO")
#     if wb_df is not None:
#         available_datasets.append("World Bank")
    
#     if available_datasets:
#         print(f"\n✅ Datasets available: {', '.join(available_datasets)}")
        
#         # Create master dataset with available data
#         master_df = create_master_dataset(weather_df=weather_df, fao_df=fao_df, wb_df=wb_df)
        
#         # Save the combined dataset
#         output_file = "fuel_consumption_master_dataset_2010_2024.csv"
#         master_df.reset_index().to_csv(output_file, index=False)
        
#         print(f"\n✅ MASTER DATASET CREATED SUCCESSFULLY!")
#         print(f"📁 File: {output_file}")
#         print(f"📊 Records: {len(master_df):,} days")
#         print(f"📈 Features: {len(master_df.columns)} variables")
#         print(f"💾 Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
#         # Show sample
#         print(f"\n👀 SAMPLE DATA (first 5 rows):")
#         sample_cols = [col for col in ['date', 'temperature_avg', 'rainfall_mm', 'woodfuel_production_m3', 'clean_fuel_access'] if col in master_df.columns]
#         print(master_df.reset_index()[sample_cols].head())
        
#     else:
#         print("❌ No datasets were successfully loaded")
#         print("🔄 Creating fallback dataset...")
#         master_df = create_fallback_data()
#         output_file = "fuel_consumption_fallback_dataset_2010_2024.csv"
#         master_df.reset_index().to_csv(output_file, index=False)
#         print(f"✅ Fallback dataset created: {output_file}")

# if __name__ == "__main__":
#     main()

# #--------------------------------------merging weather data and firewood data directly----------------------------------------
# import pandas as pd

# # === Load datasets ===
# weather_path = "../raw/rural_daily_weather_data.csv"
# firewood_path = "cameroon_woodfuel_daily_2010_2024.csv"

# weather_df = pd.read_csv(weather_path)
# firewood_df = pd.read_csv(firewood_path)

# # === Select relevant columns ===
# weather_cols = [
#     "date", "year", "month", "day",
#     "temperature_avg", "temperature_min", "temperature_max",
#     "rainfall_mm", "humidity_avg", "pressure_hpa",
#     "wind_speed_kmh", "is_rainy_day", "season",
#     "latitude", "longitude"
# ]
# firewood_cols = [
#     "date", "year", "month", "day", "value", "unit", "area"
# ]

# weather_df = weather_df[weather_cols]
# firewood_df = firewood_df[firewood_cols]

# # === Merge on 'date' (daily alignment) ===
# merged_df = pd.merge(
#     firewood_df,
#     weather_df,
#     on=["date", "year", "month", "day"],
#     how="inner"
# )

# # === Rename key column for clarity ===
# merged_df = merged_df.rename(columns={"value": "firewood_demand_m3"})

# # === Sort and clean up ===
# merged_df = merged_df.sort_values("date").reset_index(drop=True)

# # === Handle any missing values if necessary ===
# merged_df = merged_df.fillna(method="ffill")

# # === Save final merged dataset ===
# merged_df.to_csv("merged_firewood_weather_daily.csv", index=False)

# print(f"✅ Merge successful! Final shape: {merged_df.shape}")
# print("📁 Saved as 'merged_firewood_weather_daily.csv'")
# print("Sample:")
# print(merged_df.head())


#--------------------------------------merging clean fuel and firewood& weather data directly----------------------------------------
import pandas as pd

firewood = pd.read_csv("merged_firewood_weather_daily.csv")
cleanfuel = pd.read_csv("cameroon_clean_cooking_simple_2010_2024.csv")

# Convert date columns if necessary
firewood['date'] = pd.to_datetime(firewood['date'])
cleanfuel['date'] = pd.to_datetime(cleanfuel['date'])

# Merge on date (inner join)
merged_fuels = pd.merge(firewood, cleanfuel, on="date", how="inner")

# Rename clean fuel value
merged_fuels = merged_fuels.rename(columns={"value": "clean_fuel_demand"})

merged_fuels.to_csv("merged_fuel_weather_cameroon.csv", index=False)
print("✅ Merged firewood + clean fuel + weather saved as 'merged_fuel_weather_cameroon.csv'")
