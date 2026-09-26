import pandas as pd
import numpy as np
import os

cleaned_path = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\rainfall_cleaned.csv"
proj_dir = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india"
root_dir = r"c:\Users\Khurath\OneDrive\Desktop\90-days"

df_cleaned = pd.read_csv(cleaned_path)
subdivisions = sorted(df_cleaned['SUBDIVISION'].unique())
all_years = list(range(1901, 2016))

feature_records = []

for sub in subdivisions:
    # Filter for subdivision
    sub_raw = df_cleaned[df_cleaned['SUBDIVISION'] == sub].copy()
    
    # Reindex on full continuous calendar year sequence (1901-2015) to guarantee strictly causal calendar shifts
    sub_grid = sub_raw.set_index('YEAR').reindex(all_years)
    sub_grid['SUBDIVISION'] = sub
    
    # Target: ANNUAL for year Y
    target_annual = sub_grid['ANNUAL']
    
    # Feature 1: Previous-year annual rainfall (Y-1)
    lag_1 = target_annual.shift(1)
    
    # Feature 2: 2-year lag (Y-2)
    lag_2 = target_annual.shift(2)
    
    # Feature 3: 3-year lag (Y-3)
    lag_3 = target_annual.shift(3)
    
    # Feature 4: Previous 3-year rolling average: Mean of [Y-1, Y-2, Y-3]
    # Shifted first to strictly prevent lookahead into year Y
    roll_3_mean = lag_1.rolling(window=3, min_periods=3).mean()
    
    # Feature 5: Previous 5-year rolling average: Mean of [Y-1, Y-2, Y-3, Y-4, Y-5]
    roll_5_mean = lag_1.rolling(window=5, min_periods=5).mean()
    
    # Feature 6: Previous 5-year rolling standard deviation: Std of [Y-1, Y-2, Y-3, Y-4, Y-5]
    roll_5_std = lag_1.rolling(window=5, min_periods=5).std()
    
    # Feature 7-10: Previous-year seasonal rainfall (Y-1)
    winter_lag_1 = sub_grid['Jan-Feb'].shift(1)
    pre_monsoon_lag_1 = sub_grid['Mar-May'].shift(1)
    monsoon_lag_1 = sub_grid['Jun-Sep'].shift(1)
    post_monsoon_lag_1 = sub_grid['Oct-Dec'].shift(1)
    
    sub_feat = pd.DataFrame({
        'SUBDIVISION': sub,
        'YEAR': all_years,
        # Lags
        'annual_lag_1': lag_1.round(2),
        'annual_lag_2': lag_2.round(2),
        'annual_lag_3': lag_3.round(2),
        # Rolling stats
        'annual_roll_3_mean': roll_3_mean.round(2),
        'annual_roll_5_mean': roll_5_mean.round(2),
        'annual_roll_5_std': roll_5_std.round(2),
        # Previous-year seasonal rainfall
        'winter_lag_1': winter_lag_1.round(2),
        'pre_monsoon_lag_1': pre_monsoon_lag_1.round(2),
        'monsoon_lag_1': monsoon_lag_1.round(2),
        'post_monsoon_lag_1': post_monsoon_lag_1.round(2),
        # Target variable: Annual rainfall to predict for year Y
        'ANNUAL': target_annual.round(2)
    })
    
    feature_records.append(sub_feat)

# Concatenate all subdivisions
df_all_features = pd.concat(feature_records, ignore_index=True)

# Sort strictly chronologically by YEAR and SUBDIVISION (No random shuffling)
df_all_features = df_all_features.sort_values(by=['YEAR', 'SUBDIVISION']).reset_index(drop=True)

# 1. Full features dataset (preserving temporal grid, warm-up NaNs in initial 5 years)
out_full_proj = os.path.join(proj_dir, "rainfall_ml_full_features.csv")
df_all_features.to_csv(out_full_proj, index=False)

# 2. Ready-to-train ML dataset (dropping warm-up NaNs and missing target years)
df_ml_ready = df_all_features.dropna().reset_index(drop=True)
out_ml_proj = os.path.join(proj_dir, "rainfall_ml_dataset.csv")
out_ml_root = os.path.join(root_dir, "rainfall_ml_dataset.csv")

df_ml_ready.to_csv(out_ml_proj, index=False)
df_ml_ready.to_csv(out_ml_root, index=False)

print("="*70)
print("TIME-SERIES ML DATASET CREATION SUMMARY")
print("="*70)
print(f"Full feature grid shape: {df_all_features.shape}")
print(f"ML Ready-to-train dataset shape: {df_ml_ready.shape}")
print(f"Year range in ML ready dataset: {df_ml_ready['YEAR'].min()} to {df_ml_ready['YEAR'].max()}")
print(f"Number of subdivisions represented: {df_ml_ready['SUBDIVISION'].nunique()}")
print("\nTarget Variable:")
print("  - 'ANNUAL' (predicting annual rainfall for year Y)")
print("\nEngineered Predictive Features (strictly based on years < Y):")
features = [
    'annual_lag_1', 'annual_lag_2', 'annual_lag_3',
    'annual_roll_3_mean', 'annual_roll_5_mean', 'annual_roll_5_std',
    'winter_lag_1', 'pre_monsoon_lag_1', 'monsoon_lag_1', 'post_monsoon_lag_1'
]
for f in features:
    print(f"  - {f}")

print("\nNull counts in ML Ready dataset:")
print(df_ml_ready.isnull().sum())

print("\nFirst 5 rows of ML Ready dataset:")
print(df_ml_ready.head(5)[['SUBDIVISION', 'YEAR', 'annual_lag_1', 'annual_roll_3_mean', 'annual_roll_5_mean', 'annual_roll_5_std', 'monsoon_lag_1', 'ANNUAL']].to_string(index=False))

print("\nLast 5 rows of ML Ready dataset (Year 2015):")
print(df_ml_ready.tail(5)[['SUBDIVISION', 'YEAR', 'annual_lag_1', 'annual_roll_3_mean', 'annual_roll_5_mean', 'annual_roll_5_std', 'monsoon_lag_1', 'ANNUAL']].to_string(index=False))

# Verify strict temporal integrity and no lookahead:
print("\n" + "="*70)
print("VERIFYING STRICT CAUSALITY & TEMPORAL INTEGRITY")
print("="*70)
# Test a random subdivision: PUNJAB in 2010
sample_check = df_ml_ready[(df_ml_ready['SUBDIVISION'] == 'PUNJAB') & (df_ml_ready['YEAR'] == 2010)].iloc[0]
raw_punjab = df_cleaned[df_cleaned['SUBDIVISION'] == 'PUNJAB'].set_index('YEAR')

print(f"Punjab in 2010 Target ANNUAL: {sample_check['ANNUAL']} mm (Ground truth 2010: {raw_punjab.loc[2010, 'ANNUAL']} mm)")
print(f"annual_lag_1: {sample_check['annual_lag_1']} mm (Ground truth 2009: {raw_punjab.loc[2009, 'ANNUAL']} mm)")
print(f"annual_lag_2: {sample_check['annual_lag_2']} mm (Ground truth 2008: {raw_punjab.loc[2008, 'ANNUAL']} mm)")
print(f"annual_lag_3: {sample_check['annual_lag_3']} mm (Ground truth 2007: {raw_punjab.loc[2007, 'ANNUAL']} mm)")
expected_3m = round(raw_punjab.loc[2007:2009, 'ANNUAL'].mean(), 2)
expected_5m = round(raw_punjab.loc[2005:2009, 'ANNUAL'].mean(), 2)
expected_5s = round(raw_punjab.loc[2005:2009, 'ANNUAL'].std(), 2)
print(f"annual_roll_3_mean: {sample_check['annual_roll_3_mean']} mm (Expected: {expected_3m} mm)")
print(f"annual_roll_5_mean: {sample_check['annual_roll_5_mean']} mm (Expected: {expected_5m} mm)")
print(f"annual_roll_5_std: {sample_check['annual_roll_5_std']} mm (Expected: {expected_5s} mm)")
print(f"monsoon_lag_1: {sample_check['monsoon_lag_1']} mm (Ground truth 2009 Jun-Sep: {raw_punjab.loc[2009, 'Jun-Sep']} mm)")

assert sample_check['ANNUAL'] == raw_punjab.loc[2010, 'ANNUAL']
assert sample_check['annual_lag_1'] == raw_punjab.loc[2009, 'ANNUAL']
assert sample_check['annual_lag_2'] == raw_punjab.loc[2008, 'ANNUAL']
assert sample_check['annual_lag_3'] == raw_punjab.loc[2007, 'ANNUAL']
assert abs(sample_check['annual_roll_3_mean'] - expected_3m) <= 0.02
assert abs(sample_check['annual_roll_5_mean'] - expected_5m) <= 0.02
assert abs(sample_check['annual_roll_5_std'] - expected_5s) <= 0.02
assert sample_check['monsoon_lag_1'] == raw_punjab.loc[2009, 'Jun-Sep']

print("\nAll verification assertions PASSED: 100% causal, zero lookahead leakage.")
