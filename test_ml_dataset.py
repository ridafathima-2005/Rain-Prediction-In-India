import pandas as pd
import numpy as np

csv_path = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\rainfall_cleaned.csv"
df = pd.read_csv(csv_path)

print("Columns in rainfall_cleaned:", df.columns.tolist())
print("Total rows:", len(df))

# Let's inspect the subdivisions and years
subdivisions = sorted(df['SUBDIVISION'].unique())

# Method: For each subdivision, set YEAR as index (or reindex with all years 1901-2015 to guarantee calendar year alignment)
all_years = list(range(1901, 2016))

# Let's test building the dataset
feature_dfs = []

for sub in subdivisions:
    sub_df = df[df['SUBDIVISION'] == sub].copy()
    
    # Set YEAR as index and reindex across 1901-2015 to ensure calendar year shifts
    sub_df = sub_df.set_index('YEAR').reindex(all_years)
    sub_df['SUBDIVISION'] = sub  # restore subdivision name for unobserved years
    
    # 1. Target: ANNUAL for year Y
    target = sub_df['ANNUAL']
    
    # 2. Lags on ANNUAL
    # All features for year Y use ONLY shifted data: shift(1) means Y-1
    ann_lag1 = sub_df['ANNUAL'].shift(1)
    ann_lag2 = sub_df['ANNUAL'].shift(2)
    ann_lag3 = sub_df['ANNUAL'].shift(3)
    
    # 3. Rolling features on shifted series (strictly Y-1, Y-2, ...)
    # 3-year rolling average: mean of [Y-1, Y-2, Y-3]
    ann_roll3_mean = ann_lag1.rolling(window=3, min_periods=3).mean()
    
    # 5-year rolling average: mean of [Y-1, Y-2, Y-3, Y-4, Y-5]
    ann_roll5_mean = ann_lag1.rolling(window=5, min_periods=5).mean()
    
    # 5-year rolling std: std of [Y-1, Y-2, Y-3, Y-4, Y-5]
    ann_roll5_std = ann_lag1.rolling(window=5, min_periods=5).std()
    
    # 4. Previous-year seasonal rainfall (shift 1)
    jf_lag1 = sub_df['Jan-Feb'].shift(1)
    mam_lag1 = sub_df['Mar-May'].shift(1)
    jj_lag1 = sub_df['Jun-Sep'].shift(1)
    ond_lag1 = sub_df['Oct-Dec'].shift(1)
    
    # Build dataframe for this subdivision
    feat_df = pd.DataFrame({
        'SUBDIVISION': sub,
        'YEAR': all_years,
        # Lags
        'annual_lag_1': ann_lag1.values,
        'annual_lag_2': ann_lag2.values,
        'annual_lag_3': ann_lag3.values,
        # Rolling stats
        'annual_roll_3_mean': ann_roll3_mean.values,
        'annual_roll_5_mean': ann_roll5_mean.values,
        'annual_roll_5_std': ann_roll5_std.values,
        # Previous year seasonal rainfall
        'winter_lag_1': jf_lag1.values,
        'pre_monsoon_lag_1': mam_lag1.values,
        'monsoon_lag_1': jj_lag1.values,
        'post_monsoon_lag_1': ond_lag1.values,
        # Target variable (ANNUAL for year Y)
        'ANNUAL': target.values
    })
    
    feature_dfs.append(feat_df)

ml_df = pd.concat(feature_dfs, ignore_index=True)

print("\nFull feature dataframe shape (including lead-in NaNs):", ml_df.shape)
print("Null count before filtering:")
print(ml_df.isnull().sum())

# Only keep rows where both the target and the features are available (complete case for ML training)
# Or let's see how many rows are in the original dataset vs ML complete dataset:
# Notice that for 1901-1905, 5-year rolling features are naturally NaN because 5 years of history don't exist yet.
# 36 subdivisions * 5 years = 180 rows where 5-year history is not yet available.
ml_complete = ml_df.dropna().copy()
print("\nComplete ML dataset (no missing features or target):", ml_complete.shape)

# Let's inspect a sample subdivision
sample = ml_complete[ml_complete['SUBDIVISION'] == 'KERALA'].head(10)
print("\nSample Kerala (1906-1915):")
print(sample[['YEAR', 'annual_lag_1', 'annual_lag_2', 'annual_lag_3', 'annual_roll_3_mean', 'annual_roll_5_mean', 'annual_roll_5_std', 'ANNUAL']].to_string(index=False))

# Verify that for year Y, annual_lag_1 is exactly Kerala's ANNUAL at Y-1:
kerala_raw = df[df['SUBDIVISION'] == 'KERALA'].set_index('YEAR')['ANNUAL']
for yr in range(1906, 1912):
    y_target = sample.loc[sample['YEAR'] == yr, 'ANNUAL'].values[0]
    y_lag1 = sample.loc[sample['YEAR'] == yr, 'annual_lag_1'].values[0]
    raw_target = kerala_raw[yr]
    raw_lag1 = kerala_raw[yr-1]
    assert y_target == raw_target
    assert y_lag1 == raw_lag1
    # Check 3-year rolling mean
    expected_3m = round(kerala_raw.loc[yr-3:yr-1].mean(), 4)
    calc_3m = round(sample.loc[sample['YEAR'] == yr, 'annual_roll_3_mean'].values[0], 4)
    assert abs(expected_3m - calc_3m) < 1e-3, f"Mismatch in 3m: {expected_3m} vs {calc_3m}"
    
print("\nVerification PASSED: strictly causal, zero lookahead, exact calendar year alignment.")
