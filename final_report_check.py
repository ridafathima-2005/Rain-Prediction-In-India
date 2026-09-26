import pandas as pd
import numpy as np

path1 = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\sources\rainfall in india 1901-2015 (1).csv"
df = pd.read_csv(path1)

print("="*70)
print("1. DUPLICATE (SUBDIVISION, YEAR) RECORDS")
print("="*70)
dup_keys = df.duplicated(subset=['SUBDIVISION', 'YEAR'], keep=False)
print(f"Total duplicate (SUBDIVISION, YEAR) rows: {dup_keys.sum()}")
if dup_keys.sum() > 0:
    print(df[dup_keys][['SUBDIVISION', 'YEAR']])

print("\n" + "="*70)
print("2. MISSING YEARS FOR EACH SUBDIVISION (1901-2015 = 115 expected years)")
print("="*70)
all_years = set(range(1901, 2016))
subdivisions = sorted(df['SUBDIVISION'].unique())
print(f"Total subdivisions: {len(subdivisions)}")

missing_years_dict = {}
for sub in subdivisions:
    sub_years = set(df[df['SUBDIVISION'] == sub]['YEAR'])
    missing = sorted(list(all_years - sub_years))
    if missing:
        missing_years_dict[sub] = missing
        print(f"- {sub}: {len(missing)} missing years -> {missing}")

if not missing_years_dict:
    print("All subdivisions have complete 115 years.")
else:
    total_missing_years = sum(len(v) for v in missing_years_dict.values())
    print(f"Total missing subdivision-year observations across entire dataset: {total_missing_years}")

print("\n" + "="*70)
print("3. MISSING RAINFALL VALUES (MONTHLY)")
print("="*70)
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
null_monthly = df[months].isnull().sum()
print("Missing counts by month:")
for m in months:
    print(f"  {m}: {null_monthly[m]} missing ({null_monthly[m]/len(df)*100:.3f}%)")

print("\n" + "="*70)
print("4. MISSING ANNUAL RAINFALL")
print("="*70)
print(f"Missing ANNUAL values: {df['ANNUAL'].isnull().sum()} ({df['ANNUAL'].isnull().sum()/len(df)*100:.3f}%)")

print("\n" + "="*70)
print("5. MISSING SEASONAL RAINFALL")
print("="*70)
seasons = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
null_seasons = df[seasons].isnull().sum()
for s in seasons:
    print(f"  {s}: {null_seasons[s]} missing ({null_seasons[s]/len(df)*100:.3f}%)")

print("\n" + "="*70)
print("6. ARE SEASONAL TOTALS MATCHING THEIR MONTHLY COMPONENTS?")
print("="*70)
# Check only rows where all relevant component months and season are NOT null
# 1. Jan-Feb vs JAN + FEB
jf_valid = df[df[['JAN', 'FEB', 'Jan-Feb']].notnull().all(axis=1)]
diff_jf = (jf_valid['JAN'] + jf_valid['FEB'] - jf_valid['Jan-Feb']).round(4)
mismatch_jf = jf_valid[diff_jf.abs() > 0.05]
print(f"Jan-Feb vs JAN+FEB: {len(jf_valid)} valid rows checked.")
print(f"  Max absolute difference: {diff_jf.abs().max()}")
print(f"  Discrepancies > 0.05 mm: {len(mismatch_jf)}")
if len(mismatch_jf) > 0:
    print(mismatch_jf[['SUBDIVISION', 'YEAR', 'JAN', 'FEB', 'Jan-Feb']])

# 2. Mar-May vs MAR + APR + MAY
mam_valid = df[df[['MAR', 'APR', 'MAY', 'Mar-May']].notnull().all(axis=1)]
diff_mam = (mam_valid[['MAR', 'APR', 'MAY']].sum(axis=1) - mam_valid['Mar-May']).round(4)
mismatch_mam = mam_valid[diff_mam.abs() > 0.05]
print(f"\nMar-May vs MAR+APR+MAY: {len(mam_valid)} valid rows checked.")
print(f"  Max absolute difference: {diff_mam.abs().max()}")
print(f"  Discrepancies > 0.05 mm: {len(mismatch_mam)}")
if len(mismatch_mam) > 0:
    print(mismatch_mam[['SUBDIVISION', 'YEAR', 'MAR', 'APR', 'MAY', 'Mar-May']])

# 3. Jun-Sep vs JUN + JUL + AUG + SEP
jj_valid = df[df[['JUN', 'JUL', 'AUG', 'SEP', 'Jun-Sep']].notnull().all(axis=1)]
diff_jj = (jj_valid[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1) - jj_valid['Jun-Sep']).round(4)
mismatch_jj = jj_valid[diff_jj.abs() > 0.05]
print(f"\nJun-Sep vs JUN+JUL+AUG+SEP: {len(jj_valid)} valid rows checked.")
print(f"  Max absolute difference: {diff_jj.abs().max()}")
print(f"  Discrepancies > 0.05 mm: {len(mismatch_jj)}")
if len(mismatch_jj) > 0:
    print(mismatch_jj[['SUBDIVISION', 'YEAR', 'JUN', 'JUL', 'AUG', 'SEP', 'Jun-Sep']])

# 4. Oct-Dec vs OCT + NOV + DEC
ond_valid = df[df[['OCT', 'NOV', 'DEC', 'Oct-Dec']].notnull().all(axis=1)]
diff_ond = (ond_valid[['OCT', 'NOV', 'DEC']].sum(axis=1) - ond_valid['Oct-Dec']).round(4)
mismatch_ond = ond_valid[diff_ond.abs() > 0.05]
print(f"\nOct-Dec vs OCT+NOV+DEC: {len(ond_valid)} valid rows checked.")
print(f"  Max absolute difference: {diff_ond.abs().max()}")
print(f"  Discrepancies > 0.05 mm: {len(mismatch_ond)}")
if len(mismatch_ond) > 0:
    print(mismatch_ond[['SUBDIVISION', 'YEAR', 'OCT', 'NOV', 'DEC', 'Oct-Dec']])

print("\n" + "="*70)
print("7. WHETHER ANNUAL MATCHES SUM OF MONTHLY RAINFALL")
print("="*70)
ann_valid = df[df[months + ['ANNUAL']].notnull().all(axis=1)]
sum_monthly = ann_valid[months].sum(axis=1)
diff_annual = (sum_monthly - ann_valid['ANNUAL']).round(4)
mismatch_annual_strict = ann_valid[diff_annual.abs() > 0.05]
mismatch_annual_major = ann_valid[diff_annual.abs() > 1.0]

print(f"Total rows where all 12 months & ANNUAL are present: {len(ann_valid)}")
print(f"Max absolute difference between sum(JAN..DEC) and ANNUAL: {diff_annual.abs().max():.4f} mm")
print(f"Rows with diff > 0.05 mm: {len(mismatch_annual_strict)}")
print(f"Rows with diff > 1.0 mm: {len(mismatch_annual_major)}")
print("\nDistribution of differences (sum_months - ANNUAL):")
print(diff_annual.abs().describe())

# Check diff between sum of 4 seasons and ANNUAL
ann_seasons_valid = df[df[seasons + ['ANNUAL']].notnull().all(axis=1)]
sum_seasons = ann_seasons_valid[seasons].sum(axis=1)
diff_seasons = (sum_seasons - ann_seasons_valid['ANNUAL']).round(4)
print(f"\nMax absolute difference between sum(4 seasons) and ANNUAL: {diff_seasons.abs().max():.4f} mm")
print(f"Rows with diff > 0.05 mm between sum(4 seasons) and ANNUAL: {(diff_seasons.abs() > 0.05).sum()}")

print("\n" + "="*70)
print("8. INVALID OR NEGATIVE RAINFALL VALUES")
print("="*70)
all_rain_cols = months + ['ANNUAL'] + seasons
negative_counts = {}
for col in all_rain_cols:
    neg = (df[col] < 0).sum()
    if neg > 0:
        negative_counts[col] = neg
print(f"Negative values across rainfall columns: {negative_counts if negative_counts else 'NONE (0 negative values)'}")

# Check minimum values across all rainfall columns
print("\nMinimum values in each rainfall column:")
print(df[all_rain_cols].min())

# Check for suspiciously high or unphysical values (e.g. > 10,000 mm in a month or > 25,000 mm annual)
print("\nMaximum values in each rainfall column:")
print(df[all_rain_cols].max())

# Zero rainfall counts (are zeros valid meteorological observations?)
print("\nZero rainfall count by month:")
for m in months:
    zeros = (df[m] == 0).sum()
    print(f"  {m}: {zeros} zero values ({zeros/len(df)*100:.1f}%)")

print("\nZero ANNUAL rainfall:")
print(f"  ANNUAL == 0: {(df['ANNUAL'] == 0).sum()}")
