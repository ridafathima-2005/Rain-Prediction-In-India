import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

cleaned_path = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\rainfall_cleaned.csv"
district_path = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\sources\district wise rainfall normal.csv"
chart_dir = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\charts"
artifact_dir = r"C:\Users\Khurath\.gemini\antigravity-ide\brain\550705bc-4f0e-4bf1-b9b5-cdbd80abcf8b"

df = pd.read_csv(cleaned_path)
df_dist = pd.read_csv(district_path)

seasons = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']

# Calculation per subdivision
results = []
subdivisions = sorted(df['SUBDIVISION'].unique())

for sub in subdivisions:
    sub_df = df[df['SUBDIVISION'] == sub].sort_values(by='YEAR')
    
    # Non-null annual values
    ann_valid = sub_df.dropna(subset=['ANNUAL'])
    mean_ann = ann_valid['ANNUAL'].mean()
    max_ann = ann_valid['ANNUAL'].max()
    max_yr = int(ann_valid.loc[ann_valid['ANNUAL'].idxmax(), 'YEAR'])
    min_ann = ann_valid['ANNUAL'].min()
    min_yr = int(ann_valid.loc[ann_valid['ANNUAL'].idxmin(), 'YEAR'])
    std_ann = ann_valid['ANNUAL'].std()
    cv_ann = (std_ann / mean_ann) * 100
    
    # Latest available annual rainfall
    latest_row = ann_valid.iloc[-1]
    latest_yr = int(latest_row['YEAR'])
    latest_val = latest_row['ANNUAL']
    latest_anomaly = latest_val - mean_ann
    latest_anomaly_pct = (latest_anomaly / mean_ann) * 100
    
    # Seasonal averages
    jf_mean = sub_df['Jan-Feb'].mean()
    mam_mean = sub_df['Mar-May'].mean()
    jj_mean = sub_df['Jun-Sep'].mean()
    ond_mean = sub_df['Oct-Dec'].mean()
    
    # Monsoon contribution (%)
    monsoon_contrib = (jj_mean / mean_ann) * 100
    
    results.append({
        'Subdivision': sub,
        'Observations': len(ann_valid),
        'Mean_Annual_mm': round(mean_ann, 1),
        'Max_Annual_mm': round(max_ann, 1),
        'Max_Year': max_yr,
        'Min_Annual_mm': round(min_ann, 1),
        'Min_Year': min_yr,
        'Std_Dev_mm': round(std_ann, 1),
        'CV_pct': round(cv_ann, 1),
        'Latest_Year': latest_yr,
        'Latest_Annual_mm': round(latest_val, 1),
        'Latest_Anomaly_pct': round(latest_anomaly_pct, 1),
        'Winter_JF_mm': round(jf_mean, 1),
        'PreMonsoon_MAM_mm': round(mam_mean, 1),
        'Monsoon_JJAS_mm': round(jj_mean, 1),
        'PostMonsoon_OND_mm': round(ond_mean, 1),
        'Monsoon_Contribution_pct': round(monsoon_contrib, 1)
    })

df_sub = pd.DataFrame(results)

# Supporting information from district wise rainfall normal.csv where definitions align
print("="*80)
print("SUBDIVISION REGIONAL SUMMARY TABLE")
print("="*80)
print(df_sub[['Subdivision', 'Mean_Annual_mm', 'Std_Dev_mm', 'CV_pct', 'Latest_Annual_mm', 'Latest_Anomaly_pct', 'Monsoon_JJAS_mm', 'Monsoon_Contribution_pct']].to_string(index=False))

# Export subdivision summary CSV
summary_csv = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\subdivision_rainfall_summary.csv"
df_sub.to_csv(summary_csv, index=False)
print(f"\nSaved summary CSV to: {summary_csv}")

# Compatible State-to-Subdivision lookup in district-wise normals
print("\n" + "="*80)
print("SUPPORTING DISTRICT-WISE RAINFALL NORMAL BENCHMARKS (COMPATIBLE REGIONS)")
print("="*80)
# Clean state names in district dataset for comparison
df_dist_clean = df_dist.copy()
df_dist_clean['STATE_CLEAN'] = df_dist_clean['STATE_UT_NAME'].str.strip()
state_map = {
    'CHATISGARH': 'CHHATTISGARH',
    'HIMACHAL': 'HIMACHAL PRADESH',
    'UTTARANCHAL': 'UTTARAKHAND'
}
df_dist_clean['STATE_CLEAN'] = df_dist_clean['STATE_CLEAN'].replace(state_map)

compatible_states = [
    'BIHAR', 'CHHATTISGARH', 'HIMACHAL PRADESH', 'JHARKHAND', 'KERALA', 
    'ORISSA', 'PUNJAB', 'TAMIL NADU', 'UTTARAKHAND'
]

dist_benchmarks = []
for state in compatible_states:
    dist_slice = df_dist_clean[df_dist_clean['STATE_CLEAN'] == state]
    sub_match = df_sub[df_sub['Subdivision'] == state]
    if len(sub_match) > 0:
        hist_mean = sub_match.iloc[0]['Mean_Annual_mm']
        hist_monsoon = sub_match.iloc[0]['Monsoon_JJAS_mm']
        num_districts = len(dist_slice)
        dist_ann_mean = dist_slice['ANNUAL'].mean()
        dist_ann_min = dist_slice['ANNUAL'].min()
        dist_ann_max = dist_slice['ANNUAL'].max()
        dist_benchmarks.append({
            'State/Subdivision': state,
            'Num_Districts': num_districts,
            'Historical_Subdivision_Mean': hist_mean,
            'District_Normal_Mean': round(dist_ann_mean, 1),
            'District_Min_Normal': round(dist_ann_min, 1),
            'District_Max_Normal': round(dist_ann_max, 1),
            'Intra_State_Spread': round(dist_ann_max - dist_ann_min, 1)
        })

df_bench = pd.DataFrame(dist_benchmarks)
print(df_bench.to_string(index=False))

# =====================================================================
# VISUALIZATIONS
# =====================================================================
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Chart 1: Ranked Annual Rainfall with Variability Range (Min, Mean, Max)
fig, ax = plt.subplots(figsize=(14, 18))
df_sorted = df_sub.sort_values(by='Mean_Annual_mm', ascending=True)
y_pos = np.arange(len(df_sorted))

# Horizontal bar of mean
bars = ax.barh(y_pos, df_sorted['Mean_Annual_mm'], color='#2b5c8f', alpha=0.85, height=0.6, label='Long-Term Mean Annual')

# Plot Min and Max points
ax.scatter(df_sorted['Min_Annual_mm'], y_pos, color='#c0392b', s=45, zorder=4, label='Historical Minimum (Drought Peak)')
ax.scatter(df_sorted['Max_Annual_mm'], y_pos, color='#27ae60', s=45, zorder=4, label='Historical Maximum (Flood Peak)')
# Connect min to max with a thin line
for i, y in enumerate(y_pos):
    ax.plot([df_sorted.iloc[i]['Min_Annual_mm'], df_sorted.iloc[i]['Max_Annual_mm']], [y, y], color='gray', alpha=0.6, linewidth=1.5, zorder=3)

# Label means
for i, bar in enumerate(bars):
    val = df_sorted.iloc[i]['Mean_Annual_mm']
    ax.text(val + 35, y_pos[i], f"{val:.0f} mm", va='center', ha='left', fontsize=9, fontweight='bold', color='#1a252f')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted['Subdivision'], fontsize=10)
ax.set_xlabel('Annual Rainfall (mm)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Subdivision Annual Rainfall Profile: Mean, Historical Min, and Historical Max (1901–2015)', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.set_xlim(0, 6800)

plt.tight_layout()
c1_path = os.path.join(chart_dir, "subdivision_annual_comparison.png")
c1_art = os.path.join(artifact_dir, "subdivision_annual_comparison.png")
fig.savefig(c1_path, dpi=200, bbox_inches='tight')
fig.savefig(c1_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved Chart 1: {c1_path}")

# Chart 2: Stacked Seasonal Composition for all 36 Subdivisions
fig, ax = plt.subplots(figsize=(14, 18))
df_stack_sorted = df_sub.sort_values(by='Mean_Annual_mm', ascending=True)
y_pos = np.arange(len(df_stack_sorted))

p_jf = ax.barh(y_pos, df_stack_sorted['Winter_JF_mm'], color='#5dade2', height=0.65, label='Winter (Jan–Feb)')
p_mam = ax.barh(y_pos, df_stack_sorted['PreMonsoon_MAM_mm'], left=df_stack_sorted['Winter_JF_mm'], color='#58d68d', height=0.65, label='Pre-Monsoon (Mar–May)')
p_jj = ax.barh(y_pos, df_stack_sorted['Monsoon_JJAS_mm'], left=df_stack_sorted['Winter_JF_mm'] + df_stack_sorted['PreMonsoon_MAM_mm'], color='#e67e22', height=0.65, label='Monsoon (Jun–Sep)')
p_ond = ax.barh(y_pos, df_stack_sorted['PostMonsoon_OND_mm'], left=df_stack_sorted['Winter_JF_mm'] + df_stack_sorted['PreMonsoon_MAM_mm'] + df_stack_sorted['Monsoon_JJAS_mm'], color='#af7ac5', height=0.65, label='Post-Monsoon (Oct–Dec)')

# Annotate monsoon percentage share
for i, y in enumerate(y_pos):
    m_pct = df_stack_sorted.iloc[i]['Monsoon_Contribution_pct']
    tot = df_stack_sorted.iloc[i]['Mean_Annual_mm']
    ax.text(tot + 30, y, f"{m_pct:.0f}% Monsoon", va='center', ha='left', fontsize=8.5, fontweight='bold', color='#784212')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_stack_sorted['Subdivision'], fontsize=10)
ax.set_xlabel('Cumulative Seasonal Rainfall (mm)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Seasonal Composition of Annual Rainfall by Subdivision (1901–2015)', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, fontsize=11)
ax.set_xlim(0, 4200)

plt.tight_layout()
c2_path = os.path.join(chart_dir, "subdivision_seasonal_stacked.png")
c2_art = os.path.join(artifact_dir, "subdivision_seasonal_stacked.png")
fig.savefig(c2_path, dpi=200, bbox_inches='tight')
fig.savefig(c2_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved Chart 2: {c2_path}")

# Chart 3: Risk Quadrant Plot: Mean Rainfall vs. Inter-Annual Volatility (CV %)
fig, ax = plt.subplots(figsize=(13, 9))
sns.scatterplot(
    data=df_sub, 
    x='Mean_Annual_mm', 
    y='CV_pct', 
    hue='Monsoon_Contribution_pct',
    palette='viridis',
    size='Std_Dev_mm',
    sizes=(60, 450),
    alpha=0.85,
    edgecolor='black',
    ax=ax
)

# Reference quadrant lines
med_x = df_sub['Mean_Annual_mm'].median()
med_y = df_sub['CV_pct'].median()
ax.axvline(med_x, color='red', linestyle='--', alpha=0.5, label=f'Median Rainfall ({med_x:.0f} mm)')
ax.axhline(med_y, color='red', linestyle=':', alpha=0.5, label=f'Median CV ({med_y:.1f}%)')

# Annotate extreme points
key_annotations = [
    'WEST RAJASTHAN', 'SAURASHTRA & KUTCH', 'COASTAL KARNATAKA', 'ARUNACHAL PRADESH',
    'TAMIL NADU', 'KERALA', 'HARYANA DELHI & CHANDIGARH', 'ORISSA', 'ASSAM & MEGHALAYA',
    'PUNJAB', 'RAYALASEEMA'
]
for _, row in df_sub.iterrows():
    if row['Subdivision'] in key_annotations:
        ax.annotate(
            row['Subdivision'],
            (row['Mean_Annual_mm'], row['CV_pct']),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            fontweight='bold',
            color='#1a252f'
        )

# Quadrant labels
ax.text(3200, 31, "High Rain, High Volatility", fontsize=11, fontweight='bold', color='#7f1d1d', alpha=0.6, ha='center')
ax.text(3200, 12, "High Rain, Stable (Resilient)", fontsize=11, fontweight='bold', color='#14532d', alpha=0.6, ha='center')
ax.text(500, 31, "Arid/Semi-Arid, High Risk (Vulnerable)", fontsize=11, fontweight='bold', color='#7f1d1d', alpha=0.6, ha='center')
ax.text(500, 12, "Low Rain, Moderate Stability", fontsize=11, fontweight='bold', color='#1e3a8a', alpha=0.6, ha='center')

ax.set_title('Subdivision Risk Quadrant: Mean Annual Rainfall vs. Relative Inter-Annual Volatility (CV %)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Mean Annual Rainfall (mm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Coefficient of Variation - CV (%)', fontsize=11, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

plt.tight_layout()
c3_path = os.path.join(chart_dir, "subdivision_risk_quadrant.png")
c3_art = os.path.join(artifact_dir, "subdivision_risk_quadrant.png")
fig.savefig(c3_path, dpi=200, bbox_inches='tight')
fig.savefig(c3_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved Chart 3: {c3_path}")

# Chart 4: 2015 Latest Rainfall Anomaly (%) across all 36 Subdivisions
fig, ax = plt.subplots(figsize=(14, 10))
df_latest_sorted = df_sub.sort_values(by='Latest_Anomaly_pct', ascending=True)
colors_anom = ['#c0392b' if x < 0 else '#27ae60' for x in df_latest_sorted['Latest_Anomaly_pct']]

bars = ax.bar(df_latest_sorted['Subdivision'], df_latest_sorted['Latest_Anomaly_pct'], color=colors_anom, width=0.7, edgecolor='black', alpha=0.85)
ax.axhline(0, color='black', linewidth=1.2)
ax.axhline(-19, color='red', linestyle='--', alpha=0.7, label='IMD Deficient Threshold (-19%)')
ax.axhline(19, color='green', linestyle='--', alpha=0.7, label='IMD Excess Threshold (+19%)')

ax.set_xticks(range(len(df_latest_sorted)))
ax.set_xticklabels(df_latest_sorted['Subdivision'], rotation=90, fontsize=9.5, fontweight='bold')
ax.set_ylabel('2015 Rainfall Departure / Anomaly (%)', fontsize=11, fontweight='bold')
ax.set_title('Latest Available Year (2015) Rainfall Departure from Historical Normal Across Subdivisions', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True)
ax.set_ylim(-50, 40)

for bar in bars:
    yval = bar.get_height()
    va = 'bottom' if yval >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + (1.2 if yval >= 0 else -2.2), f"{yval:+.0f}%", ha='center', va=va, fontsize=8, fontweight='bold')

plt.tight_layout()
c4_path = os.path.join(chart_dir, "subdivision_2015_anomaly.png")
c4_art = os.path.join(artifact_dir, "subdivision_2015_anomaly.png")
fig.savefig(c4_path, dpi=200, bbox_inches='tight')
fig.savefig(c4_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved Chart 4: {c4_path}")

print("All subdivision regional charts and tables successfully created.")
