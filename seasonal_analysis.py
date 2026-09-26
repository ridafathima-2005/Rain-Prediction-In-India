import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Paths
csv_path = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\rainfall_cleaned.csv"
chart_dir = r"c:\Users\Khurath\OneDrive\Desktop\90-days\projects\rain_prediction_in_india\charts"
artifact_dir = r"C:\Users\Khurath\.gemini\antigravity-ide\brain\550705bc-4f0e-4bf1-b9b5-cdbd80abcf8b"
os.makedirs(chart_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

df = pd.read_csv(csv_path)

seasons = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
season_labels = ['Winter (Jan-Feb)', 'Pre-Monsoon (Mar-May)', 'Monsoon (Jun-Sep)', 'Post-Monsoon (Oct-Dec)']
season_map = dict(zip(seasons, season_labels))

print("="*70)
print("1. OVERALL SEASONAL RAINFALL STATISTICS")
print("="*70)
overall_stats = []
for s in seasons:
    data = df[s].dropna()
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    cv_val = (std_val / mean_val) * 100
    min_val = data.min()
    max_val = data.max()
    overall_stats.append({
        'Season': season_map[s],
        'Code': s,
        'Mean (mm)': mean_val,
        'Median (mm)': median_val,
        'Std Dev (mm)': std_val,
        'CV (%)': cv_val,
        'Min (mm)': min_val,
        'Max (mm)': max_val
    })
df_overall = pd.DataFrame(overall_stats)
print(df_overall.to_string(index=False))

print("\n" + "="*70)
print("2. SEASONAL CONTRIBUTION TO ANNUAL RAINFALL")
print("="*70)
# Calculated for rows where all 4 seasons and ANNUAL are available
df_complete = df[seasons + ['ANNUAL']].dropna().copy()
for s in seasons:
    df_complete[f'{s}_share'] = (df_complete[s] / df_complete['ANNUAL']) * 100

shares = []
for s in seasons:
    mean_share = df_complete[f'{s}_share'].mean()
    median_share = df_complete[f'{s}_share'].median()
    total_vol_share = (df_complete[s].sum() / df_complete['ANNUAL'].sum()) * 100
    shares.append({
        'Season': season_map[s],
        'Mean % Share': mean_share,
        'Median % Share': median_share,
        'Volume % Share': total_vol_share
    })
df_shares = pd.DataFrame(shares)
print(df_shares.to_string(index=False))

print("\n" + "="*70)
print("3. SEASONAL RAINFALL BY SUBDIVISION (TOP & BOTTOM)")
print("="*70)
sub_season = df.groupby('SUBDIVISION')[seasons + ['ANNUAL']].mean().reset_index()
for s in seasons:
    sub_season[f'{s}_share'] = (sub_season[s] / sub_season['ANNUAL']) * 100

print("\nTop 5 subdivisions by Monsoon (Jun-Sep) rainfall:")
print(sub_season.sort_values(by='Jun-Sep', ascending=False)[['SUBDIVISION', 'Jun-Sep', 'ANNUAL', 'Jun-Sep_share']].head(5).to_string(index=False))

print("\nBottom 5 subdivisions by Monsoon (Jun-Sep) rainfall:")
print(sub_season.sort_values(by='Jun-Sep', ascending=True)[['SUBDIVISION', 'Jun-Sep', 'ANNUAL', 'Jun-Sep_share']].head(5).to_string(index=False))

print("\nTop 5 subdivisions by Post-Monsoon (Oct-Dec) rainfall (Winter retreat):")
print(sub_season.sort_values(by='Oct-Dec', ascending=False)[['SUBDIVISION', 'Oct-Dec', 'ANNUAL', 'Oct-Dec_share']].head(5).to_string(index=False))

print("\nTop 5 subdivisions by Pre-Monsoon (Mar-May) rainfall:")
print(sub_season.sort_values(by='Mar-May', ascending=False)[['SUBDIVISION', 'Mar-May', 'ANNUAL', 'Mar-May_share']].head(5).to_string(index=False))

print("\n" + "="*70)
print("4. SEASONAL VARIABILITY (CV BY SUBDIVISION)")
print("="*70)
sub_cv = df.groupby('SUBDIVISION')[seasons].apply(lambda g: (g.std() / g.mean()) * 100).reset_index()
print("Average CV across 36 subdivisions per season:")
print(sub_cv[seasons].mean())

print("\n" + "="*70)
print("5. LONG-TERM SEASONAL TRENDS (1901-2015)")
print("="*70)
# Yearly national unweighted averages
yearly_trends = df.groupby('YEAR')[seasons + ['ANNUAL']].mean().reset_index()

trend_results = []
for s in seasons + ['ANNUAL']:
    valid = yearly_trends[['YEAR', s]].dropna()
    slope, intercept, r_val, p_val, std_err = stats.linregress(valid['YEAR'], valid[s])
    # Total change over 115 years (1901 to 2015)
    total_change = slope * 114
    change_pct = (total_change / valid[s].mean()) * 100
    trend_results.append({
        'Metric': s,
        'Slope (mm/year)': slope,
        'Slope (mm/decade)': slope * 10,
        'Total Change 1901-2015 (mm)': total_change,
        'Change (%)': change_pct,
        'R-squared': r_val**2,
        'p-value': p_val
    })
df_trends = pd.DataFrame(trend_results)
print(df_trends.to_string(index=False))

# Decadal comparison: First 30 years (1901-1930) vs Last 30 years (1986-2015)
p1 = yearly_trends[(yearly_trends['YEAR'] >= 1901) & (yearly_trends['YEAR'] <= 1930)][seasons + ['ANNUAL']].mean()
p2 = yearly_trends[(yearly_trends['YEAR'] >= 1986) & (yearly_trends['YEAR'] <= 2015)][seasons + ['ANNUAL']].mean()
comp = pd.DataFrame({'1901-1930 (mm)': p1, '1986-2015 (mm)': p2, 'Diff (mm)': p2 - p1, '% Change': ((p2 - p1) / p1) * 100})
print("\n30-Year Climatological Shift (1901-1930 vs 1986-2015):")
print(comp.to_string())

# =====================================================================
# CHARTS CREATION
# =====================================================================
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Seasonal Comparison Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart of Mean & CV
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
x_pos = np.arange(len(seasons))
means = df_overall['Mean (mm)']
cvs = df_overall['CV (%)']

bars = ax1.bar(x_pos, means, color=colors, alpha=0.85, edgecolor='black', width=0.55)
ax1.set_ylabel('Mean Rainfall (mm)', fontsize=13, fontweight='bold', color='#2c3e50')
ax1.set_title('Average Seasonal Rainfall (1901–2015)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([s.replace(' ', '\n') for s in season_labels], fontsize=11)

# Annotate values and CV
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f'{yval:.1f} mm\n(CV: {cvs[i]:.0f}%)', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1250)

# Donut / Pie chart of Volume % Contribution
explode = (0.05, 0.05, 0.08, 0.05)
wedges, texts, autotexts = ax2.pie(
    df_shares['Volume % Share'], 
    labels=season_labels, 
    autopct='%1.1f%%',
    pctdistance=0.75,
    explode=explode,
    colors=colors,
    startangle=140,
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
# Draw center circle for donut
centre_circle = plt.Circle((0,0), 0.50, fc='white')
ax2.add_artist(centre_circle)
ax2.set_title('Seasonal Contribution to Annual Total Rainfall', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
p1_save = os.path.join(chart_dir, "seasonal_comparison.png")
p1_art = os.path.join(artifact_dir, "seasonal_comparison.png")
fig.savefig(p1_save, dpi=200, bbox_inches='tight')
fig.savefig(p1_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved Chart 1: {p1_save}")

# 2. Subdivision vs Season Heatmap
# Sort subdivisions by Annual rainfall for clean hierarchical view
sub_sorted = sub_season.sort_values(by='ANNUAL', ascending=False)
heatmap_data = sub_sorted.set_index('SUBDIVISION')[seasons]
heatmap_data.columns = ['Winter\n(Jan-Feb)', 'Pre-Monsoon\n(Mar-May)', 'Monsoon\n(Jun-Sep)', 'Post-Monsoon\n(Oct-Dec)']

fig, ax = plt.subplots(figsize=(14, 16))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".0f", 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Mean Rainfall (mm)'}, 
    linewidths=0.5, 
    ax=ax,
    annot_kws={"size": 9}
)
ax.set_title('Subdivision vs Season Mean Rainfall Matrix (mm)\n(Sorted from Wettest to Driest Subdivision)', fontsize=15, fontweight='bold', pad=20)
ax.set_ylabel('IMD Meteorological Subdivision', fontsize=12, fontweight='bold')
ax.set_xlabel('Season', fontsize=12, fontweight='bold')

plt.tight_layout()
p2_save = os.path.join(chart_dir, "subdivision_season_heatmap.png")
p2_art = os.path.join(artifact_dir, "subdivision_season_heatmap.png")
fig.savefig(p2_save, dpi=200, bbox_inches='tight')
fig.savefig(p2_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved Chart 2: {p2_save}")

# 3. Seasonal Trend Charts (4 subplots for the 4 seasons)
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
axes = axes.flatten()

trend_colors = ['#2980b9', '#27ae60', '#d35400', '#8e44ad']

for idx, s in enumerate(seasons):
    ax = axes[idx]
    y_raw = yearly_trends[s]
    x_yrs = yearly_trends['YEAR']
    
    # 10-year rolling mean
    rolling_10 = y_raw.rolling(window=10, center=True).mean()
    
    # Linear fit
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_yrs, y_raw)
    line_fit = intercept + slope * x_yrs
    
    ax.plot(x_yrs, y_raw, color=trend_colors[idx], alpha=0.35, linewidth=1.2, label='Annual Value')
    ax.plot(x_yrs, rolling_10, color=trend_colors[idx], linewidth=2.5, label='10-Year Rolling Mean')
    ax.plot(x_yrs, line_fit, color='black', linestyle='--', linewidth=1.8, 
            label=f'Linear Trend ({slope*10:+.2f} mm/decade, p={p_val:.3f})')
    
    ax.set_title(f'{season_labels[idx]}', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Rainfall (mm)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if idx >= 2:
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')

plt.suptitle('Seasonal Rainfall Historical Trends Across India (1901–2015)', fontsize=17, fontweight='bold', y=0.99)
plt.tight_layout()
p3_save = os.path.join(chart_dir, "seasonal_trends_1901_2015.png")
p3_art = os.path.join(artifact_dir, "seasonal_trends_1901_2015.png")
fig.savefig(p3_save, dpi=200, bbox_inches='tight')
fig.savefig(p3_art, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved Chart 3: {p3_save}")

print("\nAll seasonal analyses and visualizations successfully generated.")
