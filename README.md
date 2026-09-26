# Rain Prediction in India 🌧️

## 📌 Project Overview

This project analyzes historical rainfall patterns across Indian subdivisions and provides a machine-learning-based forecast of rainfall for the year **2016**.

The project uses historical rainfall data from **1901–2015** to identify annual, seasonal, and regional rainfall patterns and generate an out-of-sample rainfall forecast for Indian subdivisions.

An interactive **Power BI dashboard** is included to visualize the historical analysis, seasonal patterns, regional statistics, and 2016 rainfall predictions.

> **Important:** The 2016 rainfall values shown in this project are machine-learning model forecasts, not observed rainfall. The prediction uses information available through 2015 only.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze historical rainfall patterns in India.
- Study annual rainfall trends from 1901 to 2015.
- Analyze rainfall across different seasons.
- Compare rainfall patterns between Indian subdivisions.
- Identify rainfall variability across regions.
- Build a machine-learning pipeline for rainfall prediction.
- Forecast rainfall for 2016 at the Indian subdivision level.
- Present the analysis through an interactive Power BI dashboard.

---

## 📊 Dataset

The project uses historical rainfall data covering the period:

**1901–2015**

The dataset contains rainfall information for Indian subdivisions, including:

- Monthly rainfall
- Annual rainfall
- Seasonal rainfall
- Subdivision
- Year

### Seasons Used

| Season | Months |
|---|---|
| Winter | January–February |
| Pre-Monsoon | March–May |
| Monsoon | June–September |
| Post-Monsoon | October–December |

The project also includes a district-wise rainfall normal dataset for regional comparison and analysis.

---

## 🧹 Data Cleaning and Preprocessing

The data preprocessing pipeline includes:

- Removing exact duplicate records.
- Checking duplicate `(SUBDIVISION, YEAR)` combinations.
- Standardizing subdivision names.
- Converting year values to integer format.
- Converting rainfall columns to numeric values.
- Preserving genuine zero-rainfall observations.
- Preserving missing rainfall values as missing rather than treating them as zero.
- Recalculating seasonal rainfall totals from monthly rainfall where appropriate.
- Validating annual rainfall against monthly rainfall totals.
- Checking for negative rainfall values.
- Checking missing years and missing rainfall values.

The cleaning process is implemented in:

```text
clean_pipeline.py
