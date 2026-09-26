# 🌧️ Rain Prediction in India

## 📌 Project Overview

This project analyzes historical rainfall patterns across Indian subdivisions using rainfall data from **1901–2015** and generates a machine-learning-based rainfall forecast for **2016**.

An interactive **Power BI dashboard** is included for historical, seasonal, regional, and prediction analysis.

> **Note:** 2016 values are machine-learning forecasts, not observed rainfall.

## 🎯 Objectives

- Analyze historical rainfall trends in India.
- Study seasonal rainfall patterns.
- Compare rainfall across Indian subdivisions.
- Analyze regional rainfall variability.
- Build a machine-learning model for rainfall prediction.
- Forecast rainfall for 2016.
- Create an interactive Power BI dashboard.

## 📊 Dataset

The dataset contains Indian rainfall information from **1901–2015**, including monthly, seasonal, annual, and subdivision-level rainfall.

### Seasons

- Winter: January–February
- Pre-Monsoon: March–May
- Monsoon: June–September
- Post-Monsoon: October–December

## 🧹 Data Processing

The project includes:

- Data cleaning and preprocessing
- Duplicate checking
- Missing-value analysis
- Seasonal and annual rainfall validation
- Time-series feature engineering
- Data leakage checks

## 🤖 Machine Learning

The project uses historical rainfall features such as:

- Annual rainfall lags
- Rolling rainfall averages
- Rolling standard deviation
- Previous-year seasonal rainfall

### Model

**Random Forest Regressor**

The model generates an out-of-sample rainfall forecast for **2016** at the Indian subdivision level using information available through 2015.

## 🔮 2016 Forecast

The forecast output includes:

- Subdivision
- 2015 observed rainfall
- Historical mean rainfall
- Predicted 2016 rainfall
- Difference from historical mean
- Percentage difference

Output file:

`forecast_2016.csv`

## 📈 Power BI Dashboard

The Power BI dashboard contains four pages:

### 1. Rainfall Overview
Historical rainfall trends and subdivision comparison.

### 2. Seasonal Analysis
Seasonal rainfall trends and monsoon contribution.

### 3. Regional Analysis
Regional rainfall statistics and variability.

### 4. 2016 Prediction
2015 observed vs 2016 predicted rainfall, historical mean comparison, and forecast details.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Matplotlib
- Power BI
- DAX
- GitHub

## ⚙️ How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
Run the Python scripts in sequence:

python clean_pipeline.py
python build_ml_dataset.py
python test_ml_dataset.py
python train_models.py
python forecast_2016_pipeline.py

Open Rain_Prediction_India_2016.pbix in Microsoft Power BI Desktop to view the dashboard.

📁 Project Files
clean_pipeline.py — Data cleaning
build_ml_dataset.py — ML dataset creation
test_ml_dataset.py — Dataset validation
train_models.py — Model training
forecast_2016_pipeline.py — 2016 forecast
seasonal_analysis.py — Seasonal analysis
subdivision_analysis.py — Regional analysis
final_report_check.py — Final validation
forecast_2016.csv — Forecast results
Rain_Prediction_India_2016.pbix — Power BI dashboard
⚠️ Limitation

The source dataset ends in 2015, so 2016 values are model-generated forecasts and should not be interpreted as actual observed rainfall.
