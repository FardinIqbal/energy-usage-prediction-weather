# Energy Usage Prediction and Weather Analytics

Advanced Data Engineering, Statistical Learning & Appliance‑Level Load Characterisation

---

## Abstract

This project delivers an end‑to‑end analytical pipeline that quantifies and models the relationship between residential energy demand and local meteorological conditions. Hourly smart‑meter data are merged with high‑resolution weather observations, aggregated to daily granularity, and used to build two supervised learning systems:

* **Multiple Linear Regression** for quantitative forecasting of daily energy consumption.
* **Logistic Regression** for binary discrimination of high‑temperature days (≥ 35 °C).

A supplementary load‑disaggregation experiment characterises diurnal appliance behaviour (refrigerator and dryer), revealing load‑shifting opportunities for demand‑response programmes.

---

## Repository Layout

```text
.
├── data/                         # Raw and processed CSV artefacts
│   └── merged_energy_weather.csv
├── notebook/
│   └── energy_weather_analysis.ipynb
├── script/
│   └── predict_energy_usage.py
├── results/
│   ├── energy_usage_predictions_linear.csv
│   ├── high_temperature_classification_logistic.csv
│   ├── fridge_usage_day_night.png
│   └── dryer_usage_day_night.png
├── report/
│   └── energy_weather_report.pdf
├── .gitignore
└── README.md
```

---

## Data Sources

| Stream  | Granularity                     | Period              | Key Fields                                      | Imputation      |
| ------- | ------------------------------- | ------------------- | ----------------------------------------------- | --------------- |
| Energy  | Hourly smart‑meter (`use [kW]`) | 1 Jan – 31 Dec 2014 | Device‑level sub‑circuits                       | None (complete) |
| Weather | 5‑min API snapshots (DarkSky)   | 1 Jan – 31 Dec 2014 | Temperature, RH, Pressure, Wind, Cloud, Precip. | None (complete) |

Hourly observations were synchronised to daily resolution via mean (weather) and sum (energy). The outer merge on `date` produced 365 fully populated records (zero missingness).

---

## Feature Engineering

* Thermodynamic: temperature, dew‑point
* Atmospheric: pressure, precipitation intensity/probability, cloud cover
* Optical: visibility
* Kinematic: wind speed, wind bearing
* Temporal: Unix epoch time

Wind bearing was examined with sine–cosine decomposition; retained raw after no material performance gain. Continuous predictors were z‑standardised where required.

---

## Modelling Protocol

### Regression Task

| Item         | Specification                  |
| ------------ | ------------------------------ |
| Train window | 334 days (Jan–Nov)             |
| Test window  | 31 days (Dec)                  |
| Estimator    | Ordinary Least Squares         |
| Validation   | Forward hold‑out               |
| Metric       | Root Mean Squared Error (RMSE) |
| Result       | RMSE = **10.73 kW**            |

### Classification Task

| Item                  | Specification                                        |
| --------------------- | ---------------------------------------------------- |
| Positive class        | `temperature ≥ 35 °C`                                |
| Class balance (train) | 24 % low / 76 % high                                 |
| Estimator             | Logistic Regression (`liblinear`, `max_iter = 1000`) |
| Metric                | F1‑Score                                             |
| Result                | F1 = **0.5909**                                      |

Class imbalance was preserved to reflect the natural prior; no oversampling applied.

---

## Appliance Load Characterisation

| Device       | Day (06:00–18:00) kWh | Night (19:00–05:00) kWh | Day : Night |
| ------------ | --------------------- | ----------------------- | ----------- |
| Refrigerator | 721.76                | 567.04                  | 1.27        |
| Dryer        | 981.84                | 228.77                  | 4.29        |

The dryer’s pronounced daytime peak indicates scope for demand‑response or TOU tariff optimisation.

---

## Reproducibility

1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Execute full pipeline

```bash
python script/predict_energy_usage.py
```

3. Interactive exploration

```bash
jupyter lab notebook/energy_weather_analysis.ipynb
```

Generated artefacts will reproduce under `results/`.

---

## Requirements

* Python ≥ 3.11
* pandas ≥ 2.2
* numpy ≥ 1.26
* scikit‑learn ≥ 1.5
* matplotlib ≥ 3.9
* jupyterlab ≥ 4.1

---

## Future Work

* Non‑linear learners (Gradient Boosting, Random Forest, XGBoost).
* Holiday and event covariates to mitigate December residuals.
* Hourly forecasting using sequence models (LSTM, N‑BEATS).
* Cost‑sensitive or β‑F1 optimisation to address class imbalance.

---

## Citation

Iqbal, F. (2025). *Quantifying Residential Energy Demand via Weather‑Driven Statistical Learning*. GitHub repository: [https://github.com/FardinIqbal/energy-usage-prediction-weather](https://github.com/FardinIqbal/energy-usage-prediction-weather)

---

## License

Distributed under the MIT License.
