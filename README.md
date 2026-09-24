# Airbnb Price Prediction — Machine Learning & EDA

An end-to-end data science project designed to predict the logarithm of Airbnb listing prices (`log_price`) across major US metropolitan areas using a combination of numerical, categorical, geographic, and textual features.

---

## 📌 Project Overview

Pricing short-term rentals accurately is critical for both hosts and platforms. This project explores listing dynamics across multiple US cities (New York, Los Angeles, San Francisco, Washington D.C., Boston, and Chicago) and builds an iterative machine learning pipeline to forecast listing prices.

The target variable is `log_price` (log-transformed price in USD), which addresses the right-skewness of raw prices and optimizes regression performance.

---

## 📊 Dataset Summary

* **Training Set:** 22,234 listings, 28 columns
* **Test Set:** 51,877 listings, 27 columns
* **Target:** `log_price` (mean $\approx 4.78$, median $\approx 4.70$, corresponding to a median raw price of ~$110)
* **Key Features:**
  * **Categorical / Metadata:** `room_type`, `property_type`, `city`, `neighbourhood`, `cancellation_policy`, `bed_type`
  * **Capacity & Layout:** `accommodates`, `bedrooms`, `beds`, `bathrooms`
  * **Text & Amenities:** `name`, `description`, `amenities` (raw JSON/set string)
  * **Geospatial:** `latitude`, `longitude`, `zipcode`
  * **Host & Reviews:** `number_of_reviews`, `review_scores_rating`, `last_review`

---

## 🔬 Methodology & Workflow

### 1. Exploratory Data Analysis (EDA)
* **Target Distribution:** Evaluated `log_price` normality versus raw skewed dollar amounts.
* **Geographic & Category Stratification:** Analyzed price disparities across cities (e.g., San Francisco and Boston commanding highest medians) and room configurations (`Entire home/apt` vs. `Private room` vs. `Shared room`).
* **Feature Correlation:** Quantified relationships between property capacity (`accommodates`, `bedrooms`) and rental price.
* **Amenities Parsing:** Extracted high-impact amenities from unstructured text.
* **Missing Value Treatment:** Imputation strategies for ratings, review dates, and missing structural counts.

### 2. Feature Engineering & Preprocessing
* **High-Cardinality Categoricals:** Applied target encoding via `category_encoders.TargetEncoder` for `neighbourhood`, `zipcode`, and `property_type`.
* **One-Hot Encoding:** Applied to low-cardinality nominal features (`room_type`, `cancellation_policy`, `city`).
* **Text / Unstructured Data:** Extracted binary flags and frequency indicators from `amenities`.
* **Scaling:** Standardized continuous features for distance- and margin-based algorithms.

### 3. Model Progression & Benchmarking
The predictive modeling follows an iterative, staged approach:
* **Baseline:** `DummyRegressor` (mean/median strategy)
* **V1 (Linear Model):** `LinearSVR` pipeline with standard scaling and one-hot encoding
* **V2 (Tree Ensembles):** `RandomForestRegressor` and baseline `XGBRegressor`
* **V3 (Optimized Final Model):** Fine-tuned `XGBoost` with early stopping and cross-validation

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Data Manipulation & Stats:** `pandas`, `numpy`, `scipy`, `statsmodels`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`, `category_encoders`

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<Lucille865>/<AirbnbProject>.git
   cd <AirbnbProject>
