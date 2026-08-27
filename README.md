# Predictive Insight Feature Generator

## Week 2 — Feature Engineering & Statistical Analytics

A modular Python feature engineering pipeline that transforms the sanitized customer Sales & Marketing dataset from Week 1 into a machine-learning-ready feature matrix.

The pipeline performs correlation analysis, correlation-driven feature synthesis, categorical encoding, numerical feature scaling, and feature matrix generation.

---

## Project Objective

The objective of this project is to transform clean data into high-signal numerical features suitable for downstream machine learning algorithms.

The pipeline implements:

- Pearson correlation analysis
- Correlation-driven feature engineering
- Date feature engineering
- Categorical one-hot encoding
- Numerical Z-score standardization
- Feature matrix generation
- Transformation validation
- Automated correlation reporting
- Correlation heatmap generation

---

## Input Dataset

The input dataset is the cleaned output generated during Week 1.

| Property | Value |
|---|---:|
| Records | 15,000 |
| Original Features | 30 |
| Domain | Customer Sales & Marketing |
| Format | CSV |
| Source | Week 1 Cleaning Pipeline |

---

## Correlation Analysis

Pearson correlation was used to analyze relationships between numerical features and the target variable `churn`.

The strongest observed correlations were:

| Feature | Correlation with Churn |
|---|---:|
| `satisfaction_score` | -0.299775 |
| `total_spent` | -0.158525 |
| `support_tickets` | 0.127910 |

These relationships were used to guide the selection of variables for feature synthesis.

---

## Correlation-Driven Feature Engineering

Two calculated ratio features were engineered based on the observed relationships in the dataset.

### 1. Spending Per Visit

Formula:

```text
spending_per_visit = total_spent / total_visits

This feature measures customer spending intensity relative to the number of customer visits.

Correlation with churn:

-0.150776

The negative correlation indicates that higher spending per visit is associated with lower churn in this dataset.

2. Support Burden

Formula:

support_burden = support_tickets / total_visits

This feature measures the relative level of customer support interaction compared with customer activity.

Correlation with churn:

0.106530

The positive correlation indicates that higher support burden is associated with higher churn in this dataset.

Both features are generated programmatically and can be reproduced automatically for new datasets following the same schema.

Date Feature Engineering

Raw date columns are transformed into meaningful numerical features instead of being one-hot encoded individually.

The pipeline creates:

Signup year
Signup month
Signup day of week
Last purchase year
Last purchase month
Last purchase day of week
Customer tenure in days

Customer tenure is calculated as:

customer_tenure_days = last_purchase_date - signup_date

The original date columns are removed after transformation to avoid unnecessary dimensionality growth.

Categorical Encoding

Categorical variables are converted into numerical representations using Scikit-learn's OneHotEncoder.

The encoder uses:

handle_unknown="ignore"

This allows the transformation pipeline to handle previously unseen categories without failing.

Categorical variables include:

Gender
Country
City
Acquisition channel
Device type
Subscription type
Coupon code
Payment method
Numerical Feature Scaling

Numerical features are standardized using StandardScaler.

The Z-score formula is:

z = (x - μ) / σ

where:

x = original feature value
μ = feature mean
σ = feature standard deviation

Scaling validation confirmed that the transformed numerical features have:

Mean ≈ 0
Standard deviation ≈ 1
Matrix Transformation

The feature transformation produced the following results:

Transformation Stage	Matrix Shape
Original Dataset	15,000 × 30
After Date Feature Engineering	15,000 × 37
After Correlation-Driven Feature Synthesis	15,000 × 39
Final Encoded & Scaled Feature Matrix	15,000 × 61

All 15,000 customer records were preserved throughout the transformation process.

The final feature matrix contains only numerical values and is ready for downstream machine learning algorithms.

Correlation Heatmap

The pipeline generates a Pearson correlation heatmap to visualize relationships among numerical features.

Project Structure
predictive-insight-feature-generator/
│
├── data/
│   ├── cleaned_dataset.csv
│   ├── feature_engineered_dataset.csv
│   └── target_churn.csv
│
├── outputs/
│   ├── figures/
│   │   └── correlation_heatmap.png
│   │
│   └── reports/
│       └── feature_correlations.csv
│
├── src/
│   └── features.py
│
├── notebooks/
│   └── Week2_Predictive_Insight_Feature_Generator_ipynb.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
Technologies
Python 3.10
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Google Colab
Git & GitHub
Dependencies

The required Python packages are listed in requirements.txt:

pandas
numpy
matplotlib
seaborn
scikit-learn
How to Run Locally
1. Clone the Repository
git clone https://github.com/saif-m7/predictive-insight-feature-generator.git
2. Navigate to the Project
cd predictive-insight-feature-generator
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1
5. Install Dependencies
pip install -r requirements.txt
6. Run the Feature Engineering Pipeline
python src/features.py

The pipeline generates:

Feature-engineered dataset → data/feature_engineered_dataset.csv
Target variable → data/target_churn.csv
Correlation report → outputs/reports/feature_correlations.csv
Correlation heatmap → outputs/figures/correlation_heatmap.png
Google Colab

The complete feature engineering pipeline has been executed in Google Colab with visible outputs demonstrating:

Week 1 dataset loading
Dataset inspection
Pearson correlation analysis
Correlation heatmap
Feature synthesis
Engineered feature validation
One-hot encoding
Numerical scaling
Matrix shape comparison
Scaling validation
Output generation
Google Colab Notebook

Open the Google Colab Notebook

Engineering Validation

The pipeline satisfies the Week 2 engineering requirements:

Correlation analysis is performed programmatically.
Two calculated domain features are generated programmatically.
Categorical variables are encoded using OneHotEncoder.
Numerical variables are standardized using StandardScaler.
Date values are transformed into meaningful features.
Raw dates are not individually one-hot encoded, avoiding unnecessary dimensionality growth.
Feature transformations are modular and repeatable.
All 15,000 records are preserved.
Correlation reports and visualizations are automatically generated.
Conclusion

The Predictive Insight Feature Generator successfully transforms the Week 1 sanitized dataset into a machine-learning-ready numerical feature matrix.

The pipeline generated two correlation-driven features, applied categorical encoding and numerical standardization, avoided unnecessary dimensionality growth from raw dates, and produced a final feature matrix of 15,000 × 61.

The resulting pipeline provides a reproducible foundation for downstream machine learning model development.