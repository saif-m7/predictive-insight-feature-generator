import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/cleaned_dataset.csv"

OUTPUT_DATASET = "data/feature_engineered_dataset.csv"

CORRELATION_REPORT = "outputs/reports/feature_correlations.csv"

HEATMAP_FILE = "outputs/figures/correlation_heatmap.png"


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """Load the cleaned Week 1 dataset."""

    df = pd.read_csv(file_path)

    print("========== LOADING DATASET ==========")
    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# 2. PREPARE DATE FEATURES
# ============================================================

def create_date_features(df):
    """
    Convert date columns into meaningful numerical features
    instead of one-hot encoding every individual date.
    """

    df = df.copy()

    date_columns = [
        "signup_date",
        "last_purchase_date"
    ]

    for column in date_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day
            df[f"{column}_day_of_week"] = (
                df[column].dt.dayofweek
            )

    # Calculate customer tenure in days.
    if (
        "signup_date" in df.columns
        and "last_purchase_date" in df.columns
    ):

        df["customer_tenure_days"] = (
            df["last_purchase_date"]
            - df["signup_date"]
        ).dt.days

    # Remove raw date columns.
    df = df.drop(
        columns=[
            column
            for column in date_columns
            if column in df.columns
        ]
    )

    print("\n========== DATE FEATURES ==========")

    print("Created date-derived features:")
    print("- signup_date_year")
    print("- signup_date_month")
    print("- signup_date_day")
    print("- signup_date_day_of_week")
    print("- last_purchase_date_year")
    print("- last_purchase_date_month")
    print("- last_purchase_date_day")
    print("- last_purchase_date_day_of_week")
    print("- customer_tenure_days")

    return df


# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================

def calculate_correlations(df):
    """Calculate Pearson correlations for numerical features."""

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    correlation_matrix = (
        df[numerical_columns]
        .corr(method="pearson")
    )

    churn_correlation = (
        correlation_matrix["churn"]
        .sort_values(ascending=False)
    )

    print("\n========== CORRELATION WITH CHURN ==========")
    print(churn_correlation)

    return correlation_matrix


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_engineered_features(df):
    """
    Create calculated domain features based on
    correlation-driven feature selection.
    """

    df = df.copy()

    # Feature 1:
    # Spending intensity relative to customer visits.
    df["spending_per_visit"] = (
        df["total_spent"]
        /
        df["total_visits"].replace(0, np.nan)
    )

    # Feature 2:
    # Relative support burden based on customer activity.
    df["support_burden"] = (
        df["support_tickets"]
        /
        df["total_visits"].replace(0, np.nan)
    )

    df["spending_per_visit"] = (
        df["spending_per_visit"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["support_burden"] = (
        df["support_burden"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    print("\n========== ENGINEERED FEATURES ==========")

    print("1. spending_per_visit")
    print("   Formula: total_spent / total_visits")

    print("2. support_burden")
    print("   Formula: support_tickets / total_visits")

    return df


# ============================================================
# 5. VERIFY ENGINEERED FEATURES
# ============================================================

def verify_engineered_features(df):
    """Check correlations of new features with churn."""

    engineered_features = [
        "spending_per_visit",
        "support_burden"
    ]

    correlations = (
        df[
            engineered_features + ["churn"]
        ]
        .corr(method="pearson")["churn"]
        .drop("churn")
        .sort_values(ascending=False)
    )

    print(
        "\n========== ENGINEERED FEATURE CORRELATIONS =========="
    )

    print(correlations)

    return correlations


# ============================================================
# 6. SAVE CORRELATION REPORT
# ============================================================

def save_correlation_report(
    correlation_matrix,
    output_path
):
    """Save the complete correlation matrix."""

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    correlation_matrix.to_csv(output_path)

    print(
        f"\nCorrelation report saved to: {output_path}"
    )


# ============================================================
# 7. GENERATE CORRELATION HEATMAP
# ============================================================

def generate_correlation_heatmap(
    correlation_matrix,
    output_path
):
    """Generate and save correlation heatmap."""

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        annot=False
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Correlation heatmap saved to: {output_path}"
    )


# ============================================================
# 8. PREPARE FEATURE COLUMNS
# ============================================================

def prepare_feature_columns(df):
    """Identify numerical and categorical features."""

    numerical_columns = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    # Target variable should not be used as an input feature.
    if "churn" in numerical_columns:
        numerical_columns.remove("churn")

    # Customer ID is an identifier, not a predictive feature.
    if "customer_id" in numerical_columns:
        numerical_columns.remove("customer_id")

    categorical_columns = (
        df
        .select_dtypes(include="object")
        .columns
        .tolist()
    )

    return numerical_columns, categorical_columns


# ============================================================
# 9. ENCODE AND SCALE
# ============================================================

def transform_features(
    df,
    numerical_columns,
    categorical_columns
):
    """
    Apply StandardScaler to numerical features and
    OneHotEncoder to categorical features.
    """

    X = df[
        numerical_columns + categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ]
    )

    feature_matrix = (
        preprocessor.fit_transform(X)
    )

    categorical_feature_names = (
        preprocessor
        .named_transformers_["categorical"]
        .get_feature_names_out(
            categorical_columns
        )
    )

    feature_names = (
        numerical_columns
        +
        list(categorical_feature_names)
    )

    transformed_df = pd.DataFrame(
        feature_matrix,
        columns=feature_names
    )

    return transformed_df


# ============================================================
# 10. SAVE FINAL FEATURE MATRIX
# ============================================================

def save_feature_matrix(
    feature_matrix,
    output_path
):
    """Save the final ML-ready feature matrix."""

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    feature_matrix.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nFeature matrix saved to: {output_path}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n==============================================")
    print("   PREDICTIVE INSIGHT FEATURE GENERATOR")
    print("==============================================")

    # Step 1: Load Week 1 cleaned data.
    df = load_dataset(INPUT_FILE)

    original_shape = df.shape

    # Step 2: Convert dates into useful features.
    df = create_date_features(df)

    # Step 3: Correlation analysis.
    correlation_matrix = calculate_correlations(df)

    # Step 4: Create two calculated features.
    df = create_engineered_features(df)

    # Step 5: Verify their correlations.
    verify_engineered_features(df)

    # Step 6: Recalculate correlation matrix
    # including engineered features.
    numerical_columns_for_correlation = (
        df
        .select_dtypes(include=np.number)
        .columns
    )

    updated_correlation_matrix = (
        df[numerical_columns_for_correlation]
        .corr(method="pearson")
    )

    # Step 7: Save correlation report.
    save_correlation_report(
        updated_correlation_matrix,
        CORRELATION_REPORT
    )

    # Step 8: Generate correlation heatmap.
    generate_correlation_heatmap(
        updated_correlation_matrix,
        HEATMAP_FILE
    )

    # Step 9: Identify numerical and categorical features.
    numerical_columns, categorical_columns = (
        prepare_feature_columns(df)
    )

    print("\n========== FEATURE COLUMNS ==========")

    print("\nNumerical features:")
    print(numerical_columns)

    print("\nCategorical features:")
    print(categorical_columns)

    # Step 10: Encode categorical features
    # and scale numerical features.
    feature_matrix = transform_features(
        df,
        numerical_columns,
        categorical_columns
    )

    # Step 11: Save final matrix.
    save_feature_matrix(
        feature_matrix,
        OUTPUT_DATASET
    )

    # Step 12: Display transformation shapes.
    print("\n========== MATRIX SHAPES ==========")

    print(
        f"Original dataset shape: {original_shape}"
    )

    print(
        f"After date feature engineering: {df.shape}"
    )

    print(
        f"Final feature matrix shape: "
        f"{feature_matrix.shape}"
    )

    print("\n========== FINAL FEATURE MATRIX ==========")
    print(feature_matrix.head())

    print("\n==============================================")
    print(" FEATURE ENGINEERING PIPELINE COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()