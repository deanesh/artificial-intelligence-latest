import pandas as pd
import os
import time
import numpy as np


# Logging function with timestamp

from log_utils import log

MODULE = "data_utils.py"


def read_csv_file(path):
    """Reads a CSV file and returns a DataFrame."""
    start = time.time()
    log(f"📥 Reading CSV: {path}", source=MODULE)
    try:
        df = pd.read_csv(path)
        log(f"✅ Loaded CSV with shape: {df.shape}", source=MODULE)
        return df
    except Exception as e:
        log(f"Error reading {path}: {e}", source=MODULE)
        return None
    finally:
        end = time.time()
        log(f"⏱️ Time taken to read CSV: {end - start:.2f} seconds", source=MODULE)


def clean_nulls(
    df, cat_fill="unknown", date_fill=pd.Timestamp("1900-01-01"), skew_threshold=1.0
):
    """
    Cleans missing values in a DataFrame based on column types:
    - Numeric: uses median if skewed, otherwise mean
    - Categorical: fills with a string like 'unknown'
    - Datetime: fills with a default Timestamp

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cat_fill (str): Fill value for categorical columns
        date_fill (Timestamp): Fill value for datetime columns
        skew_threshold (float): Skewness threshold to decide mean vs. median

    Returns:
        pd.DataFrame: DataFrame with nulls filled
    """

    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        log("✅ No nulls found in the DataFrame. Nothing to clean.", source=MODULE)
        return df.copy()

    df_copy = df.copy()

    filled_columns = {
        "mean": [],
        "median": [],
        "categorical": [],
        "datetime": [],
        "skipped": [],
    }

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                skew = df[col].skew()
                if abs(skew) > skew_threshold:
                    fill_value = df[col].median()
                    df_copy[col] = df[col].fillna(fill_value)
                    filled_columns["median"].append(col)
                else:
                    fill_value = df[col].mean()
                    df_copy[col] = df[col].fillna(fill_value)
                    filled_columns["mean"].append(col)

            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                df_copy[col] = df[col].fillna(cat_fill)
                filled_columns["categorical"].append(col)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df_copy[col] = df[col].fillna(date_fill)
                filled_columns["datetime"].append(col)

            else:
                filled_columns["skipped"].append(col)

    # ✅ log final summary
    log(f"⚠️ Found {total_missing} missing values in the DataFrame.", source=MODULE)
    log("🧹 Missing values filled using the following strategies:", source=MODULE)

    if filled_columns["mean"]:
        log(f"   • Mean fill: {filled_columns['mean']}", source=MODULE)
    if filled_columns["median"]:
        log(f"   • Median fill: {filled_columns['median']}", source=MODULE)
    if filled_columns["categorical"]:
        log(f"   • Categorical fill: {filled_columns['categorical']}", source=MODULE)
    if filled_columns["datetime"]:
        log(f"   • Datetime fill: {filled_columns['datetime']}", source=MODULE)
    if filled_columns["skipped"]:
        log(
            f"   ⚠️ Skipped unsupported columns: {filled_columns['skipped']}",
            source=MODULE,
        )

    return df_copy


def standardize_columns(df):
    """Strips and lowercases column names."""
    log("🧹 Standardizing column names", source=MODULE)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    return df


def check_duplicates(df, drop=False, subset=None, keep="first", verbose=True):
    """
    Checks for duplicate rows in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        drop (bool): Whether to drop duplicates (default: False)
        subset (list or str): Columns to consider for duplicate detection
        keep (str): Which duplicate to keep - 'first', 'last', or False (drop all)
        verbose (bool): Whether to log summary info

    Returns:
        pd.DataFrame: Original or de-duplicated DataFrame
    """
    dup_mask = df.duplicated(subset=subset, keep=keep)
    num_dupes = dup_mask.sum()

    if verbose:
        if subset:
            log(
                f"🔍 Found {num_dupes} duplicate rows based on columns: {subset}",
                source=MODULE,
            )
        else:
            log(f"🔍 Found {num_dupes} completely duplicate rows.", source=MODULE)

    if drop and num_dupes > 0:
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        if verbose:
            log(f"✅ Dropped {num_dupes} duplicate rows.", source=MODULE)
        return df_cleaned

    return df


def log_stage(stage: str, is_start=True):
    log(f"{stage} - {'Start' if is_start else 'End'}", source=MODULE)


def preprocess_data(csv_path: str):
    """Full data preprocessing pipeline."""
    log("🚀 Starting full data preprocessing pipeline", source=MODULE)
    start = time.time()
    log_stage(f"Loading data file : {os.path.basename(csv_path)}", True)
    df = read_csv_file(csv_path)
    if df is None:
        log("❌ Cannot preprocess — file not loaded.")
        return None
    log_stage(f"Loading data file : {os.path.basename(csv_path)}", False)

    log_stage("Standardizing Columns", True)
    df = standardize_columns(df)
    log_stage("Standardizing Columns", False)

    log_stage("Checking Duplicates", True)
    df = check_duplicates(df)
    log_stage("Checking Duplicates", False)

    log_stage("Cleaning Nulls", True)
    df_cleaned = clean_nulls(df)
    log_stage("Cleaning Nulls", False)

    end = time.time()
    log(f"✅ Data preprocessing completed in {end - start:.2f} seconds", source=MODULE)

    return df_cleaned


def save_cleaned_data(df: pd.DataFrame, csv_path: str):
    log_stage(f"Saving Cleaned Data file: {os.path.basename(csv_path)} ", True)
    df.to_csv(csv_path, index=False)
    log_stage(f"Saved Cleaned Data file : {os.path.basename(csv_path)}", True)


def get_dataframe_by_partial_file_name(directory: str, partial_file_name: str):
    log("🚀 Fetching data frame from partial file name", source=MODULE)
    start = time.time()

    # Ensure directory is valid
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Get matching files
    matched_files = [
        f
        for f in os.listdir(directory)
        if partial_file_name in f and f.endswith(".csv")
    ]

    if not matched_files:
        raise FileNotFoundError(
            f"No file containing '{partial_file_name}' found in {directory}"
        )

    # Get the latest matched file by modification time
    matched_files_full_paths = [os.path.join(directory, f) for f in matched_files]
    latest_file = max(matched_files_full_paths, key=os.path.getmtime)

    log(f"📄 Matched file: {os.path.basename(latest_file)}", source=MODULE)

    # Read the CSV
    df = read_csv_file(latest_file)

    end = time.time()
    log(f"✅ Fetching DataFrame completed in {end - start:.2f} seconds", source=MODULE)

    return df


def get_cat_and_con_cols_list(df: pd.DataFrame):
    log("🚀 Fetching cat and cont columns from dataframe ", source=MODULE)
    cat_cols = df.select_dtypes(include="object").columns.to_list()
    con_cols = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
    log("🚀 Fetched cat and cont columns from dataframe ", source=MODULE)
    return cat_cols, con_cols



