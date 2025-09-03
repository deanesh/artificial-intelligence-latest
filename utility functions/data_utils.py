import pandas as pd
import numpy as np
import os
import time
import inspect

from datetime import datetime

# Logging function with timestamp
from datetime import datetime as dt


def log(msg):
    # Get the name of the calling file/module
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    caller = os.path.basename(module.__file__) if module and hasattr(module, "__file__") else "JupyterNotebook"

    print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} | {caller} | {msg}")


def read_csv_file(path):
    """Reads a CSV file and returns a DataFrame."""
    start = time.time()
    log(f"📥 Reading CSV: {path}")
    try:
        df = pd.read_csv(path)
        log(f"✅ Loaded CSV with shape: {df.shape}")
        return df
    except Exception as e:
        log(f"Error reading {path}: {e}")
        return None
    finally:
        end = time.time()
        log(f"⏱️ Time taken to read CSV: {end - start:.2f} seconds")


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
        log("✅ No nulls found in the DataFrame. Nothing to clean.")
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
    log(f"⚠️ Found {total_missing} missing values in the DataFrame.")
    log("🧹 Missing values filled using the following strategies:")

    if filled_columns["mean"]:
        log(f"   • Mean fill: {filled_columns['mean']}")
    if filled_columns["median"]:
        log(f"   • Median fill: {filled_columns['median']}")
    if filled_columns["categorical"]:
        log(f"   • Categorical fill: {filled_columns['categorical']}")
    if filled_columns["datetime"]:
        log(f"   • Datetime fill: {filled_columns['datetime']}")
    if filled_columns["skipped"]:
        log(f"   ⚠️ Skipped unsupported columns: {filled_columns['skipped']}")

    return df_copy


def standardize_columns(df):
    """Strips and lowercases column names."""
    log("🧹 Standardizing column names")
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
            log(f"🔍 Found {num_dupes} duplicate rows based on columns: {subset}")
        else:
            log(f"🔍 Found {num_dupes} completely duplicate rows.")

    if drop and num_dupes > 0:
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        if verbose:
            log(f"✅ Dropped {num_dupes} duplicate rows.")
        return df_cleaned

    return df


def log_stage(stage: str, is_start=True):
    log(f"{stage} - {'Start' if is_start else 'End'}")


def preprocess_data(csv_path: str):
    """Full data preprocessing pipeline."""
    log("🚀 Starting full data preprocessing pipeline")
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
    log(f"✅ Data preprocessing completed in {end - start:.2f} seconds")

    return df_cleaned


def save_cleaned_data(df: pd.DataFrame, csv_path: str):
    log_stage(f"Saving Cleaned Data file: {os.path.basename(csv_path)} ", True)
    df.to_csv(csv_path, index=False)
    log_stage(f"Saved Cleaned Data file : {os.path.basename(csv_path)}", True)
