from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

from log_utils import log
import time

# Logging function with timestamp

MODULE = "correlation_utils.py"


def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))


def imp_cat_cols(df: pd.DataFrame, cat_cols: list, target_col: str, threshold: float):
    start = time.time()
    log(
        f"🚀 Fetching important categorical columns for target: {target_col}",
        source=MODULE,
    )

    results = {}
    for col in cat_cols:
        if col == target_col:
            continue
        try:
            results[col] = cramers_v(df[col], df[target_col])
        except Exception as e:
            log(f"⚠️ Error computing Cramér’s V for {col}: {e}", source=MODULE)
            results[col] = np.nan

    cramers_df = (
        pd.DataFrame.from_dict(results, orient="index", columns=["Cramers_V"])
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    unleaded = cramers_df[cramers_df["Cramers_V"] >= threshold].sort_values(
        "Cramers_V", ascending=False
    )

    end = time.time()
    log(
        f"✅ Completed categorical feature analysis in {end - start:.2f} secs",
        source=MODULE,
    )
    return unleaded
