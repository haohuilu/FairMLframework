import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Configuration
# ==========================
DATASETS = {
    # D7: remove 81 males from class 0
    "D7": {
        "path": "D7.csv",
        "gender_col": "gender",                 # 1=Male, 2=Female
        "label_col": "cardio",                  # 0/1
        "gender_map": {1: 1, 2: 0},             # Male=1, Female=0
        "removal": {"gender": 1, "label": 0, "n": 162}, #81
    },
    # D58: remove 17,213 females from class 0
    "D58": {
        "path": "D58.csv",
        "gender_col": "Sex",                    # 0=Female, 1=Male
        "label_col": "Diabetes_binary",         # 0/1
        "gender_map": {0: 0, 1: 1},             # Male=1, Female=0
        "removal": {"gender": 0, "label": 0, "n": 34426}, #17213
    },
    # D73: remove 460 males from class 0 (0=dead)
    "D73": {
        "path": "D73.csv",
        "gender_col": "sex_0male_1female",      # 0=Male, 1=Female
        "label_col": "hospital_outcome_1alive_0dead",  # 1=alive, 0=dead
        "gender_map": {0: 1, 1: 0},             # Male=1, Female=0
        "removal": {"gender": 1, "label": 0, "n": 920},#460
    },
}

REMOVAL_DESCRIPTIONS = {
    "D7":  "81, Male from 0 class",
    "D58": "17,213, female from 0 class",
    "D73": "460, male from 0 class",
}

# ==========================
# Helpers
# ==========================
def fmt_p(p, threshold=0.05):
    try:
        if p is None or np.isnan(p) or p > threshold:
            return "…"
        return f"{p:.4f}"
    except Exception:
        return "…"

def compute_emd_pvalue_binary(all_labels01: np.ndarray,
                              subgroup_labels01: np.ndarray,
                              n_iter: int = 2000,
                              seed: int = 42):
    """
    EMD between subgroup label distribution and overall label distribution (binary labels).
    Permutation bootstrap p-value using sampling without replacement.
    """
    rng = np.random.default_rng(seed)
    all_labels01 = np.asarray(all_labels01, dtype=int)
    subgroup_labels01 = np.asarray(subgroup_labels01, dtype=int)

    observed = wasserstein_distance(subgroup_labels01, all_labels01)

    n = len(all_labels01)
    k = len(subgroup_labels01)
    boot = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.choice(n, size=k, replace=False)
        boot[i] = wasserstein_distance(all_labels01[idx], all_labels01)

    p_val = float((boot >= observed).mean())
    return float(observed), p_val

def load_and_prepare(cfg):
    df = pd.read_csv(cfg["path"])
    # Map gender -> {Male:1, Female:0}
    if "gender_map" in cfg and cfg["gender_map"] is not None:
        df[cfg["gender_col"]] = df[cfg["gender_col"]].map(cfg["gender_map"])
    # Coerce label to 0/1
    df[cfg["label_col"]] = (df[cfg["label_col"]].astype(float) > 0).astype(int)
    # Keep only valid rows
    df = df.dropna(subset=[cfg["gender_col"], cfg["label_col"]]).copy()
    df = df[df[cfg["gender_col"]].isin([0, 1])]
    df = df[df[cfg["label_col"]].isin([0, 1])]
    df = df.dropna(axis=0, how="any")
    return df

def remove_instances(df, gender_col, label_col, gender_val, label_val, n_remove, seed=42):
    """
    Remove n_remove rows uniformly at random where (gender==gender_val & label==label_val).
    If fewer rows exist, remove all available.
    """
    mask = (df[gender_col] == gender_val) & (df[label_col] == label_val)
    idx = df[mask].index
    if len(idx) == 0:
        return df
    n = min(n_remove, len(idx))
    rng = np.random.default_rng(seed)
    drop_idx = rng.choice(idx, size=n, replace=False)
    return df.drop(index=drop_idx)

def emd_table_for_dataset(tag, cfg, n_iter=2000, seed=42):
    df = load_and_prepare(cfg)

    # Apply specified removal
    rem = cfg.get("removal", None)
    if rem is not None:
        df = remove_instances(
            df,
            gender_col=cfg["gender_col"],
            label_col=cfg["label_col"],
            gender_val=rem["gender"],
            label_val=rem["label"],
            n_remove=rem["n"],
            seed=seed,
        )

    # Build arrays
    y = df[cfg["label_col"]].astype(int).values
    g = df[cfg["gender_col"]].astype(int).values  # Male=1, Female=0

    female_mask = (g == 0)
    male_mask   = (g == 1)

    all_labels    = y
    female_labels = y[female_mask]
    male_labels   = y[male_mask]

    fem_emd, fem_p = compute_emd_pvalue_binary(all_labels, female_labels, n_iter=n_iter, seed=seed)
    mal_emd, mal_p = compute_emd_pvalue_binary(all_labels, male_labels,   n_iter=n_iter, seed=seed)

    out = pd.DataFrame([
        {"Dataset": tag, "Removal of data instances": REMOVAL_DESCRIPTIONS.get(tag, ""), "Group": "Female",
         "EMD value": round(fem_emd, 6), "p-value": fmt_p(fem_p),
         "Exhibit data bias?": "Yes" if (isinstance(fem_p, float) and fem_p <= 0.05) else "No"},
        {"Dataset": tag, "Removal of data instances": "", "Group": "Male",
         "EMD value": round(mal_emd, 6), "p-value": fmt_p(mal_p),
         "Exhibit data bias?": "Yes" if (isinstance(mal_p, float) and mal_p <= 0.05) else "No"},
    ])
    return out

# ==========================
# Main
# ==========================
def main():
    tables = []
    for tag, cfg in DATASETS.items():
        t = emd_table_for_dataset(tag, cfg, n_iter=2000, seed=42)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)

    # Pretty print
    pd.set_option("display.max_colwidth", None)
    print("\n=== EMD Bias Table after Specified Instance Removal ===")
    print(table.to_string(index=False))

    # Save
    table.to_csv("emd_bias_table_after_specified_removal.csv", index=False)
    try:
        table.to_excel("emd_bias_table_after_specified_removal.xlsx", index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()
