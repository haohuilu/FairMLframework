# Fairness-Aware ML Evaluation Pipeline (Full Script with p-values)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import wasserstein_distance
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)

# === Load and clean dataset ===
dataset = pd.read_csv('framingham.csv').dropna()

# --- Simple bias mitigation example (as given): remove 100 male-positive instances ---
male_chd = dataset[(dataset['male'] == 1) & (dataset['TenYearCHD'] == 1)]
if len(male_chd) >= 100:
    dataset = dataset.drop(male_chd.sample(n=100, random_state=seed).index)

# === Prepare data ===
y = dataset['TenYearCHD'].values
X = dataset.drop('TenYearCHD', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gender masks
male_mask = dataset['male'] == 1
female_mask = ~male_mask
X_male, X_female = X_scaled[male_mask], X_scaled[female_mask]
y_male, y_female = y[male_mask], y[female_mask]

# === Define models ===
models = {
    'SVM': SVC(probability=True, random_state=seed),
    'LR': LogisticRegression(random_state=seed, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(random_state=seed),
    'DT': DecisionTreeClassifier(random_state=seed),
    'ANN': MLPClassifier(random_state=seed, max_iter=500)
}

# === Metric calculation helpers ===
def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN = FP = FN = TP = 0
    res = {
        'TPR': TP / (TP + FN) if (TP + FN) > 0 else 0.0,
        'TNR': TN / (TN + FP) if (TN + FP) > 0 else 0.0,
        'FPR': FP / (FP + TN) if (FP + TN) > 0 else 0.0,
        'FNR': FN / (TP + FN) if (TP + FN) > 0 else 0.0,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            res['ROC_AUC'] = roc_auc_score(y_true, y_prob)
        except Exception:
            res['ROC_AUC'] = np.nan
    else:
        res['ROC_AUC'] = np.nan
    return res

# === Group-wise k-fold experiments (original idea, per gender) ===
def run_kfold(X_arr, y_arr, label, n_splits=20, random_state=seed):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr), start=1):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        fold_result = {'Fold': fold, 'Group': label}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                z = model.decision_function(X_test)
                y_prob = 1/(1+np.exp(-z))

            scores = compute_metrics(y_test, y_pred, y_prob=y_prob)
            for k, v in scores.items():
                fold_result[f"{name}_{k}"] = v
        rows.append(fold_result)
    return pd.DataFrame(rows)

print("Running Female Group (20-fold)")
female_df = run_kfold(X_female, y_female, 'Female', n_splits=20, random_state=seed)
print("Running Male Group (20-fold)")
male_df   = run_kfold(X_male, y_male, 'Male', n_splits=20, random_state=seed)
group_kfold_df = pd.concat([female_df, male_df], ignore_index=True)

# === EMD analysis (Table 3-style bias check on labels) ===
def compute_emd_pvalue(df, group_col='male', subgroup_value=0, label_col='TenYearCHD', n_iter=10000, rng=None):
    rng = np.random.default_rng(seed if rng is None else rng)
    all_labels = df[label_col].values
    subgroup = df[df[group_col] == subgroup_value][label_col].values
    observed_emd = wasserstein_distance(subgroup, all_labels)
    boot_emd = []
    for _ in range(n_iter):
        boot_sub = rng.choice(all_labels, size=len(subgroup), replace=False)
        boot_emd.append(wasserstein_distance(boot_sub, all_labels))
    boot_emd = np.array(boot_emd)
    p_val = np.mean(boot_emd >= observed_emd)
    return observed_emd, p_val

female_emd, female_p = compute_emd_pvalue(dataset, subgroup_value=0)
male_emd, male_p     = compute_emd_pvalue(dataset, subgroup_value=1)

emd_summary = pd.DataFrame({
    'Group': ['Female', 'Male'],
    'EMD':   [female_emd, male_emd],
    'p_value': [female_p, male_p],
    'Bias? (p<=0.05)': ['Yes' if female_p <= 0.05 else 'No',
                        'Yes' if male_p <= 0.05 else 'No']
})

print("\nGender Bias EMD Results:")
print(emd_summary)

# === Fairness metrics across gender (train on ALL data; between-group gaps) ===
def safe_rate(numer, denom):
    return numer / denom if denom > 0 else 0.0

def rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN = FP = FN = TP = 0
    TPR = safe_rate(TP, TP+FN)
    FPR = safe_rate(FP, FP+TN)
    FNR = safe_rate(FN, TP+FN)
    TNR = safe_rate(TN, TN+FP)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, TPR=TPR, FPR=FPR, FNR=FNR, TNR=TNR)

def fairness_from_groups(y_true, y_pred, y_prob, group_mask):
    """Compute group-wise and disparity metrics (Female=0, Male=1)."""
    g0 = (group_mask == 0)  # Female
    g1 = (group_mask == 1)  # Male

    r0 = rates(y_true[g0], y_pred[g0])
    r1 = rates(y_true[g1], y_pred[g1])

    # Positive prediction rates
    ppr0 = np.mean(y_pred[g0] == 1) if np.any(g0) else 0.0
    ppr1 = np.mean(y_pred[g1] == 1) if np.any(g1) else 0.0

    # Discrimination score (difference in mean positive probability)
    if y_prob is not None:
        dscore = float(np.nan_to_num(np.mean(y_prob[g1]) - np.mean(y_prob[g0])))
    else:
        dscore = float(ppr1 - ppr0)  # fallback

    # Core fairness metrics
    demographic_parity = abs(ppr1 - ppr0)
    disparate_impact   = (ppr0 / ppr1) if ppr1 > 0 else np.nan  # avoid inf -> NaN
    eo_tpr_gap         = abs(r1['TPR'] - r0['TPR'])
    eo_fpr_gap         = abs(r1['FPR'] - r0['FPR'])
    equal_opportunity  = eo_tpr_gap
    fpr_parity         = eo_fpr_gap
    ratio0             = (r0['FN']/r0['FP']) if r0['FP'] > 0 else np.nan
    ratio1             = (r1['FN']/r1['FP']) if r1['FP'] > 0 else np.nan
    treatment_eq       = abs(ratio1 - ratio0) if np.isfinite(ratio0) and np.isfinite(ratio1) else np.nan
    ber0               = 0.5 * (r0['FNR'] + r0['FPR'])
    ber1               = 0.5 * (r1['FNR'] + r1['FPR'])
    ber_mean           = 0.5 * (ber0 + ber1)

    return {
        'Demographic parity (|ΔP(Ŷ=1)|)': demographic_parity,
        'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M))': disparate_impact,
        'Equalised odds |ΔTPR|': eo_tpr_gap,
        'Equalised odds |ΔFPR|': eo_fpr_gap,
        'Equal opportunity |ΔTPR|': equal_opportunity,
        'FPR parity |ΔFPR|': fpr_parity,
        'Treatment equality |Δ(FN/FP)|': treatment_eq,
        'Discrimination score Δ mean ŷ_prob (M−F)': dscore,
        'Balanced Error Rate (mean)': ber_mean,
        'BER_Female': ber0,
        'BER_Male': ber1,
        'PPR_Female': ppr0,
        'PPR_Male': ppr1,
        'TPR_Female': r0['TPR'],
        'TPR_Male': r1['TPR'],
        'FPR_Female': r0['FPR'],
        'FPR_Male': r1['FPR'],
    }

def evaluate_fairness_models(X_all, y_all, group_mask, models_dict, n_splits=5, random_state=seed):
    """Stratified K-fold; compute fairness metrics per fold and average."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, clf in models_dict.items():
        fold_metrics = []
        for train_idx, test_idx in skf.split(X_all, y_all):
            Xtr, Xte = X_all[train_idx], X_all[test_idx]
            ytr, yte = y_all[train_idx], y_all[test_idx]
            gte = group_mask[test_idx] if isinstance(group_mask, np.ndarray) else group_mask.iloc[test_idx].values

            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)

            prob = None
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(Xte)[:, 1]
            elif hasattr(clf, "decision_function"):
                z = clf.decision_function(Xte)
                prob = 1/(1+np.exp(-z))

            fm = fairness_from_groups(yte, yhat, prob, gte)
            fold_metrics.append(fm)

        avg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        avg['Model'] = name
        rows.append(avg)

    df = pd.DataFrame(rows).set_index('Model').sort_index()
    return df

group_mask_int = dataset['male'].astype(int).values  # 1=Male, 0=Female
fair_df = evaluate_fairness_models(X_scaled, y, group_mask_int, models, n_splits=5, random_state=seed)

# Reorder columns to focus on requested metrics
fairness_summary = fair_df[[
    'Demographic parity (|ΔP(Ŷ=1)|)',
    'Equalised odds |ΔTPR|',
    'Equalised odds |ΔFPR|',
    'Equal opportunity |ΔTPR|',
    'FPR parity |ΔFPR|',
    'Treatment equality |Δ(FN/FP)|',
    'Discrimination score Δ mean ŷ_prob (M−F)',
    'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M))',
    'Balanced Error Rate (mean)',
    'BER_Female', 'BER_Male',
    'PPR_Female', 'PPR_Male',
    'TPR_Female', 'TPR_Male',
    'FPR_Female', 'FPR_Male'
]].copy()

print("\n=== Fairness Summary (avg over 5 folds) ===")
print(fairness_summary.round(3))

# === Permutation tests for p-values on three key metrics ===
def permutation_pvalue(X, y, group_mask, model, metric_key, n_iter=500, random_state=42):
    """
    Compute p-value for a fairness metric via permutation of group labels.
    metric_key is one of:
      'Discrimination score Δ mean ŷ_prob (M−F)',
      'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M))',
      'Balanced Error Rate (mean)'
    """
    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # observed
    obs_vals = []
    for tr, te in skf.split(X, y):
        model.fit(X[tr], y[tr])
        yhat = model.predict(X[te])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X[te])[:,1]
        elif hasattr(model, "decision_function"):
            z = model.decision_function(X[te])
            prob = 1/(1+np.exp(-z))
        fm = fairness_from_groups(y[te], yhat, prob, group_mask[te])
        obs_vals.append(fm[metric_key])
    observed = float(np.nanmean(obs_vals))

    # null distribution by shuffling group labels
    null_vals = []
    for _ in range(n_iter):
        shuffled = rng.permutation(group_mask)
        fold_vals = []
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            yhat = model.predict(X[te])
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X[te])[:,1]
            elif hasattr(model, "decision_function"):
                z = model.decision_function(X[te])
                prob = 1/(1+np.exp(-z))
            fm = fairness_from_groups(y[te], yhat, prob, shuffled[te])
            fold_vals.append(fm[metric_key])
        null_vals.append(np.nanmean(fold_vals))

    null_vals = np.array(null_vals, dtype=float)

    # For gap metrics (absolute differences), we use a right-tail p-value
    # For DI (a ratio near 1 is "fair"), use two-sided around 1.0
    if metric_key == 'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M))':
        # distance from 1
        obs_d = abs(observed - 1.0)
        null_d = np.abs(null_vals - 1.0)
        p_val = np.mean(null_d >= obs_d)
    else:
        p_val = np.mean(null_vals >= observed)

    return observed, float(p_val)

# Build p-value table for three key metrics
rows = []
metric_keys = [
    'Discrimination score Δ mean ŷ_prob (M−F)',
    'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M))',
    'Balanced Error Rate (mean)'
]

for name, clf in models.items():
    entry = {'Model': name}
    for mk in metric_keys:
        obs, p = permutation_pvalue(X_scaled, y, group_mask_int, clf, mk, n_iter=500, random_state=seed)
        entry[f'{mk}'] = obs
        entry[f'{mk} p-value'] = p
    rows.append(entry)

pval_table = pd.DataFrame(rows).set_index('Model').sort_index()

print("\n=== Key Fairness Metrics with p-values (avg over CV, permutation test) ===")
print(pval_table.round(4))

# === Print compact key metrics to console ===
print("\n=== Key Fairness Metrics (Discrimination, DI, BER) ===")
for model in pval_table.index:
    dscore = pval_table.loc[model, 'Discrimination score Δ mean ŷ_prob (M−F)']
    disp   = pval_table.loc[model, 'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M))']
    ber    = pval_table.loc[model, 'Balanced Error Rate (mean)']
    pdv1   = pval_table.loc[model, 'Discrimination score Δ mean ŷ_prob (M−F) p-value']
    pdv2   = pval_table.loc[model, 'Disparate Impact (P(Ŷ=1|F)/P(Ŷ=1|M)) p-value']
    pdv3   = pval_table.loc[model, 'Balanced Error Rate (mean) p-value']
    print(f"\nModel: {model}")
    print(f"  Discrimination Score : {dscore:.4f}  (p={pdv1:.4f})")
    print(f"  Disparate Impact     : {disp:.4f}  (p={pdv2:.4f})")
    print(f"  Balanced Error Rate  : {ber:.4f}  (p={pdv3:.4f})")

# === Save all outputs into one Excel file ===
out_file = 'results.xlsx'
with pd.ExcelWriter(out_file, mode='w') as xlw:
    group_kfold_df.to_excel(xlw, sheet_name='group_kfold', index=False)
    fairness_summary.round(6).to_excel(xlw, sheet_name='fairness_summary')
    emd_summary.round(6).to_excel(xlw, sheet_name='emd_summary', index=False)
    pval_table.round(6).to_excel(xlw, sheet_name='key_metrics_pvalues')

print(f"\nSaved all results to {out_file} (sheets: group_kfold, fairness_summary, emd_summary, key_metrics_pvalues)")
