
# Fairness-Aware ML Evaluation Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
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
from scipy.stats import ttest_ind, wasserstein_distance
import warnings

warnings.filterwarnings("ignore")
seed = 42

# Load and clean dataset
dataset = pd.read_csv('framingham.csv').dropna()

# Bias mitigation: remove 100 male-positive instances
male_chd = dataset[(dataset['male'] == 1) & (dataset['TenYearCHD'] == 1)]
if len(male_chd) >= 100:
    dataset = dataset.drop(male_chd.sample(n=100, random_state=seed).index)

# Prepare data
X = dataset.drop('TenYearCHD', axis=1)
y = dataset['TenYearCHD'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gender masks
male_mask = dataset['male'] == 1
female_mask = ~male_mask
X_male, X_female = X_scaled[male_mask], X_scaled[female_mask]
y_male, y_female = y[male_mask], y[female_mask]

# Define models
models = {
    'SVM': SVC(probability=True, random_state=seed),
    'LR': LogisticRegression(random_state=seed),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(random_state=seed),
    'DT': DecisionTreeClassifier(random_state=seed),
    'ANN': MLPClassifier(random_state=seed, max_iter=500)
}

# Metric calculation
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        'TPR': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'TNR': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'FPR': FP / (FP + TN) if (FP + TN) > 0 else 0,
        'FNR': FN / (TP + FN) if (TP + FN) > 0 else 0,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

# Run k-fold experiments
def run_kfold(X, y, label):
    kf = KFold(n_splits=20, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        fold_result = {'Fold': fold + 1, 'Group': label}
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores = compute_metrics(y_test, y_pred)

            for metric, value in scores.items():
                fold_result[f"{name}_{metric}"] = value

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            fold_result.update({
                f"{name}_Accuracy": acc,
                f"{name}_Precision": prec,
                f"{name}_Recall": rec,
                f"{name}_F1": f1
            })

        results.append(pd.DataFrame([fold_result]))
    return pd.concat(results, ignore_index=True)

# Execute and save results
print("Running Female Group")
female_df = run_kfold(X_female, y_female, 'Female')
print("Running Male Group")
male_df = run_kfold(X_male, y_male, 'Male')
results_df = pd.concat([female_df, male_df], ignore_index=True)
results_df.to_excel('results.xlsx', index=False)
print("Saved: results.xlsx")

# EMD analysis
def compute_emd_pvalue(dataset, group_col='male', subgroup_value=0, label_col='TenYearCHD', n_iter=10000):
    all_labels = dataset[label_col].values
    subgroup = dataset[dataset[group_col] == subgroup_value][label_col].values
    observed_emd = wasserstein_distance(subgroup, all_labels)

    boot_emd = [
        wasserstein_distance(np.random.choice(all_labels, len(subgroup), replace=False), all_labels)
        for _ in range(n_iter)
    ]
    p_val = np.mean(np.array(boot_emd) >= observed_emd)
    return observed_emd, p_val

# Show Table 3-style bias check
female_emd, female_p = compute_emd_pvalue(dataset, subgroup_value=0)
male_emd, male_p = compute_emd_pvalue(dataset, subgroup_value=1)

print(f"Gender Bias EMD Results:")
print(f"{'Group':<8} {'EMD':<8} {'p-value':<8} {'Bias?'}")
print(f"{'Female':<8} {female_emd:.4f}  {female_p:.4f}  {'Yes' if female_p <= 0.05 else 'No'}")
print(f"{'Male':<8} {male_emd:.4f}  {male_p:.4f}  {'Yes' if male_p <= 0.05 else 'No'}")
