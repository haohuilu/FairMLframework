#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.model_selection import KFold, GridSearchCV


# Ignore all warning messages
warnings.filterwarnings('ignore')

# Set seed for reproducibility
seed = 42

# Load the dataset
dataset = pd.read_csv('framingham.csv')


# In[ ]:


dataset


# In[ ]:


dataset = dataset.dropna()


# In[ ]:


print(dataset['TenYearCHD'].value_counts())


# In[ ]:


# Filter male entries with outcome = 1
male_chd_entries = dataset[(dataset['male'] == 1) & (dataset['TenYearCHD'] == 1)]

# Drop 100 male entries with TenYearCHD = 1 if there are at least 100 such males
if len(male_chd_entries) >= 100:
    drop_indices = male_chd_entries.sample(n=100, random_state=42).index
    dataset = dataset.drop(index=drop_indices)


# In[ ]:


dataset


# In[ ]:


# Define your features and label, but keep 'gender' for creating the mask
X = dataset.drop('TenYearCHD', axis=1)
y = dataset['TenYearCHD'].values

# Create masks for gender before scaling
gender_0_mask = dataset['male'] == 0
gender_1_mask = dataset['male'] == 1

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply masks after scaling
X_scaled_gender_0 = X_scaled[gender_0_mask]
X_scaled_gender_1 = X_scaled[gender_1_mask]
y_gender_0 = y[gender_0_mask]
y_gender_1 = y[gender_1_mask]

# Define the models
models = {
    'SVM': SVC(random_state=seed),
    'LR': LogisticRegression(random_state=seed),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(random_state=seed),
    'DT': DecisionTreeClassifier(random_state=seed),
    'ANN': MLPClassifier(random_state=seed)
}

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity, recall, or true positive rate
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity or true negative rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False positive rate
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # False negative rate

    return TPR, TNR, FPR, FNR, TP, TN, FP, FN

# Initialize a list to store temporary DataFrame objects
results_list = []

# Perform k-fold cross-validation and calculate sensitivity and specificity
kf = KFold(n_splits=20, shuffle=True, random_state=seed)

# Define function for running experiments and storing results
def run_experiment(X_data, y_data, group_label, results_list):
    for fold, (train_index, test_index) in enumerate(kf.split(X_data)):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        fold_results = {'Fold': fold + 1, 'Group': group_label}
        print(f"Processing fold {fold + 1} for group {group_label}")

        for name, model in models.items():
            print(f"   Training and evaluating model: {name}")

            # Fit the model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Calculate metrics
            TPR, TNR, FPR, FNR, TP, TN, FP, FN = calculate_metrics(y_test, y_pred)

            # Store results in the fold_results dictionary
            fold_results.update({
                f'{name}_TPR': TPR, f'{name}_TNR': TNR,
                f'{name}_FPR': FPR, f'{name}_FNR': FNR,
                f'{name}_TP': TP, f'{name}_TN': TN,
                f'{name}_FP': FP, f'{name}_FN': FN
            })

        # Append the dictionary to the results_list as a DataFrame
        results_list.append(pd.DataFrame([fold_results]))

# Running experiments for each gender
print("Starting experiments for Gender = Female (0)")
run_experiment(X_scaled_gender_0, y_gender_0, 'Female', results_list)

print("Starting experiments for Gender = Male (1)")
run_experiment(X_scaled_gender_1, y_gender_1, 'Male', results_list)

# Concatenate all DataFrames in the results_list into one DataFrame
final_results_df = pd.concat(results_list, ignore_index=True)
print(final_results_df)


# In[ ]:


results_df = pd.concat(results_list, ignore_index=True)


# In[ ]:


results_path = 'results.xlsx'
results_df.to_excel(results_path, index=False)


# In[ ]:


from scipy.stats import ttest_ind

def perform_t_tests(df, algorithm):
    # Define the column names based on the algorithm
    tpr_col = f"{algorithm}_TPR"
    fpr_col = f"{algorithm}_FPR"
    fn_col = f"{algorithm}_FN"
    fp_col = f"{algorithm}_FP"

    # Define the groups
    protected_group = df['Group'] == 'Female'
    unprotected_group = ~protected_group

    # Extract the metrics
    protected_tpr = df.loc[protected_group, tpr_col].values
    unprotected_tpr = df.loc[unprotected_group, tpr_col].values

    protected_fpr = df.loc[protected_group, fpr_col].values
    unprotected_fpr = df.loc[unprotected_group, fpr_col].values

    protected_ratio_fn_fp = (df.loc[protected_group, fn_col] / df.loc[protected_group, fp_col]).values
    unprotected_ratio_fn_fp = (df.loc[unprotected_group, fn_col] / df.loc[unprotected_group, fp_col]).values

    # Perform t-tests

    # Definition 1: Equalised Odds (TPR and FPR)
    tpr_ttest = ttest_ind(protected_tpr, unprotected_tpr)
    fpr_ttest = ttest_ind(protected_fpr, unprotected_fpr)

    # Definition 2: Equal Opportunity (TPR)
    equal_opportunity_ttest = ttest_ind(protected_tpr, unprotected_tpr)

    # Definition 3: Treatment Equality (Ratio of false negatives to false positives)
    treatment_equality_ttest = ttest_ind(protected_ratio_fn_fp, unprotected_ratio_fn_fp)

    # Definition 4: Aggregate of all conditions
    aggregate_tpr_ttest = ttest_ind(protected_tpr, unprotected_tpr)
    aggregate_fpr_ttest = ttest_ind(protected_fpr, unprotected_fpr)
    aggregate_ratio_ttest = ttest_ind(protected_ratio_fn_fp, unprotected_ratio_fn_fp)

    # Print results
    print(f"{algorithm} - Equalised Odds (TPR):", tpr_ttest)
    print(f"{algorithm} - Equalised Odds (FPR):", fpr_ttest)
    print(f"{algorithm} - Equal Opportunity (TPR):", equal_opportunity_ttest)
    print(f"{algorithm} - Aggregate - TPR:", aggregate_tpr_ttest)
    print(f"{algorithm} - Aggregate - FPR:", aggregate_fpr_ttest)


# In[ ]:


def perform_fairness_metrics(df, algorithm):
    # Define the column names
    tpr_col = f"{algorithm}_TPR"
    fpr_col = f"{algorithm}_FPR"
    fn_col = f"{algorithm}_FN"
    fp_col = f"{algorithm}_FP"
    tp_col = f"{algorithm}_TP"
    tn_col = f"{algorithm}_TN"

    # Define groups
    protected_group = df['Group'] == 'Female'
    unprotected_group = ~protected_group

    # Calculate metrics
    protected_tpr = df.loc[protected_group, tpr_col].mean()
    unprotected_tpr = df.loc[unprotected_group, tpr_col].mean()

    protected_fpr = df.loc[protected_group, fpr_col].mean()
    unprotected_fpr = df.loc[unprotected_group, fpr_col].mean()

    protected_ratio_fn_fp = (df.loc[protected_group, fn_col] / df.loc[protected_group, fp_col]).mean()
    unprotected_ratio_fn_fp = (df.loc[unprotected_group, fn_col] / df.loc[unprotected_group, fp_col]).mean()

    # Demographic Parity = (TP + FP) / Total
    protected_tp = df.loc[protected_group, tp_col]
    protected_fp = df.loc[protected_group, fp_col]
    protected_tn = df.loc[protected_group, tn_col]
    protected_fn = df.loc[protected_group, fn_col]
    protected_positive_rate = (protected_tp + protected_fp) / (protected_tp + protected_fp + protected_tn + protected_fn)
    protected_dp = protected_positive_rate.mean()

    unprotected_tp = df.loc[unprotected_group, tp_col]
    unprotected_fp = df.loc[unprotected_group, fp_col]
    unprotected_tn = df.loc[unprotected_group, tn_col]
    unprotected_fn = df.loc[unprotected_group, fn_col]
    unprotected_positive_rate = (unprotected_tp + unprotected_fp) / (unprotected_tp + unprotected_fp + unprotected_tn + unprotected_fn)
    unprotected_dp = unprotected_positive_rate.mean()

    # Print metrics
    #print(f"{algorithm} - Equalised Odds (TPR): Protected={protected_tpr:.3f}, Unprotected={unprotected_tpr:.3f}")
    #print(f"{algorithm} - Equalised Odds (FPR): Protected={protected_fpr:.3f}, Unprotected={unprotected_fpr:.3f}")
    #print(f"{algorithm} - Equal Opportunity (TPR): Protected={protected_tpr:.3f}, Unprotected={unprotected_tpr:.3f}")
    #print(f"{algorithm} - Treatment Equality (FN/FP Ratio): Protected={protected_ratio_fn_fp:.3f}, Unprotected={unprotected_ratio_fn_fp:.3f}")
    print(f"{algorithm} - Demographic Parity (Positive Rate): Protected={protected_dp:.3f}, Unprotected={unprotected_dp:.3f}")


# In[ ]:


perform_fairness_metrics(df, 'SVM')
perform_fairness_metrics(df, 'KNN')
perform_fairness_metrics(df, 'LR')
perform_fairness_metrics(df, 'DT')
perform_fairness_metrics(df, 'RF')
perform_fairness_metrics(df, 'ANN')


# In[ ]:


df = pd.read_excel('results.xlsx')


# In[ ]:


perform_t_tests(df, 'SVM')
perform_t_tests(df, 'KNN')
perform_t_tests(df, 'LR')
perform_t_tests(df, 'DT')
perform_t_tests(df, 'RF')
perform_t_tests(df, 'ANN')


# In[ ]:





# In[ ]:


from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

# Function to perform t-tests for Treatment Equality
def perform_treatment_equality_ttest(df, algorithm):
    # Define the column names based on the algorithm
    fn_col = f"{algorithm}_FN"
    fp_col = f"{algorithm}_FP"

    # Define the groups
    protected_group = df['Group'] == 'Female'
    unprotected_group = ~protected_group

    # Extract FN and FP values
    protected_fn = df.loc[protected_group, fn_col].values
    protected_fp = df.loc[protected_group, fp_col].values

    unprotected_fn = df.loc[unprotected_group, fn_col].values
    unprotected_fp = df.loc[unprotected_group, fp_col].values

    # Compute FN/FP ratios while handling division by zero
    protected_ratio_fn_fp = np.divide(protected_fn, protected_fp, out=np.zeros_like(protected_fn, dtype=float), where=protected_fp!=0)
    unprotected_ratio_fn_fp = np.divide(unprotected_fn, unprotected_fp, out=np.zeros_like(unprotected_fn, dtype=float), where=unprotected_fp!=0)

    # Perform t-test
    treatment_equality_ttest = ttest_ind(protected_ratio_fn_fp, unprotected_ratio_fn_fp, nan_policy='omit')

    # Print results
    print(f"{algorithm} - Treatment Equality (FN/FP Ratio):", treatment_equality_ttest)
    return treatment_equality_ttest

# Run treatment equality tests for all models
for model_name in models.keys():
    perform_treatment_equality_ttest(final_results_df, model_name)


# In[ ]:


from scipy.stats import wasserstein_distance
import numpy as np

def compute_emd_pvalue(dataset, label_col='TenYearCHD', group_col='male', subgroup_value=0, n_iterations=10000, random_state=42):
    """
    Compute EMD and p-value comparing a subgroup to the full population.

    Parameters:
        dataset: your DataFrame
        label_col: name of the binary label column
        group_col: name of the group (protected attribute) column
        subgroup_value: 0 for Female, 1 for Male (based on your data)
        n_iterations: number of bootstraps
        random_state: seed for reproducibility

    Returns:
        observed_emd: float
        p_value: float
    """
    np.random.seed(random_state)

    # Actual subgroup and total label distributions
    all_labels = dataset[label_col].values
    subgroup_labels = dataset[dataset[group_col] == subgroup_value][label_col].values

    # EMD between subgroup and full distribution
    observed_emd = wasserstein_distance(subgroup_labels, all_labels)

    # Bootstrap: random subsamples of same size
    group_size = len(subgroup_labels)
    emd_samples = []
    for _ in range(n_iterations):
        random_sample = np.random.choice(all_labels, size=group_size, replace=False)
        emd = wasserstein_distance(random_sample, all_labels)
        emd_samples.append(emd)

    p_value = np.mean(np.array(emd_samples) >= observed_emd)
    return observed_emd, p_value


# In[ ]:


dataset


# In[ ]:


# Female group (male = 0)
female_emd, female_p = compute_emd_pvalue(dataset, subgroup_value=0)

# Male group (male = 1)
male_emd, male_p = compute_emd_pvalue(dataset, subgroup_value=1)

# Display Table 3-style output
print("\nTable 3: Statistical assessment of gender bias (via EMD after mitigation)")
print(f"{'Group':<8} {'EMD value':<12} {'p-value':<10} {'Exhibit data bias?'}")
print(f"{'Female':<8} {female_emd:.4f}      {female_p:.4f}    {'Yes' if female_p <= 0.05 else 'No'}")
print(f"{'Male':<8} {male_emd:.4f}      {male_p:.4f}    {'Yes' if male_p <= 0.05 else 'No'}")


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

# Set seed
seed = 42

# Define features and label
X = dataset.drop('TenYearCHD', axis=1)
y = dataset['TenYearCHD'].values

# Gender masks
gender_0_mask = dataset['male'] == 0
gender_1_mask = dataset['male'] == 1

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gender-specific data
X_scaled_gender_0 = X_scaled[gender_0_mask]
X_scaled_gender_1 = X_scaled[gender_1_mask]
y_gender_0 = y[gender_0_mask]
y_gender_1 = y[gender_1_mask]

# Models
models = {
    'SVM': SVC(random_state=seed),
    'LR': LogisticRegression(random_state=seed),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(random_state=seed),
    'DT': DecisionTreeClassifier(random_state=seed),
    'ANN': MLPClassifier(random_state=seed, max_iter=500)
}

# Metric calculator
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return TPR, TNR, FPR, FNR, TP, TN, FP, FN, accuracy, precision, recall, f1

# Result collector
results_list = []
kf = KFold(n_splits=20, shuffle=True, random_state=seed)

# Experiment function
def run_experiment(X_data, y_data, group_label, results_list):
    for fold, (train_index, test_index) in enumerate(kf.split(X_data)):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        fold_results = {'Fold': fold + 1, 'Group': group_label}
        print(f"Processing fold {fold + 1} for group {group_label}")

        for name, model in models.items():
            print(f"   Training and evaluating model: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            TPR, TNR, FPR, FNR, TP, TN, FP, FN, acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)

            fold_results.update({
                f'{name}_TPR': TPR, f'{name}_TNR': TNR,
                f'{name}_FPR': FPR, f'{name}_FNR': FNR,
                f'{name}_TP': TP, f'{name}_TN': TN,
                f'{name}_FP': FP, f'{name}_FN': FN,
                f'{name}_Accuracy': acc,
                f'{name}_Precision': prec,
                f'{name}_Recall': rec,
                f'{name}_F1': f1
            })

        results_list.append(pd.DataFrame([fold_results]))

# Run experiments
print("Starting experiments for Gender = Female (0)")
run_experiment(X_scaled_gender_0, y_gender_0, 'Female', results_list)

print("Starting experiments for Gender = Male (1)")
run_experiment(X_scaled_gender_1, y_gender_1, 'Male', results_list)

# Combine results
final_results_df = pd.concat(results_list, ignore_index=True)

# Print metrics summary
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
for model in models.keys():
    print(f"\nModel: {model}")
    for metric in metrics:
        avg_score = final_results_df[f'{model}_{metric}'].mean()
        print(f"  {metric:<10}: {avg_score:.4f}")


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Set seed
seed = 42

# Define features and label
X = dataset.drop('TenYearCHD', axis=1)
y = dataset['TenYearCHD'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose train-test split ratio: change to 0.3 for 70:30 split
split_ratio = 0.2  # 80:20 split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=split_ratio, random_state=seed, stratify=y
)

print("Class distribution in train set:", np.bincount(y_train))
print("Class distribution in test set :", np.bincount(y_test))

# Define models (NO class weighting or sampling)
models = {
    'SVM': SVC(probability=True, random_state=seed),
    'LR': LogisticRegression(random_state=seed),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(random_state=seed),
    'DT': DecisionTreeClassifier(random_state=seed),
    'ANN': MLPClassifier(random_state=seed, max_iter=500)
}

# Evaluation function
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    else:
        roc = None

    print(f"\n{name} Performance (no imbalance handling):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    if roc is not None:
        print(f"  ROC-AUC  : {roc:.4f}")

# Run models
for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)


# In[ ]:




