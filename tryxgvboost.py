import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_curve, auc, precision_recall_curve

# Loading Luca's data
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes4.csv"
df = pd.read_csv(file_path)

# Features and target
features = ['BV', 'AE', 'pupil_size_median']
target = 'ON_OFF'
group_column = 'subj'
df['BV_AE'] = df['BV'] * df['AE']
df['BV_ratio_pupil'] = df['BV'] / (df['pupil_size_median'] + 1e-6)
df['AE_pupil'] = df['AE'] / (df['pupil_size_median'] + 1e-6)
features += ['BV_AE', 'BV_ratio_pupil', 'AE_pupil']

df_clean = df.dropna(subset=features + [target, group_column]).copy()
df_clean[target] = df_clean[target].map({'ON': 0, 'OFF': 1})  # Convert to 0/1

X = df_clean[features].values
y = df_clean[target].values
groups = df_clean[group_column].values

# Split dataset (No SMOTE applied here)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply SMOTE to training data
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_orig, y_train_orig)

# Define Optuna objective function
def objective(trial, X_train, y_train):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1, 10),
    }
    
    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test_orig)
    return balanced_accuracy_score(y_test_orig, y_pred)

# Optimize with and without SMOTE
study_orig = optuna.create_study(direction='maximize')
study_orig.optimize(lambda trial: objective(trial, X_train_orig, y_train_orig), n_trials=50)

study_smote = optuna.create_study(direction='maximize')
study_smote.optimize(lambda trial: objective(trial, X_train_smote, y_train_smote), n_trials=50)

# Best hyperparameters
best_params_orig = study_orig.best_params
best_params_smote = study_smote.best_params

# Train final models
best_model_orig = XGBClassifier(**best_params_orig, random_state=42, use_label_encoder=False, eval_metric='logloss')
best_model_orig.fit(X_train_orig, y_train_orig)

best_model_smote = XGBClassifier(**best_params_smote, random_state=42, use_label_encoder=False, eval_metric='logloss')
best_model_smote.fit(X_train_smote, y_train_smote)

# Predictions
y_probs_orig = best_model_orig.predict_proba(X_test_orig)[:, 1]
y_probs_smote = best_model_smote.predict_proba(X_test_orig)[:, 1]

# Compute AUC scores
precision_orig, recall_orig, _ = precision_recall_curve(y_test_orig, y_probs_orig)
pr_auc_orig = auc(recall_orig, precision_orig)

precision_smote, recall_smote, _ = precision_recall_curve(y_test_orig, y_probs_smote)
pr_auc_smote = auc(recall_smote, precision_smote)

fpr_orig, tpr_orig, _ = roc_curve(y_test_orig, y_probs_orig)
roc_auc_orig = auc(fpr_orig, tpr_orig)

fpr_smote, tpr_smote, _ = roc_curve(y_test_orig, y_probs_smote)
roc_auc_smote = auc(fpr_smote, tpr_smote)

# ROC Curve Comparison
plt.figure(figsize=(8, 6))
plt.plot(fpr_orig, tpr_orig, label=f"Without SMOTE (AUC = {roc_auc_orig:.2f})", color='blue')
plt.plot(fpr_smote, tpr_smote, label=f"With SMOTE (AUC = {roc_auc_smote:.2f})", color='red')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: With vs Without SMOTE")
plt.legend()
plt.show()

# AUC Per Participant
gkf = GroupKFold(n_splits=5)
participant_aucs_orig = {}
y_test_pooled = []
y_probs_pooled = []

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    best_model_orig.fit(X_train_fold, y_train_fold)
    y_probs_fold_orig = best_model_orig.predict_proba(X_test_fold)[:, 1]
    auc_orig = auc(*roc_curve(y_test_fold, y_probs_fold_orig)[:2])

    unique_participants = np.unique(groups[test_idx])
    for participant in unique_participants:
        participant_aucs_orig[participant] = auc_orig

    # Collect test data for pooled AUC computation
    y_test_pooled.extend(y_test_fold)
    y_probs_pooled.extend(y_probs_fold_orig)

# Convert to numpy arrays
y_test_pooled = np.array(y_test_pooled)
y_probs_pooled = np.array(y_probs_pooled)

# Compute overall AUC using pooled test set
fpr_pooled, tpr_pooled, _ = roc_curve(y_test_pooled, y_probs_pooled)
roc_auc_pooled = auc(fpr_pooled, tpr_pooled)

# Compute average participant AUC
avg_auc_orig = np.mean(list(participant_aucs_orig.values()))

print("AUC per Participant (Without SMOTE):")
for participant, auc_score in participant_aucs_orig.items():
    print(f"{participant}: {auc_score:.4f}")

print("\nRecalculated Overall AUC using Pooled Test Set: {:.4f}".format(roc_auc_pooled))
print("Original Overall AUC from ROC Curve: {:.4f}".format(roc_auc_orig))
print("Average Participant AUC: {:.4f}".format(avg_auc_orig))

# Plot AUC per participant
plt.figure(figsize=(8, 6))
plt.bar(participant_aucs_orig.keys(), participant_aucs_orig.values(), color='blue', label="Without SMOTE")
plt.xlabel("Participant")
plt.ylabel("AUC Score")
plt.title("AUC per Participant (XGBoost)")
plt.xticks(rotation=45)
plt.legend()
plt.show()
