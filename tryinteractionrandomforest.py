import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, roc_curve, auc, 
    roc_auc_score, precision_recall_curve, f1_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns

# Load dataset
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes4.csv"
df = pd.read_csv(file_path)

# Define features and target
features = ['BV', 'AE', 'pupil_size_median']
target = 'ON_OFF'
group_column = 'subj'

# Create interaction features
df['BV_AE'] = df['BV'] * df['AE']
df['BV_ratio_pupil'] = df['BV'] / (df['pupil_size_median'] + 1e-5)  # Avoid division by zero
df['AE_pupil'] = df['AE'] * df['pupil_size_median']

# New feature list
features.extend(['BV_AE', 'BV_ratio_pupil', 'AE_pupil'])

# Data preprocessing
df_clean = df.dropna(subset=features + [target, group_column]).copy()
df_clean['ON_OFF'] = df_clean['ON_OFF'].map({'ON': 0, 'OFF': 1})

X = df_clean[features].values
y = df_clean[target].values
groups = df_clean[group_column].values

# Group K-Fold cross-validation
gkf = GroupKFold(n_splits=5)

# Optuna Objective Function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        class_weight="balanced", random_state=42
    )

    balanced_accuracies = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        calibrated_model = CalibratedClassifierCV(model, cv=5)
        calibrated_model.fit(X_train, y_train)

        y_pred = calibrated_model.predict(X_test)
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))

    return np.mean(balanced_accuracies)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42)
best_model.fit(X, y)  # Fit on full dataset

calibrated_best_model = CalibratedClassifierCV(best_model, cv=5)

# Store probabilities & true labels
all_probs = np.zeros(len(y))
all_true = np.zeros(len(y))

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    calibrated_best_model.fit(X_train, y_train)
    y_probs = calibrated_best_model.predict_proba(X_test)[:, 1]

    all_probs[test_idx] = y_probs
    all_true[test_idx] = y_test

# Compute F1-score & Precision-Recall AUC for Threshold Selection
precisions, recalls, thresholds = precision_recall_curve(all_true, all_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # Avoid division by zero
best_f1_threshold = thresholds[np.argmax(f1_scores)]
pr_auc = average_precision_score(all_true, all_probs)

# Apply threshold
y_final_pred = (all_probs >= best_f1_threshold).astype(int)

# Print Results
print(f"\n🚀 Best Threshold Found (F1-score): {best_f1_threshold:.2f}")
print(f"🎯 Precision-Recall AUC: {pr_auc:.4f}")
print("Final Classification Report:")
print(classification_report(all_true, y_final_pred))

# Feature Importance
importances = best_model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Visualization: Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(all_true, all_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC Curve", color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label="Precision-Recall Curve", color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Threshold vs F1-Score
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores[:-1], marker="o", linestyle="-", color="red")
plt.axvline(best_f1_threshold, color="black", linestyle="--", label=f"Best Threshold ({best_f1_threshold:.2f})")
plt.xlabel("Threshold Value")
plt.ylabel("F1-Score")
plt.title("Threshold Optimization (F1-Score)")
plt.legend()
plt.show()

# AUC Score
auc_score = roc_auc_score(all_true, all_probs)
print("\n🚀 Overall AUC Score:", auc_score)
#auc for all the folds
#plot the values for every participants for the models 
#SVM 