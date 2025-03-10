import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import optuna.visualization as vis
import seaborn as sns

file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes4.csv"
df = pd.read_csv(file_path)

features = ['BV', 'AE', 'pupil_size_median']
target = 'ON_OFF'
group_column = 'subj'

df_clean = df.dropna(subset=features + [target, group_column]).copy()
df_clean['ON_OFF'] = df_clean['ON_OFF'].map({'ON': 0, 'OFF': 1})

X = df_clean[features].values
y = df_clean[target].values
groups = df_clean[group_column].values

def check_class_imbalance(y):
    class_counts = np.bincount(y.astype(int))
    print("\n🔹 Class distribution:")
    print(f" - Class 0 (OFF): {class_counts[0]}")
    print(f" - Class 1 (ON): {class_counts[1]}")

    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Class 0 (OFF) is {imbalance_ratio:.2f} times more frequent than Class 1 (ON)\n")
    return imbalance_ratio

imbalance_ratio = check_class_imbalance(y)

gkf = GroupKFold(n_splits=5)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    class_weights = {0: imbalance_ratio, 1: 1} if imbalance_ratio > 2.0 else "balanced"

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weights,
        random_state=42
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

study = optuna.create_study(study_name="mindwandering_rf", storage="sqlite:///optuna_study.db", load_if_exists=True, direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_score = study.best_value

print("Best Hyperparameters Found:", best_params)
print("Best Balanced Accuracy:", best_score)

best_model = RandomForestClassifier(**best_params, class_weight={0: 2, 1: 3}, random_state=42)
best_model.fit(X, y)
calibrated_best_model = CalibratedClassifierCV(best_model, cv=5)

all_probs = np.zeros(len(y))
all_true = np.zeros(len(y))
participant_aucs = {}

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    calibrated_best_model.fit(X_train, y_train)
    y_probs = calibrated_best_model.predict_proba(X_test)[:, 1]

    all_probs[test_idx] = y_probs
    all_true[test_idx] = y_test

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    participant_aucs[groups[test_idx][0]] = auc(fpr, tpr)

fpr, tpr, thresholds = roc_curve(all_true, all_probs)
youden_index = np.argmax(tpr - fpr)
best_threshold = thresholds[youden_index]
print(f"\n🚀 Best Threshold Found (Youden’s J): {best_threshold:.2f}")

y_final_pred = (all_probs >= best_threshold).astype(int)
print("🎯 Final Classification Report with Optimized Threshold:")
print(classification_report(all_true, y_final_pred))

importances = best_model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Averaged ROC Curve", color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Averaged ROC Curve Across Folds")
plt.legend()
plt.show()

thresholds = np.linspace(0.1, 0.9, 50)
bal_acc_scores = [balanced_accuracy_score(all_true, (all_probs >= t).astype(int)) for t in thresholds]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, bal_acc_scores, marker="o", linestyle="-", color="green")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best Threshold ({best_threshold:.2f})")
plt.xlabel("Threshold Value")
plt.ylabel("Balanced Accuracy")
plt.title("Threshold Optimization")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(participant_aucs.keys(), participant_aucs.values(), color='blue')
plt.xlabel("Participant")
plt.ylabel("AUC Score")
plt.title("AUC per Participant")
plt.xticks(rotation=45)
plt.show()

fig_importance = vis.plot_param_importances(study)
fig_importance.show()
auc_score = roc_auc_score(all_true, all_probs)
print(" Overall AUC Score:", auc_score)
