import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the provided CSV file
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes4.csv"
df = pd.read_csv(file_path)

# Select relevant columns
features = ['BV', 'AE', 'block_num', 'pupil_size_mean', 'pupil_size_median']  # Added more features
target = 'ON_OFF'
group_column = 'subj'  # For LOSO

# Drop rows with missing values in selected columns
df_clean = df.dropna(subset=features + [target, group_column]).copy()

# Convert ON_OFF to binary (ON = 0, OFF = 1)
df_clean['ON_OFF'] = df_clean['ON_OFF'].map({'ON': 0, 'OFF': 1})

# Extract X (features), y (target), and groups (subjects for LOSO)
X = df_clean[features].values
y = df_clean[target].values
groups = df_clean[group_column].values

# Initialize Leave-One-Subject-Out cross-validation
logo = LeaveOneGroupOut()
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=10,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

# Store results
accuracies = []
all_preds = []
all_true = []

# Perform LOSO cross-validation
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Store metrics
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    all_preds.extend(y_pred)
    all_true.extend(y_test)

# Compute overall accuracy and classification report
overall_accuracy = np.mean(accuracies)
class_report = classification_report(all_true, all_preds)

# Display results
print("Overall Accuracy:", overall_accuracy)
print("Classification Report:\n", class_report)
