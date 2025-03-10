import optuna
import pandas as pd

# Define the database storage
optuna_db = "sqlite:///optuna_study.db"

# Load previous study (if exists)
study = optuna.create_study(study_name="mindwandering_rf", 
                            storage=optuna_db, 
                            load_if_exists=True, 
                            direction="maximize")

# Export results to CSV
df_results = study.trials_dataframe()
df_results.to_csv("optuna_results.csv", index=False)
print("✅ Optuna results saved to optuna_results.csv")

# Print best hyperparameters
print("\n🔹 Best Hyperparameters Found:", study.best_params)
print("🎯 Best Balanced Accuracy:", study.best_value)

# Print all trials summary
print("\n📊 Previous Optuna Trials Summary:")
print(df_results.head())  # Show first few rows of trials
