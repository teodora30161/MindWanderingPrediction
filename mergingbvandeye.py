import pandas as pd

print("📂 Loading feature files...")
behavior_df = pd.read_csv("sliding_window_behavioral_features_labeled.csv")
eye_df = pd.read_csv("outputs/sliding_window_eye_features_labeled.csv")
probes_df = pd.read_csv("probes_eye_VF_indexed.csv")

# Standardize all column names to lowercase
behavior_df.columns = behavior_df.columns.str.lower()
eye_df.columns = eye_df.columns.str.lower()
probes_df.columns = probes_df.columns.str.lower()

print("Behavioral columns:", behavior_df.columns.tolist())
print("Eye-tracking columns:", eye_df.columns.tolist())
print("Probes columns:", probes_df.columns.tolist())

# Drop unnecessary columns
behavior_df = behavior_df.drop(columns=[col for col in behavior_df.columns if col == 'probe_time'], errors='ignore')
eye_df = eye_df.drop(columns=[col for col in eye_df.columns if col in ['probe_time', 'probe_onset_time']], errors='ignore')

# Rename 'session' column in probes_df to match other dataframes
probes_df = probes_df.rename(columns={'session': 'session_id'})

# Define keys used for merging
merge_keys = ['subj_orgid', 'session_id', 'block_num', 'probe_number']

# Ensure consistent formatting of merge keys
for df_name, df in zip(['Behavior', 'Eye', 'Probes'], [behavior_df, eye_df, probes_df]):
    for key in merge_keys:
        if key in df.columns:
            if df[key].dtype == object:
                df[key] = df[key].astype(str).str.strip()
            else:
                df[key] = pd.to_numeric(df[key], errors='coerce')
        else:
            print(f"⚠️ WARNING: '{key}' not found in {df_name} dataframe")

# === 2. Merge behavioral and eye-tracking features ===
print(" Merging behavioral and eye-tracking features...")
merged_df = pd.merge(
    behavior_df,
    eye_df,
    on=merge_keys,
    suffixes=('_behavior', '_eye'),
    how='inner'
)

# === 3. Merge with ON/OFF labels from probes ===
print(" Adding ON/OFF task labels...")
if 'on_off' in probes_df.columns:
    merged_df = pd.merge(
        merged_df,
        probes_df[merge_keys + ['on_off']],
        on=merge_keys,
        how='left'
    )
else:
    print(" 'on_off' column not found in probes_df. Skipping this step.")

# === 4. Save the final merged dataset ===
output_path = "combined_behavior_eye_features_labeled.csv"
merged_df.to_csv(output_path, index=False)

print(f" Final merged dataset saved: {output_path}")
print(f" Final shape: {merged_df.shape}")
