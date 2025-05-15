import pandas as pd

# === 1. Load files ===
behavior_df = pd.read_csv("/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/22participantsfulldata/probes_VF.csv")
eye_df = pd.read_csv("/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/22participantsfulldata/probes_eye_VF.csv")

# === 2. Standardize column names ===
behavior_df.columns = behavior_df.columns.str.lower()
eye_df.columns = eye_df.columns.str.lower()

# === 2.5. Show number of probes per block in each dataframe ===
behavior_grouped = behavior_df.groupby(['subj_orgid', 'block_num']).size().reset_index(name='behavior_probe_count')
eye_grouped = eye_df.groupby(['subj_orgid', 'block_num']).size().reset_index(name='eye_probe_count')

print("\n Number of probes per (subj_orgID, block_num) in behavior_df:")
print(behavior_grouped)

print("\n  Number of probes per (subj_orgID, block_num) in eye_df:")
print(eye_grouped)

# === 3. Count probes per (subject, block) ===
behavior_counts = behavior_grouped.rename(columns={'behavior_probe_count': 'behavior_count'})
eye_counts = eye_grouped.rename(columns={'eye_probe_count': 'eye_count'})

# === 4. Merge counts and find mismatches ===
merged_counts = pd.merge(behavior_counts, eye_counts, on=['subj_orgid', 'block_num'], how='outer', indicator=True)

mismatches = merged_counts[
    (merged_counts['behavior_count'] != merged_counts['eye_count']) |
    (merged_counts['_merge'] != 'both')
]

if not mismatches.empty:
    print("Mismatches found between behavior and eye-tracking probe counts:")
    print(mismatches)
    raise ValueError("Fix mismatches before assigning probe numbers.")

print(" Probe counts match for all (subject, block_num) pairs.")

# === 5. Assign probe_number (1 to n) inside each subject-block group ===
behavior_df['probe_number'] = behavior_df.groupby(['subj_orgid', 'block_num']).cumcount() + 1
eye_df['probe_number'] = eye_df.groupby(['subj_orgid', 'block_num']).cumcount() + 1

# === 6. Save updated CSVs ===
behavior_df.to_csv("probes_VF_indexed.csv", index=False)
eye_df.to_csv("probes_eye_VF_indexed.csv", index=False)

print("Indexed files saved as 'probes_VF_indexed.csv' and 'probes_eye_VF_indexed.csv'")

# === 7. Total and grouped summary ===
total_behavior_probes = len(behavior_df)
total_eye_probes = len(eye_df)

print(f"\n Total number of probes in behavior_df: {total_behavior_probes}")
print(f" Total number of probes in eye_df: {total_eye_probes}")

# Grouped total per subject
behavior_subject_totals = behavior_df.groupby('subj_orgid').size().reset_index(name='behavior_total_probes')
eye_subject_totals = eye_df.groupby('subj_orgid').size().reset_index(name='eye_total_probes')

print("\n Total probes per subject in behavior_df:")
print(behavior_subject_totals)

print("\n Total probes per subject in eye_df:")
print(eye_subject_totals)
