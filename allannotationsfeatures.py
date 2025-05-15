import pandas as pd
import numpy as np
import os

# === CONFIGURATION ===
annotations_csv = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/eye_metrics_output/all_annotations_only_20250515_120758.csv"
probes_csv = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/probes_eye_VF_indexed.csv"
output_dir = "outputs"
output_csv = os.path.join(output_dir, "sliding_window_eye_features_labeled.csv")

window_size = 2       # seconds
total_window = 10     # seconds before probe
n_windows = total_window - window_size + 1  # = 9 windows

# === CREATE OUTPUT DIR ===
os.makedirs(output_dir, exist_ok=True)

# === LOAD ANNOTATIONS ===
annotations_df = pd.read_csv(annotations_csv)
annotations_df['onset_time'] = annotations_df['onset_time'].astype(float)
annotations_df['description_upper'] = annotations_df['description'].str.upper()

# === FILTER RELEVANT EVENTS ===
filtered = annotations_df[
    annotations_df['description_upper'].isin(['FIXATION', 'SACCADE', 'BLINK', 'PROBES_ONSET'])
].copy()
filtered.sort_values(by='onset_time', inplace=True)

probe_events = filtered[filtered['description_upper'] == 'PROBES_ONSET']
event_map = {
    'FIXATION': filtered[filtered['description_upper'] == 'FIXATION']['onset_time'].values,
    'SACCADE': filtered[filtered['description_upper'] == 'SACCADE']['onset_time'].values,
    'BLINK': filtered[filtered['description_upper'] == 'BLINK']['onset_time'].values
}

# === EXTRACT FEATURES FOR EACH PROBE ===
output_rows = []

for _, probe_row in probe_events.iterrows():
    probe_time = probe_row['onset_time']
    row = {
        'subj_orgID': probe_row.get('subj_orgID', 'unknown'),
        'subject_id': probe_row.get('subject_id', 'unknown'),
        'session_id': probe_row.get('session_id', 'unknown'),
        'block_num': probe_row.get('block_id', np.nan),  # block_id here is treated as block_num
    }

    for i in range(n_windows):
        window_start = probe_time - total_window + i
        window_end = window_start + window_size

        row[f'fixation_w{i}'] = np.sum((event_map['FIXATION'] >= window_start) & (event_map['FIXATION'] < window_end))
        row[f'saccade_w{i}']  = np.sum((event_map['SACCADE']  >= window_start) & (event_map['SACCADE']  < window_end))
        row[f'blink_w{i}']    = np.sum((event_map['BLINK']    >= window_start) & (event_map['BLINK']    < window_end))

    output_rows.append(row)

# === CONVERT TO DATAFRAME ===
df_out = pd.DataFrame(output_rows)

# === CLEAN & ALIGN FOR MERGING ===
df_out['subj_orgID'] = df_out['subj_orgID'].astype(str).str.strip()
df_out['session_id'] = df_out['session_id'].astype(str).str.strip()
df_out['block_num'] = pd.to_numeric(df_out['block_num'], errors='coerce')

probes_info = pd.read_csv(probes_csv)
probes_info.columns = probes_info.columns.str.lower()
probes_info['subj_orgid'] = probes_info['subj_orgid'].astype(str).str.strip()
probes_info['session'] = probes_info['session'].astype(str).str.strip()
probes_info['block_num'] = pd.to_numeric(probes_info['block_num'], errors='coerce')
probes_info['probe_number'] = pd.to_numeric(probes_info['probe_number'], errors='coerce')

df_out = df_out.sort_values(by=['subj_orgID', 'session_id', 'block_num']).reset_index(drop=True)
df_out['probe_number'] = df_out.groupby(['subj_orgID', 'session_id', 'block_num']).cumcount() + 1

df_merged = pd.merge(
    df_out,
    probes_info[['subj_orgid', 'session', 'block_num', 'probe_number']],
    left_on=['subj_orgID', 'session_id', 'block_num', 'probe_number'],
    right_on=['subj_orgid', 'session', 'block_num', 'probe_number'],
    how='left'
)

df_merged.drop(columns=['subj_orgid', 'session'], inplace=True)
df_merged.to_csv(output_csv, index=False)

print(f"\nFinal CSV saved: {output_csv}")
print(" Rows:", df_merged.shape[0])

