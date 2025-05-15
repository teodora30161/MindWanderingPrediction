import pandas as pd
import numpy as np
import os

# === CONFIGURATION ===
all_data_path = '/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/22participantsfulldata/all_data_VF.csv'  # Update this
probes_data_path = '/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/probes_VF_indexed.csv'  # Update this
output_path = 'sliding_window_behavioral_features_labeled.csv'

# === LOAD DATA ===
all_data = pd.read_csv(all_data_path, sep=';')
probes_data = pd.read_csv(probes_data_path)

# === STANDARDIZE COLUMNS ===
all_data.columns = all_data.columns.str.lower()
probes_data.columns = probes_data.columns.str.lower()
all_data.rename(columns={'subj_orgid': 'subj_orgID'}, inplace=True)
probes_data.rename(columns={'subj_orgid': 'subj_orgID', 'session': 'session_id'}, inplace=True)

# === FILTER EVENTS ===
all_data = all_data[all_data['stimulus'].str.strip().str.lower() == 'tap']
probes_data = probes_data[probes_data['stimulus'].str.strip().str.lower() == 'probe_onset']

# === TYPE CASTING ===
all_data['time'] = pd.to_numeric(all_data['time'], errors='coerce')
all_data['tap_diff'] = pd.to_numeric(all_data['tap_diff'], errors='coerce')
probes_data['time'] = pd.to_numeric(probes_data['time'], errors='coerce')

# === SORT ===
all_data = all_data.sort_values(['subj_orgID', 'time'])
probes_data = probes_data.sort_values(['subj_orgID', 'session_id', 'block_num', 'probe_number'])

# === FEATURE CALCULATION FUNCTIONS ===
def calculate_bv(window):
    values = window['tap_diff'].dropna().values
    return np.std(values) if len(values) > 0 else 0

def calculate_ae(window):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    U = window['tap_diff'].dropna().values
    N = len(U)
    m = 2
    r = 0.5 * np.std(U)
    if N < m + 1:
        return 0
    return abs(_phi(m + 1) - _phi(m))

# === SLIDING WINDOW EXTRACTION ===
window_size = 2
total_window = 10
n_windows = total_window - window_size + 1

output_rows = []

for _, probe in probes_data.iterrows():
    subj = probe['subj_orgID']
    session = probe['session_id']
    block = probe['block_num']
    probe_num = probe['probe_number']
    probe_time = probe['time']

    subject_taps = all_data[all_data['subj_orgID'] == subj]

    row = {
        'subj_orgID': subj,
        'session_id': session,
        'block_num': block,
        'probe_number': probe_num
    }

    for i in range(n_windows):
        start = probe_time - total_window + i
        end = start + window_size
        window = subject_taps[(subject_taps['time'] >= start) & (subject_taps['time'] < end)]
        row[f'ApEn_win{i}'] = calculate_ae(window)
        row[f'BV_std_win{i}'] = calculate_bv(window)

    output_rows.append(row)

# === SAVE OUTPUT ===
df_out = pd.DataFrame(output_rows)
df_out.to_csv(output_path, index=False)
print(f" Behavioral features saved to: {output_path}")
