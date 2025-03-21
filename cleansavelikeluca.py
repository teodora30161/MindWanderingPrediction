import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def clean_column_names(df):
    """Cleans column names by removing extra characters & spaces."""
    df.columns = df.columns.str.strip().str.replace("'", "").str.replace(":", "").str.lower()
    return df

def load_and_clean_bhv_data(base_dir):
    """Loads and cleans all behavioral CSV files."""
    if not os.path.exists(base_dir):
        print(f"Behavioral data directory not found: {base_dir}")
        return pd.DataFrame()
    
    all_data = []
    for subject in sorted(os.listdir(base_dir)):
        subj_path = os.path.join(base_dir, subject)
        if not os.path.isdir(subj_path):
            continue

        for session in sorted(os.listdir(subj_path)):
            session_path = os.path.join(subj_path, session)
            if not os.path.isdir(session_path):
                continue

            for file in sorted(os.listdir(session_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(session_path, file)
                    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
                    df = clean_column_names(df)
                    df["subj"] = subject
                    df["session"] = session
                    all_data.append(df)

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(merged_df)} rows from {len(all_data)} behavioral files")
        return merged_df
    else:
        print("No behavioral files found!")
        return pd.DataFrame()

def load_eye_tracking_data(eye_dir):
    """Loads all raw eye-tracking `.asc` files."""
    if not os.path.exists(eye_dir):
        print(f"Eye-Tracking data directory not found: {eye_dir}")
        return pd.DataFrame()
    
    all_eye = []
    for subject in sorted(os.listdir(eye_dir)):
        subj_path = os.path.join(eye_dir, subject)
        if not os.path.isdir(subj_path):
            continue

        for session in sorted(os.listdir(subj_path)):
            session_path = os.path.join(subj_path, session)
            if not os.path.isdir(session_path):
                continue

            for file in sorted(os.listdir(session_path)):
                if file.endswith(".asc"):
                    file_path = os.path.join(session_path, file)
                    raw_et = pd.read_csv(file_path, sep="\t", encoding="utf-8", error_bad_lines=False)
                    pupil_data = raw_et.mean(numeric_only=True).to_dict()
                    pupil_data["subj"] = subject
                    pupil_data["session"] = session
                    all_eye.append(pupil_data)

    if all_eye:
        eye_df = pd.DataFrame(all_eye)
        print(f"Loaded Eye-Tracking data from {len(eye_df)} files")
        return eye_df
    else:
        print("No Eye-Tracking files found!")
        return pd.DataFrame()

def merge_all_data(bhv_df, eye_df):
    """Merges Behavioral and Eye-Tracking data into a single DataFrame."""
    if "subj" not in bhv_df.columns or "session" not in bhv_df.columns:
        print("Error: Behavioral data missing 'subj' or 'session' columns")
        return pd.DataFrame()
    
    if not eye_df.empty:
        merged_df = bhv_df.merge(eye_df, on=["subj", "session"], how="left")
    else:
        merged_df = bhv_df  # No Eye-Tracking data available, return only behavioral data
    
    print(f"Final merged dataset with {len(merged_df)} rows")
    return merged_df

bhv_dir = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/1_raw"
eye_dir = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/eye/2_fif"

bhv_data = load_and_clean_bhv_data(bhv_dir)
eye_data = load_eye_tracking_data(eye_dir)

final_df = merge_all_data(bhv_data, eye_data)

final_csv_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/final_dataset.csv"
final_df.to_csv(final_csv_path, index=False)
print(f"Final dataset saved at: {final_csv_path}")
