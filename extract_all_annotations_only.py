import mne
import os
import glob
import re
import pandas as pd
from datetime import datetime

def find_specific_data_files(root_dir):
    """
    Find only eye-tracking .fif files from blocks 2, 4, 6, and 8
    for both session 1 and session 2 of all subjects.
    """
    blocks_of_interest = ['2', '4', '6', '8']
    patterns = []

    for block in blocks_of_interest:
        # For session 1
        patterns.append(os.path.join(root_dir, "**", f"*ses_1*_{block}_cleaned_data_raw.fif"))
        # For session 2
        patterns.append(os.path.join(root_dir, "**", f"*ses_2*_{block}_cleaned_data_raw.fif"))

    all_files = []
    for pattern in patterns:
        matched = glob.glob(pattern, recursive=True)
        all_files.extend(matched)

    return all_files

def extract_subject_info(file_path):
    path_parts = file_path.split('/')
    subject_id = None
    session_id = None
    block_id = None
    subj_orgID = None

    for part in path_parts:
        if part.startswith('sub_'):
            subject_id = part
            match = re.search(r'(\d+)', part)
            if match:
                subject_num = int(match.group(1))
                subj_orgID = f"CLONESA_G1_sub_{subject_num + 20}"
        if part.startswith('ses_'):
            session_id = part

    filename = os.path.basename(file_path)

    # Extract suffix (2, 4, 6, or 8) from filename
    match = re.search(r'ses_\d+_(\d+)_cleaned_data_raw\.fif', filename)
    if match:
        block_suffix = int(match.group(1))
        # Map suffix to block_id (1–4)
        block_map = {2: 1, 4: 2, 6: 3, 8: 4}
        block_id = block_map.get(block_suffix)

    return subject_id, session_id, block_id, subj_orgID
def extract_all_annotations(raw):
    annotations_list = []
    for ann_idx, ann in enumerate(raw.annotations):
        annotations_list.append({
            'annotation_idx': ann_idx,
            'onset_time': ann['onset'],
            'duration': ann['duration'],
            'description': ann['description']
        })
    return annotations_list

def process_all_files(root_dir):
    files = find_specific_data_files(root_dir)
    print(f"Found {len(files)} eye tracking files.")

    all_annotations = []
    blink_annotations = []
    probe_annotations = []

    for file_path in files:
        try:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            subject_id, session_id, block_id, subj_orgID = extract_subject_info(file_path)
            raw = mne.io.read_raw_fif(file_path, preload=False)
            annotations = extract_all_annotations(raw)

            for ann in annotations:
                ann['subject_id'] = subject_id
                ann['subj_orgID'] = subj_orgID
                ann['session_id'] = session_id
                ann['block_id'] = block_id
                ann['filename'] = os.path.basename(file_path)

                all_annotations.append(ann)

                # Save blinks
                if 'blink' in ann['description'].lower():
                    blink_annotations.append(ann)

                # Save probes
                if 'probe' in ann['description'].lower():
                    probe_annotations.append(ann)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    output_dir = './eye_metrics_output'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_annotations:
        ann_path = os.path.join(output_dir, f"all_annotations_only_{timestamp}.csv")
        pd.DataFrame(all_annotations).to_csv(ann_path, index=False)
        print(f"Saved all annotations to {ann_path}")

    if blink_annotations:
        blink_path = os.path.join(output_dir, f"blink_annotations_only_{timestamp}.csv")
        pd.DataFrame(blink_annotations).to_csv(blink_path, index=False)
        print(f"Saved blink annotations to {blink_path}")

    if probe_annotations:
        probe_path = os.path.join(output_dir, f"probe_annotations_only_{timestamp}.csv")
        pd.DataFrame(probe_annotations).to_csv(probe_path, index=False)
        print(f"Saved probe annotations to {probe_path}")
    else:
        print(" No probe annotations found.")

if __name__ == "__main__":
    root_dir = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/22participantsfulldata/22eyetracking"
    process_all_files(root_dir)
