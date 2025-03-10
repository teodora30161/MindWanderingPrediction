import mne
import matplotlib.pyplot as plt
import numpy as np

eye_file = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/eye/3_clean/sub_01/ses_2/sub_01_ses_2_1.0_cleaned_data.fif"  

raw_eye = mne.io.read_raw_fif(eye_file, preload=True)

eye_data = raw_eye.get_data()  # Extract raw numerical data
sfreq = raw_eye.info['sfreq']  # Sampling frequency
time = np.arange(eye_data.shape[1]) / sfreq  # Create time vector

# Assuming the first channel corresponds to pupil size or eye movement
plt.figure(figsize=(12, 6))
plt.plot(time, eye_data[0, :], label="Eye Tracking Signal")

plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.title("Eye Tracking Data Visualization")
plt.legend()
plt.grid(True)
plt.show()
