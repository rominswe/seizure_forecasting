import os
import mne
import warnings

warnings.filterwarnings("ignore")

# List of subjects
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]  # chb01 to chb24

# Base data directory
base_dir = "../data/physionet.org/files/chbmit/1.0.0"

# EEG parameters
sfreq_target = 128       # Target downsample frequency
window_sec = 15          # Window length in seconds
window_size = window_sec * sfreq_target

for subject in subjects:
    subj_path = os.path.join(base_dir, subject)
    edf_files = [f for f in os.listdir(subj_path) if f.endswith(".edf")]

    print(f"\n--- {subject.upper()} ---")
    for fname in edf_files:
        fpath = os.path.join(subj_path, fname)
        try:
            raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
            n_samples = raw.n_times
            n_total_windows = n_samples // window_size
            print(f"{fname}: total samples = {n_samples}, total 15s windows = {n_total_windows}")
            del raw
        except Exception as e:
            print(f"Error with {fname}: {e}")