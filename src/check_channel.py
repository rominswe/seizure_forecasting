import os
import mne
import warnings

warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings("ignore", message="Scaling factor is not defined")

base_dir = "../data/physionet.org/files/chbmit/1.0.0"
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]  # chb01 to chb24

print("=== Channel Count Summary ===\n")

for subject in subjects:
    subj_path = os.path.join(base_dir, subject)
    edf_files = [f for f in os.listdir(subj_path) if f.endswith(".edf")]

    print(f"\n--- {subject.upper()} ---")
    for fname in edf_files:
        fpath = os.path.join(subj_path, fname)
        try:
            raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
            n_channels = len(raw.ch_names)
            print(f"{fname}: {n_channels} channels")
        except Exception as e:
            print(f"{fname}: Error reading file → {e}")