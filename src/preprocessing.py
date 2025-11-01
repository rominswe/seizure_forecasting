import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import mne
import warnings

warnings.filterwarnings("ignore")

# List of subjects to process
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]  # chb01 to chb24

# Base data directory
base_dir = "../data/physionet.org/files/chbmit/1.0.0"

# EEG parameters
sfreq_target = 128       # Downsample frequency
window_sec = 15          # Window length in seconds
window_size = window_sec * sfreq_target
slice_windows = 1000      # number of windows per slice to load (change based on memory)

for subject in subjects:
    subj_path = os.path.join(base_dir, subject)
    summary_path = os.path.join(subj_path, f"{subject}-summary.txt")

    if not os.path.exists(summary_path):
        print(f"Summary file not found for {subject}, skipping.")
        continue

    print(f"\n--- Processing {subject.upper()} ---")
    print("Summary exists:", True)

    # --- Parse summary for seizure intervals ---
    seizures = []
    with open(summary_path, "r") as f:
        lines = f.readlines()

    current_file = None
    for line in lines:
        line = line.strip()
        if "File Name:" in line:
            current_file = line.split(":")[-1].strip()
        elif "Seizure Start Time:" in line:
            start = int(line.split(":")[-1].replace("seconds", "").strip())
        elif "Seizure End Time:" in line:
            end = int(line.split(":")[-1].replace("seconds", "").strip())
            if current_file:
                seizures.append((current_file, start, end))

    print(f"Total seizure records: {len(seizures)}")

    # --- Output folder for preprocessed chunks ---
    custom_dir = ("../data/processed")
    save_dir = os.path.join(custom_dir, subject)
    os.makedirs(save_dir, exist_ok=True)

    # --- Loop through EDF files ---
    edf_files = [f for f in os.listdir(subj_path) if f.endswith(".edf")]
    print(f"Found {len(edf_files)} EDF files in {subject}")

    for fname in edf_files:
        fpath = os.path.join(subj_path, fname)
        print(f"  → Processing {fname}")

        try:
            raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
            n_channels, n_samples = len(raw.ch_names), raw.n_times
            n_total_windows = n_samples // window_size

            # Slice-by-slice processing
            for start_win in range(0, n_total_windows, slice_windows):
                stop_win = min(start_win + slice_windows, n_total_windows)
                start_sample = start_win * window_size
                stop_sample = stop_win * window_size

                # Load only this slice
                data_slice = raw.get_data(start=start_sample, stop=stop_sample).astype(np.float32)
                
                if data_slice.shape[0] > 23:
                    data_slice = data_slice[:23, :]
                elif data_slice.shape[0] < 23:
                    print(f"Skipping {fname}: only {data_slice.shape[0]} channels.")
                    continue

                # Resample slice
                info = mne.create_info(data_slice.shape[0], raw.info['sfreq'], ch_types='eeg')
                raw_slice = mne.io.RawArray(data_slice, info)
                raw_slice.resample(sfreq_target, npad="auto")
                data_slice = raw_slice.get_data().astype(np.float32)

                # Split into 15s windows (already sliced)
                windows = np.array(np.split(data_slice, stop_win - start_win, axis=1), dtype=np.float32)

                # Create seizure labels
                seizure_labels = np.zeros(stop_win - start_win, dtype=np.int8)

                # Collect all seizure intervals for the current file
                seizure_intervals = [(s, e) for file, s, e in seizures if file == fname]
                if len(seizure_intervals) > 0:
                    for s, e in seizure_intervals:
                        # Convert seizure time (seconds) into window indices
                        start_label = int(s // window_sec)
                        end_label = int(e // window_sec)

                        #Adjust for current slice
                        slice_start = start_win
                        slice_end = stop_win - 1
                        overlap_start = max(start_label, slice_start)
                        overlap_end = min(end_label, slice_end)

                        if overlap_start <= overlap_end:
                            seizure_labels[overlap_start - slice_start : overlap_end - slice_start + 1] = 1

                # Normalize per window
                X_reshaped = windows.reshape(windows.shape[0], -1)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_reshaped)
                X_scaled = X_scaled.reshape(windows.shape).astype(np.float32)

                # Save slice immediately
                slice_name = f"{fname[:-4]}_slice{start_win}-{stop_win-1}.npy"
                np.save(os.path.join(save_dir, f"X_{slice_name}"), X_scaled)
                np.save(os.path.join(save_dir, f"y_{slice_name}"), seizure_labels)

            del raw

        except MemoryError:
            print(f"MemoryError while processing {fname}. Skipping this file.")
            continue
        except Exception as e:
            print(f"Error with {fname}: {e}")
            continue

    print(f"\nFinished preprocessing for {subject}")
    print(f"All processed chunks saved in: {save_dir}")