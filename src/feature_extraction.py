import os
import numpy as np
from scipy.signal import welch
import time

start_time = time.time()  # Start timing

# List of subjects
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]  # chb01 to chb24
slice_dir = "../data/processed"

# Folder to save extracted features
features_dir = "../data/features"
os.makedirs(features_dir, exist_ok=True)

# Sampling frequency used in preprocessing (must match preprocessing.resample)
FS = 128

# Band definitions (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60)
}

# --- Helper Functions ---
def line_length(x):
    """Line length feature for a 1D array x."""
    return np.sum(np.abs(np.diff(x)))

def bandpower_from_welch(x, fs=FS, band=(0.5, 4), nperseg=256):
    """Compute band power for a single channel window using Welch PSD."""
    # If the window is shorter than nperseg, reduce nperseg
    nperseg_local = min(nperseg, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg_local)
    low, high = band
    idx = np.logical_and(f >= low, f <= high)
    if np.any(idx):
        bp = np.trapezoid(Pxx[idx], f[idx])
    else:
        bp = 0.0
    return bp

# Example: simple features per window: mean, std, max, min per channel
def extract_features(window, fs=FS):
    n_ch, n_samp = window.shape
    feats = []

    for ch in range(n_ch):
        x = window[ch]
        # basic stats
        feats.append(np.mean(x))
        feats.append(np.std(x))
        feats.append(np.min(x))
        feats.append(np.max(x))
        # line length
        feats.append(line_length(x))
        # bandpowers
        for b in BANDS.values():
            feats.append(bandpower_from_welch(x, fs=fs, band=b))
    return np.array(feats, dtype=np.float32)

# --- Main: Processing each subject---
print("Starting feature extraction for all subjects...")
 
for subject in subjects:
    subj_start = time.time()
    subj_slice_dir = os.path.join(slice_dir, subject)
    feat_subj_dir = os.path.join(features_dir, subject)
    os.makedirs(feat_subj_dir, exist_ok=True)
    
    if not os.path.isdir(subj_slice_dir):
        print(f"No processed slice folder for {subject}, skipping.")
        continue

    # Get X and y files as dicts keyed by slice index
    print(f"\nProcessing {subject} ...")
    x_files = sorted([f for f in os.listdir(subj_slice_dir) if f.startswith("X_")])
    y_files = sorted([f for f in os.listdir(subj_slice_dir) if f.startswith("y_")])
    print(f"Found {len(x_files)} X files and {len(y_files)} y files")

    if len(x_files) == 0 or len(y_files) == 0:
        print(f"No files found for {subject}. Skipping.")
        continue
    
    n_files = min(len(x_files), len(y_files))

    y_dict = {f.replace("y_", "").replace(".npy", ""): f for f in y_files}
    X_list = []
    y_list = []
    
    
    for xf in x_files:
        key = xf.replace("X_", "").replace(".npy", "")
        if key not in y_dict:
            print(f"No matching y file found for {xf}. Skipping.")
            continue
        
        X_slice = np.load(os.path.join(subj_slice_dir, xf)) # shape (n_windows, n_channels, n_samples)
        y_slice = np.load(os.path.join(subj_slice_dir, y_dict[key])) # shape (n_windows,)

        # Validate shapes
        if X_slice.ndim != 3:
            print(f"[WARN] Unexpected X slice shape {X_slice.shape} for {xf}. Skipping.")
            continue
        if y_slice.ndim != 1 or y_slice.shape[0] != X_slice.shape[0]:
            print(f"[WARN] Mismatched y slice shape {y_slice.shape} for {xf}. Skipping.")
            continue

        n_windows = X_slice.shape[0]
        expected_len = X_slice.shape[1] * (4 + 1 + len(BANDS))  # 23 * 10 = 230

        for w_idx in range(n_windows):
            window = X_slice[w_idx]  # (n_channels, n_samples)
            feats = extract_features(window, fs=FS)# expected features per channel: 4(stats)+1(line-length)+len(BANDS) = 4+1+5 = 10
            if feats.shape[0] != expected_len:
                print(
                    f"[WARN] Unexpected feature length {feats.shape[0]} vs expected {expected_len} "
                    f"for {subject} slice {xf} window {w_idx}"
                )
                continue  # Skip only this window, not the whole subject
            X_list.append(feats)
            y_list.append(y_slice[w_idx])

    if len(X_list) == 0:
        print(f"No features extracted for {subject}. Skipping.")
        continue

    # Convert to arrays
    X_all = np.stack(X_list)
    y_all = np.array(y_list, dtype=np.int8)

    # Save features for this subject
    np.save(os.path.join(feat_subj_dir, f"X_features_{subject}.npy"), X_all)
    np.save(os.path.join(feat_subj_dir, f"y_labels_{subject}.npy"), y_all)
    print(f"Features extracted for {subject}: X shape {X_all.shape}, y shape {y_all.shape}")

    subj_end = time.time()
    print(f"{subject} completed in {(subj_end - subj_start)/60:.2f} minutes")

end_time = time.time()
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes")
print("\nFeature extraction completed for all available subjects.")