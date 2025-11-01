import os
import numpy as np

# List of subjects
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]  # chb01 to chb24

# Base directory where slices are saved
base_slice_dir = "../data/processed"

for subject in subjects:
    subject_dir = os.path.join(base_slice_dir, subject)
    if not os.path.exists(subject_dir):
        print(f"Folder not found for {subject}, skipping.")
        continue

    # List all slice files for this subject
    x_files = sorted([f for f in os.listdir(subject_dir) if f.startswith("X_") and f.endswith(".npy")])
    y_files = sorted([f for f in os.listdir(subject_dir) if f.startswith("y_") and f.endswith(".npy")])

    print(f"\n--- Inspecting slices for {subject.upper()} ---")
    for xf, yf in zip(x_files, y_files):
        X_slice = np.load(os.path.join(subject_dir, xf))
        y_slice = np.load(os.path.join(subject_dir, yf))
        print(f"{xf}: X shape {X_slice.shape}, y shape {y_slice.shape}, seizures {np.sum(y_slice)}")