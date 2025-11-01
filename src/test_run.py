import os
import numpy as np

slice_dir = "../data/processed"
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)] # chb01 to chb24

for subject in subjects:
    subj_dir = os.path.join(slice_dir, subject)
    x_files = sorted([f for f in os.listdir(subj_dir) if f.startswith("X_")])
    if not x_files:
        continue
    # Load one file to check number of channels
    sample = np.load(os.path.join(subj_dir, x_files[0]))
    print(f"{subject}: {sample.shape[1]} channels")