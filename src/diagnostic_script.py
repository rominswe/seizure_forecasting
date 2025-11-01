import os
import numpy as np

features_dir = "../data/features"
subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]  # chb01 to chb24

for subject in subjects:
    y_path = os.path.join(features_dir, subject, f"y_labels_{subject}.npy")
    if os.path.exists(y_path):
        y = np.load(y_path)
        unique, counts = np.unique(y, return_counts=True)
        print(f"{subject}: Label distribution -> {dict(zip(unique, counts))}")
    else:
        print(f"{subject}: Missing y_labels file.")
    