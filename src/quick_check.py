import os
import numpy as np

# Base processed data directory
processed_dir = "../data/processed"

# List all subject folders
subjects = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]

for subject in subjects:
    subj_dir = os.path.join(processed_dir, subject)
    y_files = [f for f in os.listdir(subj_dir) if f.startswith("y_") and f.endswith(".npy")]
    
    print(f"\n--- {subject.upper()} ---")
    
    for y_file in sorted(y_files):
        y_path = os.path.join(subj_dir, y_file)
        try:
            y = np.load(y_path)
            unique_labels = np.unique(y)
            print(f"{y_file}: {unique_labels}")
        except Exception as e:
            print(f"{y_file}: ERROR -> {e}")