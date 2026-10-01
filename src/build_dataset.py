import numpy as np
import pandas as pd
import os
import hashlib
import sqlite3

#Creates hash for image
def get_image_hash(img):
    return hashlib.sha256(img.tobytes()).hexdigest()

#Check that path is there from download process
#print(os.path.exists('./output/pneumoniamnist.npz'))

data = np.load('./output/pneumoniamnist.npz')

records = []

# Process each group (train, val, test)
for group in ["train", "val", "test"]: 
    images = data[f'{group}_images']
    labels = data[f'{group}_labels'].squeeze() #Flatten 2D array to 1D (had each result being its own list)
    print(f"Processing group '{group}': {len(images)} images...")

    for idx, (img, label) in enumerate(zip(images, labels)):

        h, w = img.shape
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        img_hash = get_image_hash(img)

        records.append({
            'image_id': f"{group}_{idx:05d}",
            'split': group,
            'label': int(label),
            'width': w,
            'height': h,
            'mean_intensity': round(mean_val, 4),
            'std_intensity': round(std_val, 4),
            'min_intensity': int(min_val),
            'max_intensity': int(max_val),
            'image_hash': img_hash
        })
            
table_df = pd.DataFrame(records)


