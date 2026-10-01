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

#Validation
print("\nDataset Validations")
#1.	Number of images in each split.
print("\nNumber of images in each split")
print(table_df['split'].value_counts().to_string())
#2.	Allowed label values.
print("\nUnique Label Values: (should be only 0 and 1)")
print(set(table_df['label'].unique()))
#3.	Image dimensions.
print("\nImage Dimensions")
print(table_df[['width', 'height']].drop_duplicates().to_string())
#4.	Missing or invalid values.
print("\nMissing/Null Value")
null_counts = table_df.isnull().sum().sum()
print("Total null/missing values:", null_counts)
#5.	Images with unusually low pixel variance.
print("\nLow Pixel Variance (std_intensity < 5.0):")
low_var = table_df[table_df['std_intensity'] < 5.0]
print("Images with std_intensity < 5.0:", len(low_var))
#6.	Whether exact duplicate images occur.
print("\nExact Duplicate Detection (by SHA-256 hash):")
duplicate_hashes = table_df[table_df.duplicated(subset=['image_hash'], keep=False)]
unique_duplicates = duplicate_hashes['image_hash'].nunique()
print(f"Total duplicate image instances: {len(duplicate_hashes)}")
print(f"Unique duplicate image content hashes: {unique_duplicates}")


DB_PATH = './output/metadata.db'
PARQUET_PATH = './output/metadata.parquet'

#Save to SQLite
conn = sqlite3.connect(DB_PATH)
table_df.to_sql('image_metadata', conn, if_exists='replace', index=False)
conn.close()

# Save to Parquet
table_df.to_parquet(PARQUET_PATH, index=False)

