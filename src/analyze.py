import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths
DB_PATH = './output/metadata.db'
NPZ_PATH = './output/pneumoniamnist.npz'
OUTPUT_DIR = './output'
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, 'summary.csv')
UNUSUAL_IMG_PATH = os.path.join(OUTPUT_DIR, 'unusual_images.png')

def connect_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database '{DB_PATH}' not found. Run Part 1 script first.")
    return sqlite3.connect(DB_PATH)

# ==========================================
# Part 2A: Class Balance Analysis
# ==========================================
def analyze_class_balance(conn):
    print("\n" + "="*50)
    print("PART 2A: CLASS BALANCE REPORT")
    print("="*50)
    
    query = """
    SELECT 
        split,
        COUNT(*) AS total_images,
        SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS normal_count,
        SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS pneumonia_count,
        ROUND(100.0 * SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS percentage_pneumonia
    FROM image_metadata
    GROUP BY split
    ORDER BY CASE split 
        WHEN 'train' THEN 1 
        WHEN 'val' THEN 2 
        WHEN 'test' THEN 3 
    END;
    """
    df_class_balance = pd.read_sql_query(query, conn)
    print(df_class_balance.to_string(index=False))
    return df_class_balance

# ==========================================
# Part 2B: Image Characteristics Analysis
# ==========================================
def analyze_image_characteristics(conn):
    print("\n" + "="*50)
    print("PART 2B: IMAGE CHARACTERISTICS COMPARISON")
    print("="*50)
    
    query = """
    SELECT 
        CASE label 
            WHEN 0 THEN 'Normal' 
            WHEN 1 THEN 'Pneumonia' 
        END AS class_label,
        COUNT(*) AS sample_count,
        ROUND(AVG(mean_intensity), 2) AS avg_mean_intensity,
        ROUND(AVG(std_intensity), 2) AS avg_std_intensity,
        ROUND(MIN(min_intensity), 2) AS overall_min_intensity,
        ROUND(MAX(max_intensity), 2) AS overall_max_intensity
    FROM image_metadata
    GROUP BY label;
    """
    df_stats = pd.read_sql_query(query, conn)
    print(df_stats.to_string(index=False))
    return df_stats

# ==========================================
# Part 2C: Outlier / Unusual Image Selection
# ==========================================
def find_unusual_images(conn):
    print("\n" + "="*50)
    print("PART 2C: POTENTIALLY UNUSUAL IMAGES")
    print("="*50)
    
    # Quantitative selection criteria:
    # 1. Dark Outliers: Extremely low mean intensity (< mean - 2.5 * std across dataset)
    # 2. Bright Outliers: Extremely high mean intensity (> mean + 2.5 * std across dataset)
    # 3. Low Contrast Outliers: Extremely low pixel std (< 15.0)
    query = """
    WITH GlobalStats AS (
        SELECT 
            AVG(mean_intensity) AS global_mean,
            AVG(std_intensity) AS global_std
        FROM image_metadata
    )
    SELECT 
        m.image_id,
        m.split,
        CASE m.label WHEN 0 THEN 'Normal' ELSE 'Pneumonia' END AS class_name,
        m.mean_intensity,
        m.std_intensity,
        m.min_intensity,
        m.max_intensity,
        CASE 
            WHEN m.mean_intensity < (g.global_mean - 35) THEN 'Unusually Dark'
            WHEN m.mean_intensity > (g.global_mean + 35) THEN 'Unusually Bright'
            WHEN m.std_intensity < 20.0 THEN 'Very Low Contrast'
            ELSE 'Normal Profile'
        END AS anomaly_category
    FROM image_metadata m, GlobalStats g
    WHERE m.mean_intensity < (g.global_mean - 35)
       OR m.mean_intensity > (g.global_mean + 35)
       OR m.std_intensity < 20.0
    ORDER BY anomaly_category, m.mean_intensity ASC
    LIMIT 6;
    """
    df_unusual = pd.read_sql_query(query, conn)
    print(df_unusual.to_string(index=False))
    return df_unusual

def plot_unusual_images(df_unusual):
    """Loads raw pixel data from .npz and renders candidate unusual images."""
    if df_unusual.empty:
        print("No unusual images matched criteria.")
        return

    npz_data = np.load(NPZ_PATH)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, row in df_unusual.iterrows():
        if idx >= len(axes):
            break
        
        split, img_idx_str = row['image_id'].split('_')
        img_idx = int(img_idx_str)
        img_array = npz_data[f"{split}_images"][img_idx]

        ax = axes[idx]
        ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)
        ax.set_title(
            f"ID: {row['image_id']} ({row['class_name']})\n"
            f"Category: {row['anomaly_category']}\n"
            f"Mean: {row['mean_intensity']} | Std: {row['std_intensity']}",
            fontsize=9
        )
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(df_unusual), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(UNUSUAL_IMG_PATH, dpi=150)
    plt.close()
    print(f"\nSaved unusual images preview grid to: {UNUSUAL_IMG_PATH}")

def export_summary(df_class, df_stats):
    """Exports Part 2 summary metrics to output/summary.csv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(SUMMARY_CSV_PATH, 'w') as f:
        f.write("# PART 2A: CLASS BALANCE\n")
        df_class.to_csv(f, index=False)
        f.write("\n# PART 2B: IMAGE CHARACTERISTICS\n")
        df_stats.to_csv(f, index=False)

    print(f"Saved aggregated metrics to: {SUMMARY_CSV_PATH}")

if __name__ == '__main__':
    conn = connect_db()
    
    # Run steps
    df_class = analyze_class_balance(conn)
    df_stats = analyze_image_characteristics(conn)
    df_unusual = find_unusual_images(conn)
    
    # Save visualizations & summary CSV
    plot_unusual_images(df_unusual)
    export_summary(df_class, df_stats)
    
    conn.close()