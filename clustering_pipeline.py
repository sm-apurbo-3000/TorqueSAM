# integration_example.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_extractor import process_dataset
from torque_clustering_robust import (
    torque_clustering,
    plot_clustering_results
)

def run_clustering_pipeline(image_dir, mask_dir, n_clusters=2):
    """
    Complete pipeline: Extract features -> Cluster -> Visualize
    """
    # Step 1: Extract features
    df = process_dataset(image_dir, mask_dir, output_csv="features.csv")
    
    # Step 2: Convert to numpy array
    feature_columns = ['mean_hu', 'glcm_contrast']
    data = df[feature_columns].values
    
    # Step 3: Handle missing/zero values
    valid_mask = ~np.isnan(data).any(axis=1) & (data != 0).any(axis=1)
    data_clean = data[valid_mask]
    filenames_clean = df['filename'].values[valid_mask]
    
    # Step 4: Normalize features (CRITICAL for torque clustering)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    
    # Step 5: Run clustering
    labels, transformed = torque_clustering(
        data_scaled,
        method='iterative',
        mass_method='density',
        n_clusters=n_clusters,
        verbose=True,
        visualize_iterations=True
    )
    
    # Step 6: Add labels to DataFrame
    result_df = pd.DataFrame({
        'filename': filenames_clean,
        'mean_hu': data_clean[:, 0],
        'glcm_contrast': data_clean[:, 1],
        'cluster': labels
    })
    
    # Step 7: Visualize
    plot_clustering_results(
        data_scaled,
        {'Torque Clustering': labels},
        title="CT Image Clustering by Features"
    )
    
    return result_df, labels

if __name__ == "__main__":
    result, labels = run_clustering_pipeline(r"E:\T2510638\TorqueSAM\RenSeg-main\RenSeg-main\data\test", r"E:\T2510638\TorqueSAM\RenSeg-main\RenSeg-main\npy", n_clusters=2)
    result.to_csv("clustered_features.csv", index=False)
    print(result)