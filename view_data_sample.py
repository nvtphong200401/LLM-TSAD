"""
Script to view multiple samples from the data.pkl file
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

def view_samples(filepath, num_samples=5):
    """Load and display multiple samples from the pickle file"""

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    series_list = data['series']
    anom_list = data['anom']

    print(f"\n{'='*70}")
    print(f"Data Summary for: {filepath}")
    print(f"{'='*70}\n")
    print(f"Total number of time series: {len(series_list)}")
    print(f"Each time series shape: {series_list[0].shape}")
    print(f"Data type: {series_list[0].dtype}\n")

    # Count series with and without anomalies
    series_with_anomalies = sum(1 for anom in anom_list if len(anom[0]) > 0)
    print(f"Series with anomalies: {series_with_anomalies}")
    print(f"Series without anomalies: {len(anom_list) - series_with_anomalies}\n")

    print(f"{'='*70}")
    print(f"Viewing {num_samples} sample time series:")
    print(f"{'='*70}\n")

    for i in range(min(num_samples, len(series_list))):
        series = series_list[i]
        anom = anom_list[i]

        print(f"\n--- Sample {i+1} ---")
        print(f"Time series shape: {series.shape}")
        print(f"Length: {len(series)} timesteps")

        if len(anom[0]) > 0:
            print(f"[X] HAS ANOMALIES ({len(anom[0])} anomaly intervals):")
            for j, (start, end) in enumerate(anom[0]):
                duration = end - start
                print(f"  Anomaly {j+1}: timesteps [{start}, {end}) - duration: {duration} steps")
                # Show the values at anomaly region
                anomaly_values = series[start:min(start+5, end)].flatten()
                print(f"    Values at anomaly start: {anomaly_values}")
        else:
            print(f"[OK] NO ANOMALIES (normal throughout)")

        # Show some statistics
        print(f"Statistics:")
        print(f"  - Min: {series.min():.4f}, Max: {series.max():.4f}")
        print(f"  - Mean: {series.mean():.4f}, Std: {series.std():.4f}")

        # Show first few and last few values
        print(f"First 5 values: {series[:5].flatten()}")
        print(f"Last 5 values: {series[-5:].flatten()}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        filepath = "src/data/synthetic/trend/train/data.pkl"
        num_samples = 10

    view_samples(filepath, num_samples)
