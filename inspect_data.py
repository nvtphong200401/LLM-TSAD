"""
Script to inspect the contents of data.pkl files
"""
import pickle
import numpy as np
import sys

def inspect_pickle_file(filepath):
    """Load and display information about a pickle file"""
    print(f"\n{'='*70}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*70}\n")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"Type of data: {type(data)}")
    print(f"Keys in data: {data.keys() if isinstance(data, dict) else 'N/A'}\n")

    if isinstance(data, dict):
        # Check if it has 'series' and 'anom' keys
        if 'series' in data:
            series_list = data['series']
            print(f"Number of time series: {len(series_list)}")
            print(f"Type of series list: {type(series_list)}\n")

            # Show first series details
            if len(series_list) > 0:
                first_series = series_list[0]
                print(f"First Time Series:")
                print(f"  - Shape: {first_series.shape}")
                print(f"  - Data type: {first_series.dtype}")
                print(f"  - Min value: {first_series.min():.4f}")
                print(f"  - Max value: {first_series.max():.4f}")
                print(f"  - Mean: {first_series.mean():.4f}")
                print(f"  - Std: {first_series.std():.4f}")
                print(f"\n  - First 10 values:")
                print(f"    {first_series[:10].flatten()}\n")
                print(f"  - Last 10 values:")
                print(f"    {first_series[-10:].flatten()}\n")

        if 'anom' in data:
            anom_list = data['anom']
            print(f"Number of anomaly interval lists: {len(anom_list)}")
            print(f"Type of anomaly list: {type(anom_list)}\n")

            # Show first anomaly intervals
            if len(anom_list) > 0:
                first_anom = anom_list[0]
                print(f"First Anomaly Intervals (for first series):")
                print(f"  - Type: {type(first_anom)}")
                print(f"  - Number of sensors: {len(first_anom)}")
                if len(first_anom) > 0 and len(first_anom[0]) > 0:
                    print(f"  - Anomaly intervals for sensor 0: {first_anom[0]}")
                    print(f"\n  Explanation:")
                    for i, (start, end) in enumerate(first_anom[0]):
                        print(f"    Anomaly {i+1}: timesteps {start} to {end} (duration: {end-start})")
                else:
                    print(f"  - No anomalies in first series")

                print(f"\n  Example: Second series anomaly intervals:")
                if len(anom_list) > 1:
                    second_anom = anom_list[1]
                    if len(second_anom) > 0 and len(second_anom[0]) > 0:
                        print(f"    {second_anom[0]}")
                    else:
                        print(f"    No anomalies")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default to trend/train/data.pkl
        filepath = "src/data/synthetic/trend/train/data.pkl"

    try:
        inspect_pickle_file(filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        print("\nUsage: python inspect_data.py [path/to/data.pkl]")
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
