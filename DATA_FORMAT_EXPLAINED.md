# Understanding the Synthetic Data Format

## How `data.pkl` Files Are Created

The synthetic data files (like `src/data/synthetic/trend/train/data.pkl`) are created by the script [src/data/synthetic.py](src/data/synthetic.py) using the [synthesize.sh](synthesize.sh) bash script.

### Generation Process

1. **Run the synthesis script** (from [synthesize.sh](synthesize.sh)):
   ```bash
   python src/data/synthetic.py --generate \
       --data_dir data/synthetic/trend/train \
       --synthetic_func synthetic_dataset_with_trend_anomalies \
       --seed 3407
   ```

2. **What happens during generation**:
   - Creates 400 synthetic time series (default)
   - Each series has 1000 timesteps
   - Each series is univariate (1 sensor/dimension)
   - Anomalies are randomly injected based on exponential distributions
   - Generates visualization plots saved to `figs/` subdirectory
   - Saves everything to `data.pkl`

### Types of Synthetic Datasets

The codebase supports multiple anomaly types (defined in [src/data/synthetic.py](src/data/synthetic.py)):

1. **`synthetic_dataset_with_trend_anomalies`**: Anomalies appear as trend changes (slope changes)
2. **`synthetic_dataset_with_point_anomalies`**: Anomalies appear as noisy/chaotic points
3. **`synthetic_dataset_with_frequency_anomalies`**: Anomalies appear as frequency shifts in periodic data
4. **`synthetic_dataset_with_out_of_range_anomalies`**: Anomalies appear as out-of-range spikes
5. **`synthetic_dataset_with_flat_trend_anomalies`**: Similar to trend but with flatter slopes

## Data Structure in `data.pkl`

The pickle file contains a **dictionary** with two keys:

```python
{
    'series': [...],  # List of 400 time series (numpy arrays)
    'anom': [...]     # List of 400 anomaly interval lists
}
```

### 1. `series` - Time Series Data

- **Type**: `list` of numpy arrays
- **Length**: 400 items (one per time series)
- **Each item shape**: `(1000, 1)` - 1000 timesteps × 1 sensor
- **Data type**: `float64`
- **Value range**: Normalized to `[-1.0, 1.0]`

**Example structure**:
```python
series[0]  # First time series
# Shape: (1000, 1)
# Array([[-0.628], [-0.575], [-0.522], ..., [0.576]])
```

**Each row** represents one timestep value:
```python
series[0][0]    # First timestep: [-0.628]
series[0][1]    # Second timestep: [-0.575]
series[0][999]  # Last timestep: [0.576]
```

### 2. `anom` - Anomaly Intervals

- **Type**: `list` of lists of tuples
- **Length**: 400 items (one per time series)
- **Structure**: `[[list of (start, end) tuples]]` (nested because it supports multi-sensor data)

**Example structure**:
```python
anom[0]  # Anomaly intervals for first series
# [[]]  - No anomalies (empty list for sensor 0)

anom[1]  # Anomaly intervals for second series
# [[(985, 1000)]]  - One anomaly from timestep 985 to 1000

anom[2]  # Anomaly intervals for third series
# [[(800, 899)]]  - One anomaly from timestep 800 to 899

anom[4]  # Anomaly intervals for fifth series
# [[(800, 1000)]]  - One long anomaly from timestep 800 to end
```

**Interpretation**:
- `(start, end)`: Anomaly occurs from timestep `start` (inclusive) to `end` (exclusive)
- Duration = `end - start`
- An empty list `[[]]` means no anomalies (normal throughout)

## Example: Reading the Data

### Using Python Directly

```python
import pickle
import numpy as np

# Load the data
with open('src/data/synthetic/trend/train/data.pkl', 'rb') as f:
    data = pickle.load(f)

series_list = data['series']  # 400 time series
anom_list = data['anom']      # 400 anomaly interval lists

# Get first time series
first_series = series_list[0]
print(f"Shape: {first_series.shape}")  # (1000, 1)
print(f"First 5 timesteps: {first_series[:5].flatten()}")

# Get anomaly intervals for first series
first_anomalies = anom_list[0][0]  # [0] gets sensor 0's anomalies
if len(first_anomalies) > 0:
    print(f"Anomaly intervals: {first_anomalies}")
    for start, end in first_anomalies:
        print(f"  Anomaly from timestep {start} to {end}")
else:
    print("No anomalies in this series")
```

### Using the SyntheticDataset Class

```python
from src.data.synthetic import SyntheticDataset

# Load dataset
dataset = SyntheticDataset('src/data/synthetic/trend/train/')
dataset.load()

print(f"Number of series: {len(dataset)}")

# Get a specific sample
anom, series = dataset[0]  # Returns torch tensors
print(f"Series shape: {series.shape}")
print(f"Anomaly intervals: {anom}")
```

## Statistics from `trend/train/data.pkl`

Based on the data inspection:

- **Total series**: 400
- **Series shape**: (1000, 1) each
- **Series with anomalies**: 183 (45.75%)
- **Series without anomalies**: 217 (54.25%)
- **Value range**: [-1.0, 1.0] (normalized)
- **Typical anomaly duration**: 9 to 200 timesteps
- **Anomaly location**: Random, but often in middle or end of series

## How Trend Anomalies Work

For **trend anomalies** specifically ([synthetic_dataset_with_trend_anomalies](src/data/synthetic.py#L160-L244)):

1. **Normal data**: Sine wave with a slow upward trend (slope = 3.0)
2. **Anomaly**: The trend slope suddenly increases (to 6-20x faster) or decreases (negative)
3. **Detection task**: Identify where the trend changes from normal to abnormal

Example visualization:
```
Normal:  ~~~___---~~~___---~~~___---    (gentle upward trend)
Anomaly: ~~~___---~~~_______--------    (sudden fast increase)
                      ^^^^^^^ <- Anomaly region
```

## Useful Scripts

I've created two utility scripts for you:

1. **[inspect_data.py](inspect_data.py)** - View detailed information about a pickle file
   ```bash
   python inspect_data.py src/data/synthetic/trend/train/data.pkl
   ```

2. **[view_data_sample.py](view_data_sample.py)** - View multiple sample time series
   ```bash
   python view_data_sample.py src/data/synthetic/trend/train/data.pkl 10
   ```

## Directory Structure

```
src/data/synthetic/
├── trend/
│   ├── train/
│   │   ├── data.pkl          # Training data (400 series)
│   │   └── figs/             # Visualization plots (001.png, 002.png, ...)
│   └── eval/
│       ├── data.pkl          # Evaluation data (400 series, different seed)
│       └── figs/
├── point/
│   ├── train/
│   └── eval/
├── freq/
├── range/
└── ... (other anomaly types)
```

Each dataset type has both `train/` and `eval/` splits generated with different random seeds for reproducibility.
