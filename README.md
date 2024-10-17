### README File for GitHub

# Wine Quality Dataset with PyTorch DataLoader

## Overview

This repository demonstrates how to:
1. Load a dataset using pandas.
2. Split the dataset into features and labels.
3. Convert the data into PyTorch tensors.
4. Use the PyTorch `DataLoader` to load data in batches.

The dataset used here is the **Wine Quality Dataset**. The goal is to predict the quality of wine based on several chemical properties.

## Dataset

The dataset used in this project is the **Wine Quality Dataset** from UCI Machine Learning Repository. It contains several features about the chemical properties of red wine and a target variable, `quality`, which is a score given to the wine.

### Dataset Information:

- **Input Features**: Chemical properties like acidity, sugar, alcohol content, etc.
- **Target**: `quality`, a score between 0 and 10 indicating the quality of the wine.

## Requirements

- Python 3.x
- PyTorch
- Pandas
- NumPy

You can install the necessary dependencies using pip:

```bash
pip install pandas numpy torch
```

## Steps

### 1. Load the Dataset

```python
import pandas as pd

# Load the dataset using pandas
df = pd.read_csv("path_to_dataset/winequality-red.csv", delimiter=';')

# Display some basic information
df.head()
df.info()
```

### 2. Feature and Label Splitting

```python
# Split features (X) and labels (Y)
x = df.drop("quality", axis=1).values
y = df["quality"].values
```

### 3. Convert to PyTorch Tensors

```python
import torch

# Convert features and labels to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Check the shapes of the tensors
print("X Tensor Shape:", x_tensor.shape)
print("Y Tensor Shape:", y_tensor.shape)
```

### 4. Use PyTorch DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset

# Create a TensorDataset
dataset = TensorDataset(x_tensor, y_tensor)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through one batch of data
for batch_x, batch_y in data_loader:
    print("Batch X (features):", batch_x)
    print("Batch Y (targets):", batch_y)
    break  # Print only the first batch
```

## Running the Code

1. Ensure that the dataset file `winequality-red.csv` is located in the specified path.
2. Run the Python script to load the data, convert it to tensors, and print out batches of data using the DataLoader.

## License

This project is licensed under the MIT License.

