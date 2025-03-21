# GraphArchitect

## Features

- Automated dataset downloading and management
- Support for multiple data types (tabular, image, audio, text, time series)
- Flexible configuration via YAML files
- Efficient caching to prevent redownloading
- Graph construction methods for various applications
- Semi-supervised classification support with label propagation

## Installation

```bash
# Clone the repository
git clone https://github.com/username/GraphArchitect.git
cd GraphArchitect

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.data_manager import DataManager

# Initialize the manager with a DataLoader instance
manager = DataManager(your_data_loader_instance)

# Download and load datasets
manager.load_datasets()
```

### Configuration

Datasets are configured in `config/config_dataset.yaml`. Example structure:

```yaml
datasets:
  iris:
    type: sklearn
    data_type: tabular
    filename: iris.csv
    description: The Iris dataset with flower measurements
  
  adult_income:
    type: url
    data_type: tabular
    filename: adult_income.csv
    description: Adult income classification dataset
    url: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

### Graph Construction

Graph construction methods are available for various applications. Example usage:

```python
from src.graph_construction import build_graph

# Example data
data = ...

# Build a KNN graph
graph = build_graph("knn", data, k=5)
```

### Semi-Supervised Classification and Label Propagation

GraphArchitect provides helper functions for semi-supervised learning. You can easily split your data into labeled and unlabeled subsets and prepare features, the graph adjacency matrix, and targets required for label propagation.

For example:

```python
from src.utils import split_semi_supervised, prepare_label_propagation_data

# Suppose X, y are your visual features and labels
X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_semi_supervised(X, y, labeled_fraction=0.1)

# Create a unified label vector, where -1 indicates unlabeled samples.
import numpy as np
num_samples = X.shape[0]
y_semi = np.full(num_samples, -1)
# Update y_semi for labeled indices
# (You can derive the indices from your splitting procedure)

# Prepare the data for label propagation:
features, graph_adj, targets = prepare_label_propagation_data(X, y_semi, graph_method="knn", k=5)
```

This will return:
- **features**: The input features.
- **graph_adj**: The adjacency matrix of the constructed graph.
- **targets**: The label vector used for propagation (with unlabeled samples marked as -1).

## Project Structure

```
GraphArchitect/
├── CHANGELOG.md
├── README.md
├── config
│   └── config_dataset.yaml
├── data
│   ├── image
│   ├── tabular
│   ├── text
│   └── time_series
├── main.py
├── requirements.txt
├── setup.py
└── src
    ├── data_loader.py
    ├── data_manager.py
    ├── graph_construction.py
    └── utils.py
```

## License

[MIT License](LICENSE)
