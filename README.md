# GraphArchitect

A comprehensive framework for working with various types of datasets and building machine learning models.

## Features

- Automated dataset downloading and management
- Support for multiple data types (tabular, image, audio, text, time series)
- Flexible configuration via YAML files
- Efficient caching to prevent redownloading

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
from data_manager import DatasetManager

# Initialize the manager
manager = DatasetManager()

# Download all configured datasets
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
  
  mnist:
    type: torchvision
    data_type: image
    filename: mnist.pt
    description: Handwritten digits dataset (28x28 grayscale images)
```

## Project Structure

```
GraphArchitect/
├── config/
│   └── config_dataset.yaml    # Dataset configuration
├── data_manager.py            # Dataset management functionality
├── main.py                    # Entry point for the application
├── requirements.txt           # Python dependencies
└── Data/                      # Downloaded datasets (gitignored)
    ├── tabular/               # Tabular datasets (CSV, etc.)
    ├── images/                # Image datasets
    ├── audio/                 # Audio datasets
    ├── text/                  # Text datasets
    └── time_series/           # Time series datasets
```

## License

[MIT License](LICENSE)
