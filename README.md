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
  
  adult_income:
    type: url
    data_type: tabular
    filename: adult_income.csv
    description: Adult income classification dataset
    url: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

## Project Structure

```
GraphArchitect/
├── config/
│   └── config_dataset.yaml    # Dataset configuration
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Prétraitement des données brutes
│   ├── feature_extraction.py  # Extraction des caractéristiques (embeddings)
│   ├── graph_construction.py  # Méthodes de construction de graphes (k-NN, seuil, etc.)
│   ├── sparsification.py      # Algorithmes de filtrage et réduction de densité
│   ├── validation.py          # Évaluation structurelle et validation indirecte des graphes
│   └── utils.py               # Fonctions utilitaires (chargement/sauvegarde, visualisation, etc.)
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
