import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import librosa
import pickle

# Define constants for CIFAR-10
CIFAR_BATCH_SIZE = 10000
input_shape = (32, 32, 3)

def load_CIFAR_batch(file_path):
    """
    Load a single batch of CIFAR-10 images from the binary file and return as a NumPy array.
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")
        # Extract data and labels from the dictionary.
        X = data_dict["data"]
        y = data_dict["labels"]
        # Reshape from (10000, 3072) to (10000, 3, 32, 32) then transpose to (10000, 32, 32, 3)
        X = X.reshape(CIFAR_BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(y)
        return X, y

class DataManager:
    def __init__(self, data_loader):
        """
        Args:
            data_loader (DataLoader): An instance of DataLoader containing configuration and data paths.
        """
        self.data_loader = data_loader
        self.datasets_config = data_loader.datasets_config
        self.data_root = data_loader.data_root

    def load_data_for_ml(self, dataset_name: str):
        """
        Loads data from the specified dataset (as configured in config_dataset.yaml) 
        in a format compatible with scikit-learn.
        
        Supports:
          - 'tabular': CSV file with a 'target' column.
          - 'image': Images from directories (subdirectory names as labels).
          - 'audio': .wav files with MFCC feature extraction.
          - 'time_series': CSV file with a 'target' column.
          - 'text': .txt files from directories (subdirectory names as labels).
        
        Args:
            dataset_name (str): The key of the dataset in the configuration.
            
        Returns:
            X, y: Features and labels suitable for machine learning.
        """
        dataset_info = self.datasets_config.get("datasets", {}).get(dataset_name, {})
        data_type = dataset_info.get("data_type", "tabular")
        filename = dataset_info.get("filename", f"{dataset_name}")
        save_dir = os.path.join(self.data_root, data_type)
        file_path = os.path.join(save_dir, filename)
        
        if data_type == "tabular":
            return self._load_tabular_data(file_path, dataset_name)
        elif data_type == "image":
            return self._load_image_data(save_dir, dataset_name)
        elif data_type == "audio":
            return self._load_audio_data(save_dir, dataset_name)
        elif data_type == "time_series":
            return self._load_time_series_data(file_path, dataset_name)
        elif data_type == "text":
            return self._load_text_data(save_dir, dataset_name)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _load_tabular_data(self, file_path: str, dataset_name: str):
        """Load tabular data from a CSV file. Expects a 'target' column."""
        df = pd.read_csv(file_path)
        if "target" not in df.columns:
            target_col = self.datasets_config["datasets"][dataset_name]["target"]
            if target_col in df.columns:
                df = df.rename(columns={target_col: "target"})
            else:
                raise ValueError(f"Target column '{target_col}' not found in {file_path}")
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        return X, y
    
    def _load_image_data(self, directory: str, dataset_name: str):
        """Load image data from a directory.
        
        For 'cifar10', it loads data from native binary batch files.
        For 'mnist' or 'fashion_mnist', it uses idx2numpy to convert IDX files to numpy arrays.
        """
        if dataset_name == "cifar10":
            # ...existing CIFAR-10 loading code...
            cifar_dir = os.path.join(directory, "cifar-10-batches-py")
            X_batches = []
            y_batches = []
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
            for batch in batch_files:
                batch_path = os.path.join(cifar_dir, batch)
                X_batch, y_batch = load_CIFAR_batch(batch_path)
                X_batches.append(X_batch)
                y_batches.append(y_batch)
            X = np.concatenate(X_batches, axis=0)
            y = np.concatenate(y_batches, axis=0)
            return X, y
        elif dataset_name in ("mnist", "fashion_mnist"):
            # Use idx2numpy to load idx files.
            try:
                import idx2numpy
            except ImportError:
                raise ImportError("Please install idx2numpy: pip install idx2numpy")
            # Set file paths based on dataset_name.
            if dataset_name == "mnist":
                images_path = os.path.join(directory, "MNIST", "raw", "t10k-images-idx3-ubyte")
                labels_path = os.path.join(directory, "MNIST", "raw", "t10k-labels-idx1-ubyte")
            else:  # "fashion_mnist"
                images_path = os.path.join(directory, "FashionMNIST", "raw", "train-images-idx3-ubyte")
                labels_path = os.path.join(directory, "FashionMNIST", "raw", "train-labels-idx1-ubyte")
            X = idx2numpy.convert_from_file(images_path)
            y = idx2numpy.convert_from_file(labels_path)
            return X, y
        else:
            # Fallback: load common image formats (e.g., PNG, JPG)
            X, y, labels = [], [], []
            from sklearn.preprocessing import LabelEncoder
            from PIL import Image
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert("RGB")
                        X.append(np.array(img))
                        label = os.path.basename(root)
                        labels.append(label)
            X = np.array(X)
            y = LabelEncoder().fit_transform(labels) if labels else np.array([])
            return X, y
    
    def _load_audio_data(self, directory: str, dataset_name: str):
        """Load audio data from a directory and extract MFCC features."""
        X, y, labels = [], [], []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".wav"):
                    audio_path = os.path.join(root, file)
                    audio, sr = librosa.load(audio_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    X.append(np.mean(mfcc.T, axis=0))
                    label = os.path.basename(root)
                    labels.append(label)
        X = np.array(X)
        y = LabelEncoder().fit_transform(labels) if labels else np.array([])
        return X, y
    
    def _load_time_series_data(self, file_path: str, dataset_name: str):
        """Load time series data from a CSV file. Expects a 'target' column."""
        df = pd.read_csv(file_path)
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        return X, y
    
    def _load_text_data(self, directory: str, dataset_name: str):
        """Load text data from a directory. Expects .txt files in subdirectories."""
        X, y, labels = [], [], []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".txt"):
                    text_path = os.path.join(root, file)
                    with open(text_path, "r", encoding="utf-8") as f:
                        X.append(f.read())
                    label = os.path.basename(root)
                    labels.append(label)
        y = LabelEncoder().fit_transform(labels) if labels else np.array([])
        return X, y
