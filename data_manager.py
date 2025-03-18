import os
import yaml
import requests
import tarfile
import zipfile

from sklearn.datasets import load_iris, load_wine
from torchvision import datasets, transforms


class DatasetManager:
    def __init__(self, config_path="config/config_dataset.yaml", data_root="Data"):
        """
        Args:
            config_path (str): Path to the YAML configuration file.
            data_root (str): Root directory where the data will be stored.
        """
        self.config_path = config_path
        self.data_root = data_root
        self.datasets_config = {}

        # Load the configuration during initialization
        self.load_yml_config()

        # Ensure that the root data directory exists
        os.makedirs(self.data_root, exist_ok=True)

    def load_yml_config(self):
        """Loads the YAML configuration file and stores the config in self.datasets_config."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.datasets_config = yaml.safe_load(f)

    def download_sklearn_dataset(self, dataset_name, dataset_info):
        """
        Downloads/loads a dataset from sklearn and saves it as a CSV file.
        """

        data_type = dataset_info.get("data_type", "tabular")
        filename = dataset_info.get("filename", f"{dataset_name}.csv")

        # Prepare the final save directory
        save_dir = os.path.join(self.data_root, data_type)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)

        if os.path.exists(output_path):
            print(f"[INFO] File {output_path} already exists. Skipping download.")
            return

        if dataset_name == "iris":
            dataset = load_iris(as_frame=True)
            df = dataset.frame  # df contains both data and target
        elif dataset_name == "wine":
            dataset = load_wine(as_frame=True)
            df = dataset.frame
        else:
            raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

        # Save the DataFrame as a CSV file
        df.to_csv(output_path, index=False)
        print(f"[SKLEARN] {dataset_name} saved to {output_path}.")

    def download_torchvision_dataset(self, dataset_name, dataset_info):
        """
        Downloads/loads a dataset from torchvision and saves it as a .pt file or similar.
        """
        data_type = dataset_info.get("data_type", "image")

        # Destination folder
        save_dir = os.path.join(self.data_root, data_type)
        os.makedirs(save_dir, exist_ok=True)

        # Check if the folder containing the dataset already exists.
        # Here, we use a mapping between dataset_name and the folder created by TorchVision.
        dataset_dir_map = {
            "mnist": "MNIST",
            "fashion_mnist": "FashionMNIST",
            "cifar10": "cifar-10-batches-py",
        }
        target_subdir = dataset_dir_map.get(dataset_name, dataset_name)
        existing_path = os.path.join(save_dir, target_subdir)

        if os.path.exists(existing_path):
            print(f"[INFO] Folder {existing_path} already exists. Skipping download.")
            return

        # You can customize transforms as needed
        transform = transforms.Compose([transforms.ToTensor()])

        if dataset_name == "mnist":
            # Download MNIST via torchvision
            datasets.MNIST(
                root=save_dir, train=True, download=True, transform=transform
            )
            datasets.MNIST(
                root=save_dir, train=False, download=True, transform=transform
            )
            print(f"[TORCHVISION] MNIST downloaded in {save_dir}.")
        elif dataset_name == "fashion_mnist":
            datasets.FashionMNIST(
                root=save_dir, train=True, download=True, transform=transform
            )
            datasets.FashionMNIST(
                root=save_dir, train=False, download=True, transform=transform
            )
            print(f"[TORCHVISION] FashionMNIST downloaded in {save_dir}.")
        elif dataset_name == "cifar10":
            datasets.CIFAR10(
                root=save_dir, train=True, download=True, transform=transform
            )
            datasets.CIFAR10(
                root=save_dir, train=False, download=True, transform=transform
            )
            print(f"[TORCHVISION] CIFAR10 downloaded in {save_dir}.")
        else:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name}")

        # If you need a single .pt file, you could consider dumping the dataset
        # torch.save(obj, os.path.join(save_dir, filename))
        # But in practice, we often leave torchvision to manage its default structure.

    def download_url_dataset(self, dataset_name, dataset_info):
        data_type = dataset_info.get("data_type", "tabular")
        filename = dataset_info.get("filename", f"{dataset_name}")
        file_url = dataset_info.get("url")
        extract = dataset_info.get("extract", False)

        save_dir = os.path.join(self.data_root, data_type)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, filename)

        # Check for existence
        if os.path.exists(file_path):
            print(f"[INFO] File {file_path} already exists. Skipping download.")
            # Optionally, if extract=True, you could also check if extraction has already been done.
            return

        print(f"[URL] Downloading {dataset_name} from {file_url} ...")
        self._download_file(file_url, file_path)
        print(f"[URL] File saved to {file_path}.")

        if extract:
            extract_dir = os.path.join(save_dir, dataset_name)
            os.makedirs(extract_dir, exist_ok=True)
            self._extract_archive(file_path, extract_dir)
            print(f"[URL] Archive extracted to {extract_dir}.")

    def _download_file(self, url, dest_path):
        """Downloads a file from a URL and saves it to disk."""
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def _extract_archive(self, archive_path, extract_dir):
        """Extracts an archive (tar, gz, zip, etc.) to a given folder."""
        # If it is a tar.* or .gz
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_dir)
        # If it is a ZIP
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(path=extract_dir)
        else:
            print(
                f"[WARN] Unrecognized archive format for {archive_path}. No extraction performed."
            )

    def load_datasets(self):
        """
        Iterates over the YAML configuration and downloads/loads each dataset.
        """
        for dataset_name, dataset_info in self.datasets_config.get(
            "datasets", {}
        ).items():
            dataset_type = dataset_info.get("type", "")

            if dataset_type == "sklearn":
                self.download_sklearn_dataset(dataset_name, dataset_info)
            elif dataset_type == "torchvision":
                self.download_torchvision_dataset(dataset_name, dataset_info)
            elif dataset_type == "url":
                self.download_url_dataset(dataset_name, dataset_info)
            else:
                print(f"[INFO] Unsupported or unknown dataset type: {dataset_type}")
