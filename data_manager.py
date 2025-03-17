import os
import yaml
import requests
import shutil
import tarfile
import zipfile

from sklearn.datasets import load_iris, load_wine
from torchvision import datasets, transforms

class DatasetManager:
    def __init__(self, config_path="config/config_dataset.yaml", data_root="Data"):
        """
        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
            data_root (str): Répertoire racine où stocker les données.
        """
        self.config_path = config_path
        self.data_root = data_root
        self.datasets_config = {}
        
        # Charger la config dès l'initialisation
        self.load_yml_config()
        
        # S'assure que le répertoire racine existe
        os.makedirs(self.data_root, exist_ok=True)

    def load_yml_config(self):
        """Charge le fichier YAML et stocke la config dans self.datasets_config."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.datasets_config = yaml.safe_load(f)

    def download_sklearn_dataset(self, dataset_name, dataset_info):
        """
        Télécharge/charge un dataset depuis sklearn, et l'enregistre au format CSV.
        """
        import pandas as pd
        
        data_type = dataset_info.get('data_type', 'tabular')
        filename = dataset_info.get('filename', f"{dataset_name}.csv")
        
        # Prépare le chemin de sauvegarde final
        save_dir = os.path.join(self.data_root, data_type)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)

        if os.path.exists(output_path):
            print(f"[INFO] Le fichier {output_path} existe déjà, skip le téléchargement.")
            return

        if dataset_name == "iris":
            dataset = load_iris(as_frame=True)
            df = dataset.frame  # df contiendra data + target
        elif dataset_name == "wine":
            dataset = load_wine(as_frame=True)
            df = dataset.frame
        else:
            raise ValueError(f"Dataset sklearn inconnu : {dataset_name}")

        # Enregistre le DataFrame au format CSV
        df.to_csv(output_path, index=False)
        print(f"[SKLEARN] {dataset_name} sauvegardé dans {output_path}.")

    def download_torchvision_dataset(self, dataset_name, dataset_info):
        """
        Télécharge/charge un dataset depuis torchvision et l'enregistre au format .pt ou similaire.
        """
        data_type = dataset_info.get('data_type', 'image')
        filename = dataset_info.get('filename', f"{dataset_name}.pt")
        
        # Dossier de destination
        save_dir = os.path.join(self.data_root, data_type)
        os.makedirs(save_dir, exist_ok=True)
        
        # Vérification si le dossier contenant le dataset existe déjà
        # On utilise un petit mapping entre dataset_name et dossier créé par TorchVision
        dataset_dir_map = {
            "mnist": "MNIST",
            "fashion_mnist": "FashionMNIST",
            "cifar10": "cifar-10-batches-py"
        }
        target_subdir = dataset_dir_map.get(dataset_name, dataset_name)
        existing_path = os.path.join(save_dir, target_subdir)
        
        if os.path.exists(existing_path):
            print(f"[INFO] Le dossier {existing_path} existe déjà, skip le téléchargement.")
            return
        
        # Tu peux personnaliser les transforms selon tes besoins
        transform = transforms.Compose([transforms.ToTensor()])
        
        if dataset_name == "mnist":
            # Télécharge MNIST via torchvision
            datasets.MNIST(root=save_dir, train=True, download=True, transform=transform)
            datasets.MNIST(root=save_dir, train=False, download=True, transform=transform)
            # Dans ce cas, torchvision crée déjà ses propres fichiers
            # Optionnellement, tu peux sauvegarder un fichier .pt personnalisé
            # Mais souvent, on laisse torchvision gérer l'arborescence (MNIST/raw, MNIST/processed)
            print(f"[TORCHVISION] MNIST téléchargé dans {save_dir}.")
        elif dataset_name == "fashion_mnist":
            datasets.FashionMNIST(root=save_dir, train=True, download=True, transform=transform)
            datasets.FashionMNIST(root=save_dir, train=False, download=True, transform=transform)
            print(f"[TORCHVISION] FashionMNIST téléchargé dans {save_dir}.")
        elif dataset_name == "cifar10":
            datasets.CIFAR10(root=save_dir, train=True, download=True, transform=transform)
            datasets.CIFAR10(root=save_dir, train=False, download=True, transform=transform)
            print(f"[TORCHVISION] CIFAR10 téléchargé dans {save_dir}.")
        else:
            raise ValueError(f"Dataset torchvision inconnu : {dataset_name}")

        # Si tu veux absolument un fichier unique .pt, tu peux envisager un dump
        # torch.save(obj, os.path.join(save_dir, filename))
        # Mais en pratique, on laisse souvent la structure par défaut de torchvision.

    def download_url_dataset(self, dataset_name, dataset_info):
        data_type = dataset_info.get('data_type', 'tabular')
        filename = dataset_info.get('filename', f"{dataset_name}")
        file_url = dataset_info.get('url')
        extract = dataset_info.get('extract', False)
    
        save_dir = os.path.join(self.data_root, data_type)
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, filename)
    
        # Vérification de l'existence
        if os.path.exists(file_path):
            print(f"[INFO] Le fichier {file_path} existe déjà, skip le téléchargement.")
            # Optionnel: si extract=True, tu peux aussi vérifier si l'extraction a déjà été faite.
            return
    
        print(f"[URL] Téléchargement de {dataset_name} depuis {file_url} ...")
        self._download_file(file_url, file_path)
        print(f"[URL] Fichier sauvegardé dans {file_path}.")
    
        if extract:
            extract_dir = os.path.join(save_dir, dataset_name)
            os.makedirs(extract_dir, exist_ok=True)
            self._extract_archive(file_path, extract_dir)
            print(f"[URL] Archive extraite dans {extract_dir}.")

    def _download_file(self, url, dest_path):
        """Télécharge un fichier depuis une URL et l’enregistre sur le disque."""
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def _extract_archive(self, archive_path, extract_dir):
        """Extrait une archive (tar, gz, zip...) dans un dossier donné."""
        # Si c’est un tar.* ou .gz
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=extract_dir)
        # Si c’est un ZIP
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(path=extract_dir)
        else:
            print(f"[WARN] Format d’archive non reconnu pour {archive_path}. Aucune extraction effectuée.")

    def load_datasets(self):
        """
        Parcourt la configuration YAML et lance le téléchargement/le chargement
        pour chaque dataset.
        """
        for dataset_name, dataset_info in self.datasets_config.get('datasets', {}).items():
            data_type = dataset_info.get('type', '')
            
            if data_type == 'sklearn':
                self.download_sklearn_dataset(dataset_name, dataset_info)
            elif data_type == 'torchvision':
                self.download_torchvision_dataset(dataset_name, dataset_info)
            elif data_type == 'url':
                self.download_url_dataset(dataset_name, dataset_info)
            else:
                print(f"[INFO] Type de dataset non pris en charge ou inconnu : {data_type}")