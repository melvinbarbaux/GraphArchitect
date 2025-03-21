from src.data_loader import DataLoader
from src.data_manager import DataManager
from src.graph_construction import build_graph
from src.utils import split_semi_supervised
from scr.utils import prepare_label_propagation_data
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def main():
    loader = DataLoader()
    loader.load_datasets()
    manager = DataManager(loader)

    for dataset_name in loader.datasets_config["datasets"].keys():
        X, y = manager.load_data_for_ml(dataset_name)

        if X.shape[0] > 11000:
            X = X[:10000]
            y = y[:10000]
        print(f"Loaded dataset {dataset_name} with {X.shape[0]} samples.")

        X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_semi_supervised(X, y, labeled_fraction=0.1)

        num_nodes = X.shape[0]
        y_semi = np.full(shape=(num_nodes,), fill_value=-1)
        
        labeled_indices = np.isin(X.tolist(), X_labeled.tolist()).nonzero()[0]
        y_semi[labeled_indices] = y_labeled

        lb = LabelBinarizer()
        lb.fit(y)  # Fit sur tous les labels possibles
        targets = np.zeros((num_nodes, len(lb.classes_)))
        labeled_mask = y_semi != -1
        targets[labeled_mask] = lb.transform(y_semi[labeled_mask])

        features, graph_adj, _ = prepare_label_propagation_data(X, y_semi, graph_method="knn", k=5)

if __name__ == "__main__":
    main()