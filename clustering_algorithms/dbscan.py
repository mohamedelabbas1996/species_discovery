from .base_clusterer import BaseClusterer
from sklearn.cluster import DBSCAN
from .clustering_metrics import get_clustering_metrics
import os
import numpy as np


class DBSCANClusterer(BaseClusterer):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.data_dict = None
        self.eps = self.config.dbscan.eps
        self.min_samples = self.config.dbscan.min_samples

    def clustering(self, data_dict):
        save_dir = self.config.output_dir
        filename = f"test_processed_pca_{self.config.pca.n_components}.npz"

        if not os.path.exists(os.path.join(save_dir, filename)):
            self.setup(data_dict)

        for key in ["val", "test"]:
            data_dict[key] = np.load(os.path.join(save_dir, filename))

        features = data_dict["test"]["feat_list"]
        labels = data_dict["test"]["label_list"].astype(int)

        test_data_dict = np.load(os.path.join(save_dir, filename))

        features, labels = test_data_dict["feat_list"], test_data_dict["label_list"]

        old_mask = labels < self.config.dataset.num_classes

        print(f"eps: {self.eps} min_samples: {self.min_samples}")

        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(features)

        preds = clusterer.labels_

        metrics = get_clustering_metrics(
            labels, preds, old_mask, self.config.split_cost, self.config.merge_cost
        )
        metrics["K"] = preds.max() + 1

        return metrics
