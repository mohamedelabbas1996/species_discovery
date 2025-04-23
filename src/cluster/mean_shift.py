from sklearn.cluster import MeanShift
from .base_clusterer import BaseClusterer
from .clustering_metrics import get_clustering_metrics
from sklearn import preprocessing

from .preprocessing_features import standardize, dimension_reduction
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os


class MeanShiftClusterer(BaseClusterer):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.data_dict = None
        self.bandwidth = self.config.mean_shift.bandwidth

    def setup(self, data_dict):
        # estimate the distance threshold
        new_data_dict = {}
        save_dir = self.config.output_dir
        if not self.setup_flag:
            for data_type in data_dict:
                new_data_dict[data_type] = {}
                features = data_dict[data_type]["feat_list"]
                features = dimension_reduction(
                    standardize(features), self.config.pca.n_components
                )
                labels = data_dict[data_type]["label_list"]
                new_data_dict[data_type]["feat_list"] = features
                new_data_dict[data_type]["label_list"] = labels

                np.savez(
                    os.path.join(
                        save_dir,
                        f"{data_type}_processed_pca_{self.config.pca.n_components}",
                    ),
                    feat_list=features,
                    label_list=labels,
                )
            self.data_dict = new_data_dict
            self.setup_flag = True
        else:
            pass

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
        print("bandwidth: ", self.bandwidth)
        clusterer = MeanShift(bandwidth=self.bandwidth).fit(features)

        preds = clusterer.labels_

        metrics = get_clustering_metrics(
            labels, preds, old_mask, self.config.split_cost, self.config.merge_cost
        )
        metrics["K"] = preds.max() + 1
        metrics["bandwidth"] = self.bandwidth

        return metrics
