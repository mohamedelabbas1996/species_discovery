from .estimate_k import scipy_optimise, binary_search
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
from .base_clusterer import BaseClusterer

from .clustering_metrics import get_clustering_metrics
from .preprocessing_features import standardize, dimension_reduction


def acc(y_true, y_pred):
    return (y_true == y_pred).sum() / len(y_true)


class KMeansClusterer(BaseClusterer):
    def __init__(self, config):
        # self.config = config
        # self.setup_flag = False
        # self.k = config.k
        # self.data_dict = None
        self.config = config
        self.setup_flag = False
        self.k = None

    def setup(self, data_dict):
        # estimate k
        if not self.setup_flag:
            if self.config.search_mode.name == "brent":
                print("Optimising with Brents algorithm")
                k = scipy_optimise(data_dict, self.config)
            else:
                k = binary_search(data_dict, self.config)
            self.k = k
        else:
            pass

        # # estimate k
        # if not self.setup_flag:
        #     for data_type in data_dict:
        #         features = data_dict[data_type]
        #         data_dict[data_type] = dimension_reduction(standardize(features))

        #     self.data_dict = data_dict
        #     self.setup_flag = True

        #     if self.config.search_mode.name == "brent":
        #         print("Optimising with Brents algorithm")
        #         k = scipy_optimise(self.data_dict, self.config)
        #     else:
        #         k = binary_search(self.data_dict, self.config)
        #     self.k = k
        # else:
        #     pass

    def clustering(self, data_dict):
        self.setup(data_dict)
        features = data_dict["test"]["feat_list"]
        labels = data_dict["test"]["label_list"].astype(int)

        old_mask = labels < self.config.dataset.num_classes

        clusterer = MiniBatchKMeans(
            n_clusters=self.k, batch_size=1024, random_state=42
        ).fit(features)
        preds = clusterer.labels_

        metrics = get_clustering_metrics(labels, preds, old_mask)

        # add the estimated k
        metrics["K"] = self.k

        return metrics
