from sklearn.cluster import AgglomerativeClustering
from .base_clusterer import BaseClusterer
from .clustering_metrics import get_clustering_metrics
from sklearn import preprocessing

from .preprocessing_features import standardize, dimension_reduction
from scipy.spatial.distance import pdist, squareform
import numpy as np


def get_distance_threshold(features, labels):
    # Compute pairwise distances
    distance_matrix = squareform(pdist(features))

    # Compute intra-cluster and inter-cluster distances
    intra_cluster_distances = []
    inter_cluster_distances = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if labels[i] == labels[j]:  # Same cluster
                intra_cluster_distances.append(distance_matrix[i, j])
            else:  # Different cluster
                inter_cluster_distances.append(distance_matrix[i, j])

    # Determine a reasonable threshold
    # A good threshold might be around the 95th percentile of intra-cluster distances
    threshold = np.percentile(intra_cluster_distances, 95)
    return threshold


class AgglomerativeClusterer(BaseClusterer):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.data_dict = None
        self.distance_threshold = None

    def setup(self, data_dict):
        # estimate the distance threshold
        new_data_dict = {}
        if not self.distance_threshold:
            for data_type in data_dict:
                new_data_dict[data_type] = {}
                features = data_dict[data_type]["feat_list"]
                features = dimension_reduction(standardize(features))
                new_data_dict[data_type]["feat_list"] = features
                new_data_dict[data_type]["label_list"] = data_dict[data_type][
                    "label_list"
                ]

            self.data_dict = new_data_dict
            self.setup_flag = True

            self.distance_threshold = get_distance_threshold(
                data_dict["val"]["feat_list"], data_dict["val"]["label_list"]
            )
        else:
            pass

    def clustering(self, data_dict):
        self.setup(data_dict)
        data_dict = self.data_dict
        features = data_dict["test"]["feat_list"]
        # normalize features

        if self.config.standardize:
            scaler = preprocessing.StandardScaler().fit(features)
            features = scaler.transform(features)

        labels = data_dict["test"]["label_list"].astype(int)

        old_mask = labels < self.config.dataset.num_classes

        print("distance threshold: ", self.distance_threshold)
        clusterer = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold
        ).fit(features)

        preds = clusterer.labels_

        metrics = get_clustering_metrics(labels, preds, old_mask)
        metrics["K"] = clusterer.n_clusters_
        metrics["distance_threshold"] = self.distance_threshold

        return metrics
