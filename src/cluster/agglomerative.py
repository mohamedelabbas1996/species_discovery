from sklearn.cluster import AgglomerativeClustering
from .base_clusterer import BaseClusterer
from .clustering_metrics import get_clustering_metrics
from sklearn import preprocessing

from .preprocessing_features import standardize, dimension_reduction
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os


def get_distance_threshold(features, labels):
    distance_matrix = squareform(pdist(features))
    intra_cluster_distances = []
    inter_cluster_distances = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if labels[i] == labels[j]:
                intra_cluster_distances.append(distance_matrix[i, j])
            else:
                inter_cluster_distances.append(distance_matrix[i, j])
    # choose the 95th percentile of intra-cluster distances
    threshold = np.percentile(intra_cluster_distances, 95)
    return threshold


class AgglomerativeClusterer(BaseClusterer):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.data_dict = None
        self.distance_threshold = self.config.agglomerative.distance_threshold

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

            # doesn't work :/
            if not self.distance_threshold:
                self.distance_threshold = get_distance_threshold(
                    data_dict["val"]["feat_list"], data_dict["val"]["label_list"]
                )
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
        print("distance threshold: ", self.distance_threshold)
        clusterer = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold
        ).fit(features)

        preds = clusterer.labels_

        metrics = get_clustering_metrics(
            labels, preds, old_mask, self.config.split_cost, self.config.merge_cost
        )
        metrics["K"] = clusterer.n_clusters_
        metrics["distance_threshold"] = self.distance_threshold

        return metrics
