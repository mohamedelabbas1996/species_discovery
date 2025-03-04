from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from .estimate_k import scipy_optimise, binary_search, cluster_acc
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
from .base_clusterer import BaseClusterer


class KMeansClusterer(BaseClusterer):
    def __init__(self, config):
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

    def clustering(self, data_dict):
        self.setup(data_dict)
        save_dir = self.config.output_dir

        features = data_dict["test"]["feat_list"]
        labels = data_dict["test"]["label_list"].astype(int)

        old_mask = labels >= self.config.dataset.num_classes

        kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=1024, random_state=42).fit(features)
        preds = kmeans.labels_
        old_preds = preds[old_mask]
        new_preds = preds[~old_mask]  # save the prediction

        # TODO: save the predicted "label"
        old_gt = labels[old_mask]
        new_gt = labels[~old_mask]

        all_acc, all_nmi, all_ari = (
            cluster_acc(labels.astype(int), preds.astype(int)),
            nmi_score(labels, preds),
            ari_score(labels, preds),
        )

        old_acc, old_nmi, old_ari = (
            cluster_acc(old_gt.astype(int), old_preds.astype(int)),
            nmi_score(old_gt, old_preds),
            ari_score(old_gt, old_preds),
        )

        new_acc, new_nmi, new_ari = (
            cluster_acc(new_gt.astype(int), new_preds.astype(int)),
            nmi_score(new_gt, new_preds),
            ari_score(new_gt, new_preds),
        )

        metrics = {
            "ACC_all": all_acc,
            "NMI_all": all_nmi,
            "ARI_all": all_ari,
            "ACC_old": old_acc,
            "NMI_old": old_nmi,
            "ARI_old": old_ari,
            "ACC_new": new_acc,
            "NMI_new": new_nmi,
            "ARI_new": new_ari,
        }

        return metrics
