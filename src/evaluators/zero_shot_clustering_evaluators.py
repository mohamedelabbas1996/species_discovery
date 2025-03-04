from .base_evaluator import BaseEvaluator
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import numpy as np
from sklearn.cluster import KMeans
import os

# from ..cluster.estimate_k import scipy_optimise, binary_search, cluster_acc
from ..cluster import get_clusterer


class ZeroShotClusteringEvaluator(BaseEvaluator):
    def __init__(self, config):
        super(ZeroShotClusteringEvaluator, self).__init__(config)
        self.config = config

    def eval_clustering(self, net, dataloader_dict):
        val_dataloader, test_dataloader = dataloader_dict["val"], dataloader_dict["test"]
        self.extract(net, val_dataloader, filename="val")
        self.extract(net, test_dataloader, filename="test")

        save_dir = self.config.output_dir

        data_dict = {}

        for key in ["val", "test"]:
            data_dict[key] = np.load(os.path.join(save_dir, f"{key}.npz"))

        clusterer = get_clusterer(self.config)
        metrics = clusterer.clustering(data_dict)

        return metrics

        # # clustering
        # # estimate k
        # if self.config.search_mode.name == "brent":
        #     print("Optimising with Brents algorithm")
        #     k = scipy_optimise(data_dict, self.config)
        # else:
        #     k = binary_search(data_dict, self.config)

        # # run k means
        # all_feats = np.concatenate([data_dict["labeled"]["feat_list"], data_dict["unlabeled"]["feat_list"]])
        # labeled_mask = np.array(
        #     [True for _ in range(len(data_dict["labeled"]["feat_list"]))]
        #     + [False for _ in range(len(data_dict["unlabeled"]["feat_list"]))]
        # )

        # kmeans = KMeans(n_clusters=k, random_state=0).fit(all_feats)
        # labeled_preds = kmeans.labels_[labeled_mask]
        # unlabeled_preds = kmeans.labels_[~labeled_mask]  # save the prediction

        # print(save_dir)
        # np.save(os.path.join(save_dir, "unlabeled_cluster_pred.npy"), unlabeled_preds)
        # labelled_acc, labelled_nmi, labelled_ari = (
        #     cluster_acc(data_dict["labeled"]["label_list"].astype(int), labeled_preds.astype(int)),
        #     nmi_score(data_dict["labeled"]["label_list"], labeled_preds),
        #     ari_score(data_dict["labeled"]["label_list"], labeled_preds),
        # )

        # metrics = {"ACC": labelled_acc, "NMI": labelled_nmi, "ARI": labelled_ari}

        # return metrics
