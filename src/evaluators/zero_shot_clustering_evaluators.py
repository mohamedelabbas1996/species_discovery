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
        net.eval()
        val_dataloader, test_dataloader = (
            dataloader_dict["val"],
            dataloader_dict["test"],
        )
        self.extract(net, val_dataloader, filename="val")
        self.extract(net, test_dataloader, filename="test")

        save_dir = self.config.output_dir

        data_dict = {}

        for key in ["val", "test"]:
            data_dict[key] = np.load(os.path.join(save_dir, f"{key}.npz"))

        clusterer = get_clusterer(self.config)
        metrics = clusterer.clustering(data_dict)

        return metrics
