import numpy as np
from .estimate_k import scipy_optimise, binary_search, cluster_acc
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def get_clustering_metrics(labels, preds, old_mask):
    all_acc, ind, _ = cluster_acc(
        labels.astype(int), preds.astype(int), return_ind=True
    )

    cluster_mapping = {pair[0]: pair[1] for pair in ind}

    preds = np.array([cluster_mapping[c] for c in preds])

    all_nmi, all_ari = (
        nmi_score(labels, preds),
        ari_score(labels, preds),
    )

    old_preds = preds[old_mask]
    new_preds = preds[~old_mask]

    old_gt = labels[old_mask]
    new_gt = labels[~old_mask]

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
