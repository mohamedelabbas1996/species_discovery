# Adapted from generalized category discovery


from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.cluster import KMeans

from tqdm import tqdm

from scipy.optimize import minimize_scalar
from functools import partial
from scipy.optimize import linear_sum_assignment as linear_assignment

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def pairwise_cost(y_true, y_pred, split_cost=1, merge_cost=2):
    true_match = y_true[:, None] == y_true
    pred_match = y_pred[:, None] == y_pred

    split = true_match & ~pred_match  # true labels same, but cluster labels different
    merge = ~true_match & pred_match  # true labels different, but cluster labels same

    cost = np.sum(np.triu(split * split_cost | merge * merge_cost, k=1))

    return cost


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def test_kmeans(k, data_dict, config, metric):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    all_feats = np.concatenate(
        [data_dict["val"]["feat_list"], data_dict["test"]["feat_list"]]
    )
    labeled_mask = np.array(
        [True for _ in range(len(data_dict["val"]["feat_list"]))]
        + [False for _ in range(len(data_dict["test"]["feat_list"]))]
    )

    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_feats)
    preds = kmeans.labels_[labeled_mask]

    split_cost, merge_cost = config.split_cost, config.merge_cost

    if metric == "acc":
        cost = cluster_acc(
            data_dict["val"]["label_list"].astype(int),
            preds.astype(int),
        )
    if metric == "pairwise":
        cost = -pairwise_cost(
            data_dict["val"]["label_list"].astype(int),
            preds.astype(int),
            split_cost,
            merge_cost,
        )

    return cost


def test_kmeans_for_scipy(k, data_dict, config, metric):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    k = int(k)

    all_feats = np.concatenate(
        [data_dict["val"]["feat_list"], data_dict["test"]["feat_list"]]
    )

    labeled_mask = np.array(
        [True for _ in range(len(data_dict["val"]["feat_list"]))]
        + [False for _ in range(len(data_dict["test"]["feat_list"]))]
    )
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_feats)
    preds = kmeans.labels_[labeled_mask]

    split_cost, merge_cost = config.split_cost, config.merge_cost

    if metric == "acc":
        cost = -cluster_acc(
            data_dict["val"]["label_list"].astype(int), preds.astype(int)
        )
    if metric == "pairwise":
        cost = pairwise_cost(
            data_dict["val"]["label_list"].astype(int),
            preds.astype(int),
            split_cost,
            merge_cost,
        )

    return cost


def binary_search(data_dict, config):
    min_classes = config.dataset.num_classes

    # Iter 0
    big_k = config.dataset.max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    metric = config.metric

    labelled_acc_big = test_kmeans(big_k, data_dict, config, metric)
    labelled_acc_small = test_kmeans(small_k, data_dict, config, metric)
    labelled_acc_middle = test_kmeans(middle_k, data_dict, config, metric)

    print(
        f"Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} "
    )
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f"Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}")

    for i in range(1, int(np.log2(diff)) + 1):

        if labelled_acc_big > labelled_acc_small:

            best_acc = max(labelled_acc_middle, labelled_acc_big)

            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)

        else:

            best_acc = max(labelled_acc_middle, labelled_acc_small)
            big_k = middle_k

            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = test_kmeans(middle_k, data_dict, config, metric)

        print(
            f"Iter {i}: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} "
        )
        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f"Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}")

    return best_acc_at_k


def scipy_optimise(data_dict, config):

    small_k = config.dataset.num_classes
    big_k = config.dataset.max_classes

    metric = config.metric

    test_k_means_partial = partial(
        test_kmeans_for_scipy, data_dict=data_dict, config=config, metric=metric
    )
    res = minimize_scalar(
        test_k_means_partial,
        bounds=(small_k, big_k),
        method="bounded",
        options={"disp": True},
    )
    k = int(res.x)
    print(f"Optimal K is {k}")

    return k
