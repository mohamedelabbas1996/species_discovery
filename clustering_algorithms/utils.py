from .kmeans import KMeansClusterer
from .agglomerative import AgglomerativeClusterer
from .mean_shift import MeanShiftClusterer
from .dbscan import DBSCANClusterer


def get_clusterer(config):
    clusterers = {
        "kmeans": KMeansClusterer,
        "agglomerative": AgglomerativeClusterer,
        "mean_shift": MeanShiftClusterer,
        "dbscan": DBSCANClusterer,
    }

    return clusterers[config.cluster.name](config)
