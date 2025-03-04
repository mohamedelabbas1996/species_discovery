from src.utils import Config
from .kmeans import KMeansClusterer


def get_clusterer(config):
    clusterers = {
        "kmeans": KMeansClusterer,
    }

    return clusterers[config.cluster.name](config)
