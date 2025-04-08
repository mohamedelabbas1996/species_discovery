from src.utils import Config
from .kmeans import KMeansClusterer
from .agglomerative import AgglomerativeClusterer


def get_clusterer(config):
    clusterers = {"kmeans": KMeansClusterer, "agglomerative": AgglomerativeClusterer}

    return clusterers[config.cluster.name](config)
