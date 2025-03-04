from .base_evaluator import BaseEvaluator
from .zero_shot_clustering_evaluators import ZeroShotClusteringEvaluator
from src.utils import Config


def get_evaluator(config: Config):
    evaluators = {
        "base": BaseEvaluator,
        "zero_shot_clustering": ZeroShotClusteringEvaluator,
    }
    return evaluators[config.evaluator.name](config)
