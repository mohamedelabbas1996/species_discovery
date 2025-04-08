from .preprocessing_features import standardize, dimension_reduction
from copy import deepcopy


class BaseClusterer:
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.data_dict = None

    def setup(self, data_dict):
        new_data_dict = {}
        if not self.setup_flag:
            for data_type in data_dict:
                new_data_dict[data_type] = {}
                features = data_dict[data_type]["feat_list"]
                features = dimension_reduction(standardize(features))
                new_data_dict[data_type]["feat_list"] = features
                new_data_dict[data_type]["label_list"] = data_dict[data_type][
                    "label_list"
                ]

            self.data_dict = new_data_dict
            self.setup_flag = True
        else:
            pass

    def clustering(self, data_dict):
        pass
