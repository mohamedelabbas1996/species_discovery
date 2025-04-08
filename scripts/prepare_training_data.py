import pandas as pd
import json
import numpy as np
from collections import Counter

df_train = pd.read_csv(
    "/network/scratch/y/yuyan.chen/species_discovery/metadata/classification/panama_train.csv"
)
train_species = df_train.speciesKey.unique()
taxon_df = pd.read_csv(
    "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/taxonomy_map.csv"
)
taxon_df = taxon_df[taxon_df["speciesKey"].isin(train_species)]
species = taxon_df[["speciesKey", "species"]].drop_duplicates()
species = species.sort_values(by="speciesKey")
# categories_map = {
#     row.species: idx for idx, row in enumerate(species.itertuples(index=False))
# }

# print(categories_map)
# output_file = "panama_plus_category_map-with_names.json"

# # Save dictionary as JSON
# with open(output_file, "w") as f:
#     json.dump(categories_map, f, indent=4)

categories = sorted(list(df_train["speciesKey"].unique()))
categories_map = {categ: id for id, categ in enumerate(categories)}

df_train["label"] = df_train["speciesKey"].map(categories_map)

# num_classes = len(df_train.label.unique())
# cls_idx = np.array(df_train["label"]).astype(int)
# label_stat = Counter(cls_idx)
# cls_num = [-1 for _ in range(num_classes)]
# for i in range(num_classes):
#     cat_num = int(label_stat[i])
#     cls_num[i] = cat_num
# targets = cls_num / np.sum(cls_num)
# Get class counts
cls_idx = df_train["label"].astype(int).values
num_classes = df_train["label"].nunique()

# Count occurrences using numpy.bincount (much faster than Counter for integer labels)
cls_num = np.bincount(cls_idx, minlength=num_classes)

# Compute class distribution (normalized)
targets = cls_num / cls_num.sum()

print(targets)
