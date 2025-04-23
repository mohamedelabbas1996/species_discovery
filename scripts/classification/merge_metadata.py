import pandas as pd
import json

df_panama = pd.read_csv(
    "/home/mila/y/yuyan.chen/projects/species_discovery/scripts/classification/Panama Plus Species List (extended category map) - panama_plus_species.csv"
)

df_global = pd.read_csv(
    "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/global_moth_taxonomy_map.csv"
)


with open("panama_plus_category_map-with_names.json", "r") as file:
    label_map = json.load(file)

df_panama = df_panama.drop(columns=["gbif_name", "genus", "family"])

# print(label_map)
df_panama = df_panama.merge(
    df_global, left_on="gbif_key", right_on="speciesKey", how="left"
)
df_panama = df_panama.rename(columns={"species": "gbif_name"})


df_panama["category_map_index"] = df_panama["gbif_name"].map(label_map)
df_panama.to_csv("scripts/classification/complete_panama_list.csv", index=False)

print(len(df_panama.category_map_index.unique()))
