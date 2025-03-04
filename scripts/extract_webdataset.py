import webdataset as wds
import os
import numpy as np
import json
import pandas as pd


def get_statistics(speciesKey):
    with open(
        "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/03_ami-gbif_fine-grained_c-america_category_map.json"
    ) as f:
        c_america_map = json.load(f)

    id_speciesKey = np.array([int(sk) for sk in c_america_map.keys()])

    speciesKey = np.array(speciesKey)
    # speciesKey = speciesKey[speciesKey != -1]
    unique, counts = np.unique(speciesKey, return_counts=True)

    trap_df = pd.DataFrame(zip(unique, counts), columns=["speciesKey", "count"])

    trap_df.to_csv("data/ami_trap_statistics.csv", index=False)
    id_mask = np.isin(speciesKey, id_speciesKey)

    df = pd.read_csv("/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/03_c-america_test_ood_local.csv")
    ood_speciesKey = np.array([int(sk) for sk in df.speciesKey.unique()])
    ood_mask = np.isin(speciesKey, ood_speciesKey)

    print(
        f"Total images: {len(speciesKey)}\n",
        f"Labeled images: {len(speciesKey[speciesKey != -1])}\n",
        f"ID labeled images: {speciesKey[id_mask].size}\n",
        f"Total species: {len(np.unique(speciesKey)) -1}\n",  # remove -1
        f"ID species: {len(np.unique(speciesKey[id_mask]))}\n",
        f"OOD species: {len(np.unique(speciesKey[ood_mask]))}\n",
        f"Merged species: {len(set(speciesKey) | set(id_speciesKey) | set(ood_speciesKey))}",
    )


def to_txt(df, filename):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            line = f"{row['image_path']} {row['label']}\n"
            f.write(line)


def create_csv():
    """
    Create a CSV to store the metadata of the AMI-trap dataset
    """
    data_dir = "/network/scratch/y/yuyan.chen/ami_trap/crops"
    filenames = set([fn.split(".")[0] for fn in os.listdir(data_dir)])
    img_paths, speciesKey = [], []
    for fn in filenames:
        img_paths.append(fn + ".png")
        with open(f"{data_dir}/{fn}.json") as f:
            annotation = json.load(f)

        if annotation["speciesKey"]:
            speciesKey.append(annotation["speciesKey"])
        else:
            speciesKey.append(-1)

    trap_df = pd.DataFrame(zip(img_paths, speciesKey), columns=["image_path", "speciesKey"])
    trap_df.to_csv("data/ami_trap.csv", index=False)


def create_txt():

    trap_df = pd.read_csv("data/ami_trap.csv")

    trap_test = trap_df[trap_df["speciesKey"] != -1]  # we need labels for testing so that we can compare ACC
    trap_train = trap_df[trap_df["speciesKey"] == -1]

    with open(
        "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/03_ami-gbif_fine-grained_c-america_category_map.json"
    ) as f:
        id_map = json.load(f)

    species_all = set(trap_test.speciesKey)
    species_id = set([int(sk) for sk in id_map.keys()])
    species_ood = sorted(list(species_all - species_id))

    ood_map = {str(categ): id + len(id_map.keys()) for id, categ in enumerate(species_ood)}

    taxon_map = id_map | ood_map
    assert len(taxon_map.keys()) == 1117  # total number of species: 636 ID and 481 OOD

    with open("data/clustering_taxon_map.json", "w") as f:
        json.dump(taxon_map, f)

    trap_test["label"] = [taxon_map[str(sk)] for sk in trap_test["speciesKey"]]
    trap_train["label"] = [-1 for _ in range(len(trap_train))]

    to_txt(trap_test, "data/ami_trap_test_unlabeled.txt")
    to_txt(trap_train, "data/ami_trap_train_unlabeled.txt")


create_txt()
