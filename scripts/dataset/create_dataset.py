import pandas as pd
import random
import itertools
import json
import os

random.seed(42)


class PanamaPlusIncrementalGCDDataset:
    df_id = pd.read_csv(
        "/network/scratch/y/yuyan.chen/species_discovery/metadata/classification/panama_train.csv"
    )
    # df_id = df_id.merge(df_global[["species", "genus"]])
    df_global = pd.read_csv(
        "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/global_moths_with_names.csv"
    )

    metadata_dir = (
        "/network/scratch/y/yuyan.chen/species_discovery/metadata/incremental_gcd/"
    )

    def get_nearby_species(self, range):
        df_in = self.df_id
        df_global = self.df_global

        lat_min = range["lat_min"] - 3
        lat_max = range["lat_max"] + 3
        lng_min = range["lng_min"] - 10
        lng_max = range["lng_max"] + 10
        in_range = (
            (df_global["decimalLatitude"] > lat_min)
            & (df_global["decimalLatitude"] < lat_max)
            & (df_global["decimalLongitude"] > lng_min)
            & (df_global["decimalLongitude"] < lng_max)
        )
        species_all = set(df_global[in_range].speciesKey)
        print("num of all species: ", len(species_all))
        species_in = set(df_in.speciesKey.unique())
        print("num of id species: ", len(species_in))
        species_out = species_all - species_in
        df_out = df_global[df_global["speciesKey"].isin(species_out)]

        print("num of ood species: ", len(species_out))
        print("num of total ood images: ", len(df_out))

        return df_out

    def sample_speices(self, region_range):
        def sample(df):
            return df.groupby("speciesKey", group_keys=False).apply(
                lambda x: x.sample(n=100, random_state=42) if len(x) > 100 else x
            )

        df = self.get_nearby_species(region_range)
        species = list(df.speciesKey.unique())
        unknown_species_list = []
        for i in range(10):
            unknown_species = random.sample(species, 200)
            df_unk = df[df["speciesKey"].isin(unknown_species)]
            df_test = sample(df_unk)
            df_train = df_unk.drop(df_test.index)
            df_test = df_test.astype({"speciesKey": "int64"})
            self.save_metadata(df_train, df_test, i)
            unknown_species_list.append(set(unknown_species))

            print(f"Day {i} & {len(df_train)} & {len(df_test)} \\\\")

        # set_pair = list(itertools.combinations(unknown_species_list, 2))
        # for pair in set_pair:
        #     s1, s2 = pair
        #     print(len(s1 & s2))

    def save_metadata(self, df_train, df_test, i):
        categories = sorted(list(df_test.speciesKey.unique()))
        categories_map = {str(categ): id for id, categ in enumerate(categories)}
        with open(
            os.path.join(self.metadata_dir, f"category_map_day_{i}.json"), "w"
        ) as f:
            json.dump(categories_map, f)

        df_test["label"] = df_test["speciesKey"].astype(str).map(categories_map)

        # save csv and txt
        df_test.to_csv(
            os.path.join(self.metadata_dir, f"test_day_{i}.csv"),
            index=False,
        )
        test_lines = [
            f"{image_path} {label}"
            for image_path, label in zip(df_test["image_path"], df_test["label"])
        ]

        with open(os.path.join(self.metadata_dir, f"test_day_{i}.txt"), "w") as f:
            for line in test_lines:
                f.write(f"{line}\n")

        df_train.to_csv(
            os.path.join(self.metadata_dir, f"train_day_{i}.csv"),
            index=False,
        )
        train_lines = [
            f"{image_path} -1" for image_path in df_train["image_path"]
        ]  # -1 as this dataset is the "unknown unlabeled" training dataset

        with open(os.path.join(self.metadata_dir, f"train_day_{i}.txt"), "w") as f:
            for line in train_lines:
                f.write(f"{line}\n")


if __name__ == "__main__":
    dataset = PanamaPlusIncrementalGCDDataset()
    region_range = {
        "lat_min": 7,
        "lat_max": 10,
        "lng_min": -83,
        "lng_max": -77,
    }
    dataset.sample_speices(region_range)

    # filename = "/network/scratch/y/yuyan.chen/species_discovery/metadata/classification/panama_train.csv"
    # df = pd.read_csv(filename)
    # labels_csv = to_txt(df)
    # with open(
    #     "/network/scratch/y/yuyan.chen/species_discovery/metadata/classification/panama_train.txt"
    # ) as file:
    #     labels_txt = [int(line.rstrip().split()[-1]) for line in file]

    # print(labels_csv == labels_txt)
    # print(labels_csv[0])
    # print(labels_txt[0])
