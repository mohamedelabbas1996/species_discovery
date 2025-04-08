import pandas as pd


def to_txt(df, filename):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            line = f"{row['image_path']} -1\n"
            f.write(line)


df_trap = pd.read_csv("data/ami_trap.csv")
df_trap = df_trap[df_trap["speciesKey"] != -1]
trap_species = df_trap.speciesKey.unique()

df_global = pd.read_csv(
    "/network/scratch/y/yuyan.chen/species_discovery/metadata/classification/global_train.csv"
)

print(len(df_global))
df_global = df_global[df_global["speciesKey"].isin(trap_species)]

print(len(df_global))
df_global = df_global.groupby("speciesKey", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 20), random_state=42)
)

print(len(df_global))

df_global.to_csv("data/ami_gbif_gcd.csv", index=False)

to_txt(df_global, "data/ami_gbif_gcd.txt")
