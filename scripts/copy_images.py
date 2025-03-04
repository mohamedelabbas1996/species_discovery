import shutil
import os

# Define paths
txt_file = "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/txt/03_c-america_test_id.txt"  # Replace with the actual path of your text file
source_dir = "/network/scratch/y/yuyan.chen/ami_gbif_train_resized/"  # Replace with the actual source directory
destination_dir = (
    "/network/scratch/y/yuyan.chen/species_discovery/images/"  # Replace with the actual destination directory
)

# Ensure the destination directory exists

# Read the text file and copy the files
with open(txt_file, "r") as f:
    for line in f:
        filename = line.split()[0]  # Extract the filename (first part of each line)
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)

        os.makedirs("/".join(destination_path.split("/")[:-1]), exist_ok=True)  # create subdir

        # Copy file if it exists
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"File not found: {filename}")
