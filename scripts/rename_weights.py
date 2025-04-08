import torch


weights_path = "/network/scratch/y/yuyan.chen/species_discovery/weights/panama_mobilenet_20240417_161141_30.pth"
weights = torch.load(weights_path)

new_weights = {}

for key in weights:
    new_weights["model." + key] = weights[key]

torch.save(new_weights, weights_path)
