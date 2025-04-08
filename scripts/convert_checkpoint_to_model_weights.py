import torch

checkpoint = torch.load(
    "/network/scratch/y/yuyan.chen/species_discovery/checkpoints/classification/6337820/checkpoints/model_best.pth",
    weights_only=False,
)
weights = checkpoint["model_state"]
torch.save(
    weights,
    "/network/scratch/y/yuyan.chen/species_discovery/checkpoints/classification/6337820/checkpoints/model_weights.pth",
)
