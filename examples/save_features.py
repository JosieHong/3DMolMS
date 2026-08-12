import torch
from molnetpack import MolNet

# CPU; for GPU use torch.device(f"cuda:{gpu_index}")
device = torch.device("cpu")

molnet_engine = MolNet(device, seed=42)

molnet_engine.load_data(path_to_test_data="./examples/demo_input.csv")

# Extract the encoder embedding of each molecule (for downstream tasks)
ids, features = molnet_engine.save_features()

print("Titles:", ids)
print("Features shape:", features.shape)
