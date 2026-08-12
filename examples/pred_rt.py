import torch
from molnetpack import MolNet

# CPU; for GPU use torch.device(f"cuda:{gpu_index}")
device = torch.device("cpu")

molnet_engine = MolNet(device, seed=42)

molnet_engine.load_data(path_to_test_data="./examples/demo_input.csv")

# Predict retention time (METLIN-SMRT conditions). The released checkpoint
# downloads automatically; pass path_to_checkpoint="..." to use your own model.
rt_df = molnet_engine.pred_rt(path_to_results="./examples/output_rt.csv")
