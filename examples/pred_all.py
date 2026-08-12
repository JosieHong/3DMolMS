import torch
from molnetpack import MolNet

# CPU; for GPU use torch.device(f"cuda:{gpu_index}")
device = torch.device("cpu")

molnet_engine = MolNet(device, seed=42)

molnet_engine.load_data(path_to_test_data="./examples/demo_input.csv")

# Predict MS/MS, RT, and CCS in one call. The result is one row per molecule with
# the spectrum plus "Pred RT" / "Pred CCS" columns; the MGF output stores RT and
# CCS as PRED_RT and PRED_CCS parameters in each ion block.
all_df = molnet_engine.pred_all(
    path_to_results="./examples/output_all.mgf",  # or a .csv path
    instrument="qtof",  # or "orbitrap"
)
print(all_df[["ID", "Pred RT", "Pred CCS"]])
