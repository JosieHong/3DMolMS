import torch
import molnetpack
from molnetpack import MolNet

# CPU; for GPU use torch.device(f"cuda:{gpu_index}")
device = torch.device("cpu")

molnet_engine = MolNet(device, seed=42)

# Load input molecules. CSV, MGF, and PKL are supported; increase batch_size to speed up.
molnet_engine.load_data(path_to_test_data="./examples/demo_input.csv")
# molnet_engine.load_data(path_to_test_data="./examples/demo_input.mgf")

# Predict MS/MS. The released checkpoint downloads automatically on first use;
# pass path_to_checkpoint="..." to use your own model instead.
msms_res_df = molnet_engine.pred_msms(
    path_to_results="./examples/output_msms.mgf",  # or a .csv path
    instrument="qtof",  # or "orbitrap"
)

# Plot the predicted spectra with their 2D structures
molnetpack.plot_msms(msms_res_df=msms_res_df, dir_to_img="./img")
