import torch
from molnetpack import MolNet

# Set the device to CPU for CPU-only usage:
device = torch.device("cpu")

# For GPU usage, set the device as follows (replace '0' with your desired GPU index):
# gpu_index = 0
# device = torch.device(f"cuda:{gpu_index}")

# Instantiate a MolNet object
molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

# Load input data (here we use a CSV file as an example)
molnet_engine.load_data(path_to_test_data='./test/input_msms.csv') # Increasing the batch size if you wanna speed up.
# molnet_engine.load_data(path_to_test_data='./test/input_msms.mgf') # MGF file is also supported
# molnet_engine.load_data(path_to_test_data='./test/input_msms.pkl') # PKL file is faster. 

# Predict MS/MS
spectra = molnet_engine.pred_msms(path_to_results='./test/output_msms.mgf', path_to_checkpoint='./check_point/molnet_qtof_etkdgv3.pt', instrument='qtof')

# Plot the predicted MS/MS with 3D molecular conformation
molnet_engine.plot_msms(dir_to_img='./img/', instrument='qtof')