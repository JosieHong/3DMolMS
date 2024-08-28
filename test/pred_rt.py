import torch
from molnetpack import MolNet

# CPU
device = torch.device("cpu")

# GPU
# gpu_index = 0 # please set this into the index of GPU you plan to use
# device = torch.device("cuda:" + str(gpu_index))

molnet_engine = MolNet(device, seed=42)

# Load input data
molnet_engine.load_data(path_to_test_data='./test/input_rt.csv')

# Pred RT
# rt_df = molnet_engine.pred_rt(path_to_results='./test/output_rt.csv', path_to_checkpoint='./check_point/molnet_rt_etkdgv3_tl.pt')
rt_df = molnet_engine.pred_rt(path_to_results='./test/output_rt.csv') # Download checkpoint from GitHub release page. 
