import torch
from molnetpack import MolNet

# CPU
device = torch.device("cpu")

# GPU
# gpu_index = 0 # please set this into the index of GPU you plan to use
# device = torch.device("cuda:" + str(gpu_index))

molnet_engine = MolNet(device, seed=42)

# molnet_engine.load_data(path_to_test_data='./test/demo_input.csv', path_to_save_pkl='./test/demp_input.pkl')
molnet_engine.load_data(path_to_test_data='./test/demo_input.csv')

ccs_df = molnet_engine.pred_ccs(path_to_results='./test/demo_ccs.csv')
