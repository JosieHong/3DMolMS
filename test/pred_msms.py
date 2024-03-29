import torch
from molnetpack import MolNet

# CPU
device = torch.device("cpu")

# GPU
# gpu_index = 0 # please set this into the index of GPU you plan to use
# device = torch.device("cuda:" + str(gpu_index))

molnet_engine = MolNet(device, seed=42)

# molnet_engine.load_data(path_to_test_data='./test/demo_input.csv', path_to_save_pkl='./test/demp_input.pkl')
molnet_engine.load_data(path_to_test_data='./test/demo_msms_input.csv')

spectra = molnet_engine.pred_msms(path_to_results='./test/demo_msms.mgf')

molnet_engine.plot_msms(dir_to_img='./test/img/')
