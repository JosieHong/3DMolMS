import torch
from molnetpack import MolNet

# CPU; for GPU use torch.device(f"cuda:{gpu_index}")
device = torch.device("cpu")

molnet_engine = MolNet(device, seed=42)

# Training pickles come from `python scripts/build_msms_dataset.py --group qtof`;
# see the documentation's MS/MS training guide.

# ---------------------------------------------------------------------------
# Option 1: fine-tune from a pretrained encoder (recommended).
# The encoder is frozen by default (head-only training);
# pass freeze_encoder=False to train everything.
# ---------------------------------------------------------------------------
molnet_engine.train(
    task="msms",
    train_data="./data/qtof_all_train.pkl",
    valid_data="./data/qtof_all_val.pkl",
    checkpoint_path="./check_point/molnet_qtof_tl.pt",
    resume_path="./check_point/molnet_pre_geobond.pt",  # pretrained encoder
    transfer=True,
)

# ---------------------------------------------------------------------------
# Option 2: train from scratch (no pretrained weights).
# ---------------------------------------------------------------------------
# molnet_engine.train(
#     task="msms",
#     train_data="./data/qtof_all_train.pkl",
#     valid_data="./data/qtof_all_val.pkl",
#     checkpoint_path="./check_point/molnet_qtof.pt",
# )

# ---------------------------------------------------------------------------
# After training, the model is immediately ready for inference; no reload needed.
# ---------------------------------------------------------------------------
molnet_engine.load_data(path_to_test_data="./data/qtof_all_test.pkl")
pred_df = molnet_engine.pred_msms(
    path_to_results="./result/pred_qtof_test.mgf",
    instrument="qtof",
)

# ---------------------------------------------------------------------------
# Evaluate predicted spectra against ground truth.
# ---------------------------------------------------------------------------
results_df = molnet_engine.evaluate(
    test_pkl="./data/qtof_all_test.pkl",
    pred_mgf="./result/pred_qtof_test.mgf",
    result_path="./result/eval_qtof_test.csv",
    plot_path="./result/eval_qtof_test.png",
)
