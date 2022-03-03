'''
Date: 2021-09-08 13:52:50
LastEditors: yuhhong
LastEditTime: 2022-04-19 16:21:31
'''
import numpy as np
import torch.nn.functional as F


def get_metrics(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    acc = F.cosine_similarity(y_true, y_pred, dim=1).mean().item()
    # dot_product = 
    dice, precision, recall = np.mean([dice_precision_recall(x, y) for x, y in zip(y_true, y_pred)], axis=0)
    return {'acc': acc, 'dice': dice, 'recall': recall, 'precision': precision}

def MZ(ms):
    mzs = [mz for mz, inten in enumerate(ms) if inten != 0]
    return set(mzs)

def dice_precision_recall(x, y):
    mz_x = MZ(x)
    mz_y = MZ(y)
    return 2 * len(mz_x.intersection(mz_y)) / (len(mz_x) + len(mz_y)), \
        len(mz_x.intersection(mz_y)) / len(mz_y), \
        len(mz_x.intersection(mz_y)) / len(mz_x)


