* 2024.03.08 (v1.1.3): add the function `save_features`, which can be used to save embedded features; these features can then be used in downstream tasks.
* 2024.02.27 (v1.1.2): fix the independences; update ccs prediction. 
* 2024.02.27 (v1.1.1): PyPI package release. 
* 2023.10.30 (v1.1.0): enlarging training set by MoNA and Waters QTOF datasets. 
* 2023.10.22 (v1.0.2): pretraining on QM9-mu dataset + ETKDG algorithm. We establish a dataset from QM9-mu (dipole moment) with the generated conformations using ETKDG for pretraining 3DMolMS. It helps the model learning knowledge of molecular 3D conformations and pretraining enhances the performance on MS/MS slightly (~0.01 cosine similarity). 
* 2023.09.14 (v1.0.1): data augmentation by flipping atomic coordinates. Notably, this model is sensitive to the geometric structure of molecules. For tasks insensitive to geometric structure, e.g. mass spectrometry is chirally blind, please use data augmentation. However, for the tasks sensitive to geometric structure, e.g. different enantiomers with varying retention times, avoid data augmentation. 
* 2023.06.30 (v1.0.0): initial version. 