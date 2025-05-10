Fine-tune on your own data
==========================

This section introduces how to fine-tune the model for regression tasks, such as retention time prediction, on your own data.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Please prepare the data of molecular properties as:

.. code-block:: text

   ,id,smiles,prop
   0,0382_00004,NC(=O)N1c2ccccc2[C@H](O)[C@@H](O)c2ccccc21,5.79
   1,0382_00005,CN(C)[C@@H]1C(=O)C(C(N)=O)=C(O)[C@@]2(O)C(=O)C3=C(O)c4c(O)ccc(Cl)c4[C@@](C)(O)[C@H]3C[C@@H]12,4.5
   2,0382_00008,Cc1onc(-c2c(Cl)cccc2Cl)c1C(=O)N[C@@H]1C(=O)N2[C@@H](C(=O)O)C(C)(C)S[C@H]12,7.8
   3,0382_00009,C[C@H]1c2cccc(O)c2C(O)=C2C(=O)[C@]3(O)C(O)=C(C(N)=O)C(=O)[C@@H](N(C)C)[C@@H]3[C@@H](O)[C@@H]21,6.2
   4,0382_00010,C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(O)ccc4[C@H]3CC[C@@]21C,9.46
   5,0382_00012,Cc1onc(-c2ccccc2)c1C(=O)N[C@@H]1C(=O)N2[C@@H](C(=O)O)C(C)(C)S[C@H]12,6.9

where ``prop`` column is the RT or CCS values.

**Step 2**: Training
--------------------

Fine-tune the model!

.. code-block:: bash

   python ./src/train_csv.py --data <path to csv/pkl data> \ 
   --model_config_path <path to configuration> \
   --checkpoint_path <path to save the checkpoint> \
   --transfer \
   --resume_path <path to pretrained model> \
   --result_path <path to save the prediction results> \
   --seed 42 # for randomly data splitting and randomly dropping out the neurons in the model

   # If the data is already preprocessed, you could use pkl file directly.

**Step 3**: Running prediction
------------------------------

Predict the unlabeled data: 

.. code-block:: bash

   python pred_csv.py --data <path to csv/pkl data> \
   --model_config_path <path to configuration> \
   --checkpoint_path <path to save the checkpoint> \
   --result_path <path to save the prediction results> \
   --seed 42

