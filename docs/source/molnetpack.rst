PyPI package installation and usage
====================================

Using 3DMolMS through ``molnetpack`` requires minimal coding and is easy to use, but it does not support model training. If you want to train your own model, please refer to the :doc:`./sourcecode` page.

Installing from PyPI
--------------------

3DMolMS is available on PyPI as the package ``molnetpack``. You can install the latest version using ``pip``:

.. code-block:: bash

   pip install molnetpack

PyTorch must be installed separately. Check the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for the proper version for your system. For example:

.. code-block:: bash

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Using ``molnetpack`` for MS/MS prediction
-----------------------------------------

The sample input files, a CSV and an MGF, are located at ``./test/demo_input.csv`` and ``./test/demo_input.mgf``, respectively. It's important to note that during the data loading phase, any input formats that are not supported will be automatically excluded. Below is a table outlining the types of input data that are supported:

.. list-table::
   :header-rows: 1

   * - Item
     - Supported input
   * - Atom number
     - <=300
   * - Atom types
     - 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'
   * - Precursor types
     - '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+', '[M+2H]2+'
   * - Collision energy
     - any number

To get started quickly, you can instantiate a MolNet and load a CSV or MGF file for MS/MS prediction using ``load_data`` function:

.. autofunction:: molnetpack.MolNet.load_data

Then predict the MS/MS spectra using ``pred_msms`` function. The predicted MS/MS spectra will be saved in the specified path. The default format is MGF, but you can also save it as a CSV file by specifying the file name with a ``.csv`` extension.

.. autofunction:: molnetpack.MolNet.pred_msms

For example: 

.. code-block:: python

   import torch
   from molnetpack import MolNet, plot_msms

   # Set the device to CPU for CPU-only usage:
   device = torch.device("cpu")

   # For GPU usage, set the device as follows (replace '0' with your desired GPU index):
   # gpu_index = 0
   # device = torch.device(f"cuda:{gpu_index}")

   # Instantiate a MolNet object
   molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

   # Load input data (here we use a CSV file as an example)
   molnet_engine.load_data(path_to_test_data='./test/input_msms.csv')
   
   # Predict MS/MS
   pred_spectra_df = molnet_engine.pred_msms(instrument='qtof')


Plot predicted MS/MS
--------------------

The predicted MS/MS spectra can be visualized using the ``plot_msms`` function: 

.. autofunction:: molnetpack.plot_msms

You may customize the plot by updating the source code directory, such as the size of the image and the color scheme. 

For example:

.. code-block:: python

   # Plot the predicted MS/MS with 3D molecular conformation
   plot_msms(pred_spectra_df, dir_to_img='./img/')

Below is an example of a predicted MS/MS spectrum plot.

.. figure:: https://raw.githubusercontent.com/JosieHong/3DMolMS/main/img/demo_0.png
   :width: 600
   :align: center

Using ``molnetpack`` for properties prediction
----------------------------------------------

Before doing any prediction, please intantiate ``MolNet``:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   # Set the device to CPU for CPU-only usage:
   device = torch.device("cpu")

   # For GPU usage, set the device as follows (replace '0' with your desired GPU index):
   # gpu_index = 0
   # device = torch.device(f"cuda:{gpu_index}")

   # Instantiate a MolNet object
   molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

RT prediction
~~~~~~~~~~~~~~

For RT prediction, please use ``pred_rt`` function after instantiating a MolNet object. Please note that since this model is trained on the METLIN-SMRT dataset, the predicted retention time is under the same experimental conditions as the METLIN-SMRT set.

.. autofunction:: molnetpack.MolNet.pred_rt

For example:

.. code-block:: python

   # Load input data
   molnet_engine.load_data(path_to_test_data='./test/input_rt.csv')

   # Pred RT
   rt_df = molnet_engine.pred_rt()

CCS prediction
~~~~~~~~~~~~~~

For CCS prediction, please use ``pred_ccs`` function after instantiating a MolNet object. 

.. autofunction:: molnetpack.MolNet.pred_ccs

For example: 

.. code-block:: python

   # Load input data
   molnet_engine.load_data(path_to_test_data='./test/input_ccs.csv')

   # Pred CCS
   ccs_df = molnet_engine.pred_ccs()

Molecular feature embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For saving the molecular embeddings, please use the following ``save_features`` function after instantiating a MolNet object. 

.. autofunction:: molnetpack.MolNet.save_features

For example: 

.. code-block:: python

   # Load input data
   molnet_engine.load_data(path_to_test_data='./test/input_savefeat.csv')

   # Inference to get the features
   ids, features = molnet_engine.save_features()

   print('Titles:', ids)
   print('Features shape:', features.shape)