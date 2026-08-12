Source code setup
=================

Installing from source code
---------------------------

3DMolMS can also be installed through source code:

.. code-block:: bash

   git clone https://github.com/JosieHong/3DMolMS.git
   cd 3DMolMS
   pip install -e .

PyTorch must be installed separately. Check the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for the proper version for your system. For example:

.. code-block:: bash

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

The ``-e`` (editable) install means code changes take effect immediately, with no
reinstall needed. Then you can use the package for various tasks as shown in the :doc:`./usage/index` and :doc:`./advanced_usage/index` page.

Requirements
------------

3DMolMS has the following dependencies:

* Python 3.10+
* PyTorch
* RDKit
* NumPy
* Pandas
* matplotlib
* PyYAML
* Other dependencies listed in ``pyproject.toml``

Most dependencies will be automatically installed when using pip, but PyTorch should be installed separately as mentioned above to ensure compatibility with your system's CUDA version.