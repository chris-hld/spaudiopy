Installation
============

For the unpatient, you can just install the pip version
  `pip install spaudiopy`


Requirements
------------
It's easiest to start with something like `Anaconda <https://www.anaconda.com/distribution/>`_ as a Python distribution.
You'll need Python >= 3.6 .

#. Create a conda environment:  
    * `conda create --name spaudio python=3.6 anaconda joblib portaudio`
#. Activate this new environment:  
    * `conda activate spaudio`


Have a look at the `setup.py` file, all dependencies are listed there.
When using `pip` to install this package as shown below, all remaining dependencies not available from conda will be downloaded and installed automatically.

Installation
------------
Download this package from `GitHub <https://github.com/chris-hld/spaudiopy>`_ and navigate to there. Then simply run the line: ::

  pip install -e .

This will check all dependencies and install this package as editable.

  
