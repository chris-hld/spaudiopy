""" spaudiopy

.. rubric:: Submodules

.. autosummary::
   :toctree:

   IO
   sig
   sph
   decoder
   process
   utils
   grids
   sdm
   plots

"""
from subprocess import check_output


try:
    release = check_output(['git', 'describe', '--tags', '--always'])
    __version__ = release.decode().strip()
except Exception:
    __version__ = "0.1.2-dirty"


from . import decoder
from . import grids
from . import IO
from . import plots
from . import process
from . import sdm
from . import sph
from . import sig
from . import utils
