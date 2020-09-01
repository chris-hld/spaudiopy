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
from pathlib import Path
from subprocess import check_output

file_dir = Path(__file__).parent.absolute()


try:
    release = check_output(['git', 'describe', '--tags', '--always', '--long',
                            '--dirty'],
                           cwd=str(file_dir))
    __version__ = release.decode().strip()

except Exception:
    __version__ = "unknown"


from . import decoder
from . import grids
from . import IO
from . import plots
from . import process
from . import sdm
from . import sph
from . import sig
from . import utils
