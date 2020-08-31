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
from subprocess import run

file_dir = Path(__file__).parent.absolute()


try:
    # release = check_output(['git', 'describe', '--tags', '--always'])
    # __version__ = release.decode().strip()

    release = run(['git', 'describe', '--tags', '--always'],
                  cwd=str(file_dir),
                  capture_output=True)
    __version__ = release.stdout.decode().strip()

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
