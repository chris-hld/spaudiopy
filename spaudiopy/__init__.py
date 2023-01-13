""" spaudiopy

.. rubric:: Submodules

.. autosummary::
   :toctree:

   io
   sig
   sph
   decoder
   process
   utils
   grids
   parsa
   plot

"""
from pathlib import Path
from subprocess import run

file_dir = Path(__file__).parent.absolute()


try:
    r = run(['git', 'describe', '--tags', '--always', '--long', '--dirty'],
            check=True, capture_output=True, cwd=str(file_dir))
    __version__ = r.stdout.decode().strip()

except Exception:
    __version__ = "unknown (v0.2.0)"


from . import decoder
from . import grids
from . import io
from . import plot
from . import process
from . import parsa
from . import sph
from . import sig
from . import utils
