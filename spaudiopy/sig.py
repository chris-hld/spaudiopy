# -*- coding: utf-8 -*-
""" Signal class.
Avoid code duplications (and errors) by defining a few custom classes here.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

"""

import copy
from warnings import warn

import numpy as np
from scipy import signal as scysig
import soundfile as sf
try:
    import sounddevice as sd
except (ImportError, OSError) as e:
    print(str(e))
    warn("Sounddevice not available.")

from . import utils, IO, sph
from . import process as pcs


# CLASSES
class MonoSignal:
    """Signal class for a MONO channel audio signal."""

    def __init__(self, signal, fs):
        """Constructor

        Parameters
        ----------
        signal : array_like
        fs : int

        """
        self.signal = utils.asarray_1d(signal)
        self.fs = fs

    def __len__(self):
        """Override len()."""
        return len(self.signal)

    def __getitem__(self, key):
        """Enable [] operator, returns signal data."""
        return self.signal[key]

    @classmethod
    def from_file(cls, filename, fs=None):
        """Alternative constructor, load signal from filename."""
        sig, fs_file = sf.read(filename)
        if fs is not None:
            if fs != fs_file:
                raise ValueError("File: Found different fs:" + str(fs_file))
        else:
            fs = fs_file
        if sig.ndim != 1:
            raise ValueError("Signal must be mono. Try MultiSignal.")
        return cls(sig, fs)

    def copy(self):
        """Return an independent (deep) copy of the signal."""
        return copy.deepcopy(self)

    def save(self, filename):
        """Save to file."""
        IO.save_audio(self, filename)

    def trim(self, start, stop):
        """Trim audio to start and stop in seconds."""
        assert start < len(self) / self.fs, "Trim start exceeds signal."
        self.signal = self.signal[int(start * self.fs): int(stop * self.fs)]

    def apply(self, func, *args, **kwargs):
        """Apply function 'func' to signal, arguments are forwarded."""
        self.signal = func(*args, **kwargs)

    def conv(self, h, **kwargs):
        """Convolve signal, kwargs are forwarded to signal.convolve."""
        h = utils.asarray_1d(h)
        self.signal = scysig.convolve(self.signal, h, **kwargs)
        return self

    def play(self, gain=1, wait=True):
        """Play sound signal. Adjust gain and wait until finished."""
        sd.play(gain * self.signal, int(self.fs))
        if wait:
            sd.wait()


class MultiSignal(MonoSignal):
    """Signal class for a MULTI channel audio signal."""

    def __init__(self, signals, fs=None):
        """Constructor

        Parameters
        ----------
        signals : list of array_like
        fs : int

        """
        assert isinstance(signals, (list, tuple))
        self.channel = []
        if fs is None:
            raise ValueError("Provide fs (as kwarg).")
        else:
            self.fs = fs
        for s in signals:
            self.channel.append(MonoSignal(s, fs))
        self.channel_count = len(self.channel)

    def __len__(self):
        """Override len()."""
        return len(self.channel[0])

    def __getitem__(self, key):
        """Override [] operator, returns signal channel."""
        return self.channel[key]

    @classmethod
    def from_file(cls, filename, fs=None):
        """Alternative constructor, load signal from filename."""
        sig, fs_file = sf.read(filename)
        if fs is not None:
            if fs != fs_file:
                raise ValueError("File: Found different fs:" + str(fs_file))
        else:
            fs = fs_file
        if np.ndim(sig) == 1:
            raise ValueError("Only one channel. Try MonoSignal.")
        return cls([*sig.T], fs=fs)

    def get_signals(self):
        """Return ndarray of signals, stacked along rows."""
        return np.asarray([x.signal for x in self.channel])

    def trim(self, start, stop):
        """Trim all channels to start and stop in seconds."""
        assert start < len(self) / self.fs, "Trim start exceeds signal."
        for c in self.channel:
            c.signal = c.signal[int(start * c.fs): int(stop * c.fs)]

    def apply(self, func, *args, **kwargs):
        """Apply function 'func' to all signals, arguments are forwarded."""
        for c in self.channel:
            c.signal = func(*args, **kwargs)

    def conv(self, irs, **kwargs):
        for c, h in zip(self.channel, irs):
            h = utils.asarray_1d(h)
            c.signal = scysig.convolve(c, h, **kwargs)
        return self

    def play(self, gain=1, wait=True):
        """Play sound signal. Adjust gain and wait until finished."""
        sd.play(gain * self.get_signals().T, int(self.fs))
        if wait:
            sd.wait()


class AmbiBSignal(MultiSignal):
    """Signal class for first order Ambisonics B-format signals."""
    def __init__(self, signals, fs=None):
        MultiSignal.__init__(self, signals, fs=fs)
        assert self.channel_count == 4, "Provide four channels!"
        self.W = utils.asarray_1d(self.channel[0].signal)
        self.X = utils.asarray_1d(self.channel[1].signal)
        self.Y = utils.asarray_1d(self.channel[2].signal)
        self.Z = utils.asarray_1d(self.channel[3].signal)

    @classmethod
    def from_file(cls, filename, fs=None):
        """Alternative constructor, load signal from filename."""
        return super().from_file(filename, fs=fs)

    @classmethod
    def sh_to_b(cls, multisig):
        """Alternative constructor, convert from sig.Multisignal.
        Assumes ACN channel order.
        """
        assert isinstance(multisig, MultiSignal)
        _B = sph.sh_to_b(multisig.copy().get_signals())
        return cls([*_B], fs=multisig.fs)


class HRIRs:
    """Signal class for head-related impulse responses."""

    def __init__(self, left, right, grid, fs):
        """Constructor."""
        assert len(left) == len(right), "Signals must be of same length."
        self.left = left
        self.right = right
        self.grid = grid
        self.fs = fs
        assert len(grid) == len(left)
        self.grid_points = len(grid)

    def __len__(self):
        """Override len() to count of samples per hrir."""
        return self.left.shape[1]

    def __getitem__(self, key):
        """Enable [] operator, returns hrirs."""
        return self.left[key, :], self.right[key, :]

    def nearest_hrirs(self, phi, theta):
        """
        For a point on the sphere, select closest hrir defined on grid,
        based on the haversine distance.

        Parameters
        ----------
        phi : float
            Azimuth.
        theta : float
            Elevation (colat).

        Returns
        -------
        h_l : (n,) array_like
            h(t) closest to [phi, theta].
        h_r : (n,) array_like
            h(t) closest to [phi, theta].
        """
        grid_phi = self.grid['azi'].values
        grid_theta = self.grid['colat'].values
        # search closest gridpoint
        d_idx = np.argmin(utils.haversine(grid_phi, grid_theta, phi, theta))
        VERBOSE = False
        if VERBOSE:
            with open("selected_hrtf.txt", "a") as f:
                f.write("idx {}, phi: {}, g_phi: {}, th: {}, g_th: {}".format(
                    d_idx,
                    utils.rad2deg(phi), utils.rad2deg(grid_phi[d_idx]),
                    utils.rad2deg(theta), utils.rad2deg(grid_theta[d_idx])))
                f.write('\n')
        # get hrirs to that angle
        return self[d_idx]

    def nearest(self, phi, theta):
        """
        Index of nearest hrir grid point based on haversine distance.

        Parameters
        ----------
        phi : float
            Azimuth.
        theta : float
            Colaitude.
        Returns
        -------
        idx : int
            Index.
        """
        grid_phi = self.grid['azi'].values
        grid_theta = self.grid['colat'].values
        return np.argmin(utils.haversine(grid_phi, grid_theta, phi, theta))


def trim_audio(A, start, stop):
    """Trim copy of MultiSignal audio to start and stop in seconds."""
    B = copy.deepcopy(A)
    assert start < len(B) / B.fs, 'Trim start exceeds signal.'
    for c in range(B.channel_count):
        B.channel[c].signal = B.channel[c].signal[
            int(start * B.fs):
            int(stop * B.fs)]
    return B
