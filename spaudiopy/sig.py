# -*- coding: utf-8 -*-
"""
Avoid code duplications (and errors) by defining a few custom classes here.
"""

import copy

import numpy as np
from scipy import signal as scysig
import soundfile as sf

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

    def save(self, filename):
        IO.save_audio(self, filename)

    def trim(self, start, stop):
        """Trim audio to start and stop in seconds."""
        assert start < len(self) / self.fs, "Trim start exceeds signal."
        self.signal = self.signal[int(start * self.fs): int(stop * self.fs)]

    def apply(self, func, *args, **kwargs):
        """Apply function 'func' to signal, arguments are forwarded."""
        self.signal = func(*args, **kwargs)

    def filter(self, h, **kwargs):
        """Convolve signal, kwargs are forwarded to signal.convolve."""
        h = utils.asarray_1d(h)
        return scysig.convolve(self.signal, h, **kwargs)


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

    def filter(self, h, **kwargs):
        raise NotImplementedError


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
        return super().from_file(filename, fs=fs)

    def sh_to_b(self):
        # Assume signals are in ACN
        _B = sph.sh_to_b(self.get_signals())
        self.channel[0].signal = _B[0, :]
        self.channel[1].signal = _B[1, :]
        self.channel[2].signal = _B[2, :]
        self.channel[3].signal = _B[3, :]
        self.W = utils.asarray_1d(self.channel[0].signal)
        self.X = utils.asarray_1d(self.channel[1].signal)
        self.Y = utils.asarray_1d(self.channel[2].signal)
        self.Z = utils.asarray_1d(self.channel[3].signal)


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
        d_idx = np.argmin(pcs.haversine_dist(grid_phi, grid_theta, phi, theta))
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
        return np.argmin(pcs.haversine_dist(grid_phi, grid_theta, phi, theta))


def trim_audio(A, start, stop):
    """Trim copy of MultiSignal audio to start and stop in seconds."""
    B = copy.deepcopy(A)
    assert start < len(B) / B.fs, 'Trim start exceeds signal.'
    for c in range(B.channel_count):
        B.channel[c].signal = B.channel[c].signal[
            int(start * B.fs):
            int(stop * B.fs)]
    return B
