# -*- coding: utf-8 -*-
"""
Avoid code duplications (and errors) by defining a few custom classes here.
"""

import copy

import numpy as np
from scipy import signal as scysig
import soundfile as sf

from . import utils
from . import process


# CLASSES
class MonoSignal:
    """Signal class for a MONO channel audio signal."""

    def __init__(self, signal, fs):
        """Constructor."""
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

    def __init__(self, *signals, fs=None):
        """Constructor."""
        self.channel = []
        if fs is None:
            raise ValueError("Provide fs (as kwarg).")
        else:
            self.fs = fs
        if isinstance(signals[0], list):
            signals = signals[0]  # unpack tuṕle if list was given
        for s in signals:
            self.channel.append(MonoSignal(s, fs))
        self.channel_count = len(self.channel)

    def __len__(self):
        """Override len()."""
        return len(self.channel[0])

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
        return cls(*sig.T, fs=fs)

    def get_signals(self):
        """Return ndarray of signals, stacked along first dimension."""
        return np.asarray([x.signal for x in self.channel])

    def trim(self, start, stop):
        """Trim all channels to start and stop in seconds."""
        assert start < len(self) / self.fs, "Trim start exceeds signal."
        for c in self.channel:
            c.signal = c.signal[int(start * c.fs): int(stop * c.fs)]


class AmbiBSignal(MultiSignal):
    """Signal class for first order Ambisonics B-format signals."""
    def __init__(self, *signals, fs=None):
        MultiSignal.__init__(self, *signals, fs=fs)
        assert self.channel_count == 4, "Provide four channels!"
        self.W = self.channel[0].signal
        self.X = self.channel[1].signal
        self.Y = self.channel[2].signal
        self.Z = self.channel[3].signal


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

    def select_direction(self, phi, theta):
        """Return HRIRs for given direction."""
        grid_phi = np.array(self.grid['az'])
        grid_theta = np.array(self.grid['el'])
        h_l, h_r = process.select_hrtf(self.left, self.right,
                                       grid_phi, grid_theta, phi, theta)
        return h_l, h_r


def trim_audio(A, start, stop):
    """Trim copy of MultiSignal audio to start and stop in seconds."""
    B = copy.deepcopy(A)
    assert start < len(B) / B.fs, 'Trim start exceeds signal.'
    for c in range(B.channel_count):
        B.channel[c].signal = B.channel[c].signal[
            int(start * B.fs):
            int(stop * B.fs)]
    return B
