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

import os
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

from . import io, utils, sph
from . import process as pcs


# CLASSES
class MonoSignal:
    """Signal class for a MONO channel audio signal."""

    def __init__(self, signal, fs):
        """Constructor.

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
        sig, fs_file = sf.read(os.path.expanduser(filename))
        if fs is not None:
            if fs != fs_file:
                raise ValueError("File: Found different fs:" + str(fs_file))
        else:
            fs = fs_file
        if sig.ndim != 1:
            raise ValueError("Signal must be mono. Try MultiSignal.")
        return cls(sig, fs)

    def copy(self):
        """Return an independent (deep) copy of the instance."""
        return copy.deepcopy(self)

    def save(self, filename, subtype='FLOAT'):
        """Save to file."""
        io.save_audio(self, os.path.expanduser(filename), subtype=subtype)

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

    def resample(self, fs_new):
        """Resample signal to new sampling rate fs_new."""
        if fs_new == self.fs:
            warn("Same sampling rate requested, no resampling.")
        else:
            sig_resamp = pcs.resample_signal(self.signal, self.fs, fs_new)
            self.__init__(sig_resamp, fs_new)

    def play(self, gain=1, wait=True):
        """Play sound signal. Adjust gain and wait until finished."""
        sd.play(gain * self.signal, int(self.fs))
        if wait:
            sd.wait()


class MultiSignal(MonoSignal):
    """Signal class for a MULTI channel audio signal."""

    def __init__(self, signals, fs=None):
        """Constructor.

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
        sig, fs_file = sf.read(os.path.expanduser(filename))
        if fs is not None:
            if fs != fs_file:
                raise ValueError("File: Found different fs:" + str(fs_file))
        else:
            fs = fs_file
        if np.ndim(sig) == 1:
            raise ValueError("Only one channel. Try MonoSignal.")
        return cls([*sig.T], fs=fs)

    def get_signals(self):
        """Return ndarray of signals, stacked along rows (nCH, nSmps)."""
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

    def resample(self, fs_new):
        """Resample signal to new sampling rate fs_new."""
        if fs_new == self.fs:
            warn("Same sampling rate requested, no resampling.")
        else:
            sig_resamp = pcs.resample_signal(self.get_signals(),
                                             self.fs, fs_new, axis=-1)
            self.__init__([*sig_resamp], fs_new)

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

    def __init__(self, left, right, azi, zen, fs):
        """Constructor.

        Parameters
        ----------
        left : (numDirs, numTaps) ndarray
            Left ear HRIRs.
        right : (numDirs, numTaps) ndarray
            Right ear HRIRs.
        azi : (numDirs,) array_like, in rad
        fs : int

        """
        left = np.asarray(left)
        right = np.asarray(right)
        assert len(left) == len(right), "Signals must be of same length."
        assert left.ndim == 2
        assert right.ndim == 2
        azi = utils.asarray_1d(azi)
        zen = utils.asarray_1d(zen)
        assert len(azi) == len(zen)
        assert len(azi) == left.shape[0]

        self.left = left
        self.right = right
        self.azi = azi
        self.zen = zen
        self.fs = fs
        self.num_grid_points = len(azi)
        self.num_samples = left.shape[1]

    def __len__(self):
        """Override len() to count of samples per hrir."""
        return self.left.shape[1]

    def __getitem__(self, key):
        """Enable [] operator, returns hrirs."""
        return self.left[key, :], self.right[key, :]

    def copy(self):
        """Return an independent (deep) copy of the instance."""
        return copy.deepcopy(self)

    def update_hrirs(self, left, right):
        """Update and replace HRIRs in place.

        Parameters
        ----------
        left : (numDirs, numTaps) ndarray
            Left ear HRIRs.
        right : numDirs, numTaps ndarray
            Right ear HRIRs.

        Returns
        -------
        None.

        """
        # reinitialize
        self.__init__(left, right, self.azi, self.zen, self.fs)

    def nearest_hrirs(self, azi, zen):
        """For a point on the sphere, select closest HRIR defined on grid.

        Based on the haversine distance.

        Parameters
        ----------
        azi : float
            Azimuth.
        zen : float
            Zenith / Colatitude.

        Returns
        -------
        h_l : (n,) array_like
            h(t) closest to [phi, theta].
        h_r : (n,) array_like
            h(t) closest to [phi, theta].
        """
        # search closest gridpoint
        d_idx = self.nearest_idx(azi, zen)
        # get hrirs to that angle
        return self[d_idx]

    def nearest_idx(self, azi, zen):
        """
        Index of nearest HRIR grid point based on dot product.

        Parameters
        ----------
        azi : float, array_like
            Azimuth.
        zen : float, array_like
            Zenith / Colatitude.

        Returns
        -------
        idx : int, np.ndarray
            Index.
        """
        azi = utils.asarray_1d(azi)
        zen = utils.asarray_1d(zen)
        vec = np.stack(utils.sph2cart(azi, zen), axis=1)
        vec_g = np.stack(utils.sph2cart(self.azi, self.zen), axis=1)
        return np.argmax(vec@vec_g.T, axis=1).squeeze()

    def apply_ctf_eq(self, eq_taps=None, mode='full'):
        """
        Equalize common transfer function (CTF) of HRIRs.

        Parameters
        ----------
        eq_taps : array_like, optional
            FIR filter, `None` will calculate. The default is None.
        mode : string, optional
            Forwarded to scipy.signal.convolve(). The default is 'full'.

        Returns
        -------
        None.

        """
        if eq_taps is None:
            eq_taps = pcs.hrirs_ctf(self)
        self.left = scysig.convolve(self.left, eq_taps[None, :], mode)
        self.right = scysig.convolve(self.right, eq_taps[None, :], mode)
        self.num_samples = self.left.shape[1]


def trim_audio(A, start, stop):
    """Trim copy of MultiSignal audio to start and stop in seconds."""
    B = copy.deepcopy(A)
    assert start < len(B) / B.fs, 'Trim start exceeds signal.'
    for c in range(B.channel_count):
        B.channel[c].signal = B.channel[c].signal[
            int(start * B.fs):
            int(stop * B.fs)]
    return B
