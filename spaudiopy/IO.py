# -*- coding: utf-8 -*-
"""
@author: chris
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

import soundfile as sf

from . import utils
from . import sig
from . import sph
from . import decoder


def load_hrir(fs, filename=None, dummy=False):
    """
    Convenience function to load HRTF.mat.

    Parameters
    ----------
    fs : int
        fs(t).
    filename : string, optional
        HRTF.mat file or default set.
    dummy : bool, optional
        Returns dummy hrirs (debugging).

    Returns
    -------
    HRIRs : sig.HRIRs instance
        left : (g, h) numpy.ndarray
            h(t) for grid position g.
        right : (g, h) numpy.ndarray
            h(t) for grid position g.
        grid : (g, 2) pandas.dataframe
            [azimuth, elevation(colat)] for hrirs.
        fs : int
            fs(t).
    """
    if filename is None:
        if fs == 44100:
            default_file = '../data/HRTF_default.mat'
        elif fs == 48000:
            default_file = '../data/HRTF_default48k.mat'
        else:
            raise ValueError("No default hrirs.")
        current_file_dir = os.path.dirname(__file__)
        filename = os.path.join(current_file_dir, default_file)

    mat = loadmat(filename)
    hrir_l = np.array(np.squeeze(mat['hrir_l']), dtype=float)
    hrir_r = np.array(np.squeeze(mat['hrir_r']), dtype=float)
    hrir_fs = int(mat['SamplingRate'])
    azi = np.array(np.squeeze(mat['azi']), dtype=float)
    elev = np.array(np.squeeze(mat['elev']), dtype=float)
    grid = pd.DataFrame({'az': azi, 'el': elev})
    if dummy is True:
        # Create diracs as dummy
        hrir_l = np.zeros_like(hrir_l)
        hrir_l[:, 0] = np.ones(hrir_l.shape[0])
        hrir_r = np.zeros_like(hrir_r)
        hrir_r[:, 0] = np.ones(hrir_r.shape[0])

    HRIRs = sig.HRIRs(hrir_l, hrir_r, grid, hrir_fs)
    assert HRIRs.fs == fs
    return HRIRs


def load_sdm(filename):
    """
    Convenience function to load SDM.mat.

    Parameters
    ----------
    filename : string
        SDM.mat file

    Returns
    -------
    h : (n,) array_like
        p(t).
    sdm_phi : (n,) array_like
        Azimuth angle.
    sdm_theta : (n,) array_like
        Elevation (colat) angle.
    fs : int
        fs(t).
    """
    mat = loadmat(filename)
    h = np.array(np.squeeze(mat['h_ref']), dtype=float)
    sdm_phi = np.array(np.squeeze(mat['sdm_phi']), dtype=float)
    sdm_theta = np.array(np.squeeze(mat['sdm_theta']), dtype=float)
    fs = int(mat['fs'])
    return h, sdm_phi, sdm_theta, fs


def load_audio(*filenames):
    """
    Convenience function to load multichannel audio from files.

    Parameters
    ----------
    *filename : string, list of strings
        Audio files

    Returns
    -------
    MultiSignal : utils.MultiSignal instance
    """
    loaded_data = []
    loaded_fs = []
    for file in filenames:
        data, fs = sf.read(file)
        if data.ndim != 1:
            # detect and split interleaved wav
            for c in data.T:
                loaded_data.append(c)
        else:
            loaded_data.append(data)
        loaded_fs.append(fs)
    # Assert same sample rate for all channels
    assert all(x == loaded_fs[0] for x in loaded_fs)
    return sig.MultiSignal(*loaded_data, fs=fs)


def load_SOFA_data(filename):
    """Load .sofa file into python dictionary that contains the data in
    numpy arrays."""
    with h5py.File(filename, 'r') as f:
        out_dict = {}
        for key, value in f.items():
            out_dict[key] = np.squeeze(value)
    return out_dict
