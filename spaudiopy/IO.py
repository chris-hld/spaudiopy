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


def load_audio(filenames, fs=None):
    """
    Convenience function to load mono and multichannel audio from files.

    Parameters
    ----------
    filenames : string or list of strings
        Audio files

    Returns
    -------
    sig :  sig.MonoSignal or sig.MultiSignal
        Audio signal.
    """
    loaded_data = []
    loaded_fs = []
    # pack in list if only a single string
    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    for file in filenames:
        data, fs_file = sf.read(file)
        if data.ndim != 1:
            # detect and split interleaved wav
            for c in data.T:
                loaded_data.append(c)
        else:
            loaded_data.append(data)
        loaded_fs.append(fs_file)
    # Assert same sample rate for all channels
    assert all(x == loaded_fs[0] for x in loaded_fs)
    # Check against provided samplerate
    if fs is not None:
        if fs != loaded_fs[0]:
            raise ValueError("File: Found different fs:" + str(loaded_fs[0]))
    else:
        fs = loaded_fs[0]
    # MonoSignal or MultiSignal
    if len(loaded_data) == 1:
        return sig.MonoSignal(loaded_data, fs=fs)
    else:
        return sig.MultiSignal([*loaded_data], fs=fs)


def load_hrirs(fs, filename=None, dummy=False):
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
            [azi: azimuth, colat: colatitude] for hrirs.
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

    try:
        mat = loadmat(filename)
    except FileNotFoundError:
        raise ValueError("No default hrirs. Try running HRIRs_from_SH.py")

    hrir_l = np.array(np.squeeze(mat['hrir_l']), dtype=float)
    hrir_r = np.array(np.squeeze(mat['hrir_r']), dtype=float)
    hrir_fs = int(mat['SamplingRate'])
    azi = np.array(np.squeeze(mat['azi']), dtype=float)
    elev = np.array(np.squeeze(mat['elev']), dtype=float)
    grid = pd.DataFrame({'azi': azi, 'colat': elev})
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
        Colatitude angle.
    fs : int
        fs(t).
    """
    mat = loadmat(filename)
    h = np.array(np.squeeze(mat['h_ref']), dtype=float)
    sdm_phi = np.array(np.squeeze(mat['sdm_phi']), dtype=float)
    sdm_theta = np.array(np.squeeze(mat['sdm_theta']), dtype=float)
    fs = int(mat['fs'])
    return h, sdm_phi, sdm_theta, fs


def load_sofa_data(filename):
    """Load .sofa file into python dictionary that contains the data in
    numpy arrays."""
    with h5py.File(filename, 'r') as f:
        out_dict = {}
        for key, value in f.items():
            out_dict[key] = np.squeeze(value)
    return out_dict
