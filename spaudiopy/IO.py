# -*- coding: utf-8 -*-
"""Input Output (IO) helpers.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

"""

import os
from warnings import warn
import multiprocessing
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat, wavfile
import h5py

import soundfile as sf

from . import utils, sig, decoder, sdm, grids, sph, process, __version__


def load_audio(filenames, fs=None):
    """Load mono and multichannel audio from files.

    Parameters
    ----------
    filenames : string or list of strings
        Audio files.

    Returns
    -------
    sig : sig.MonoSignal or sig.MultiSignal
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


def save_audio(signal, filename, fs=None, subtype='FLOAT'):
    """Save signal to audio file.

    Parameters
    ----------
    signal : sig. MonoSignal, sig.MultiSignal or np.ndarray
        Audio Signal, forwarded to sf.write(); (frames x channels). 
    filename : string
        Audio file name.
    fs : int
        fs(t).
    subtype : optional

    """
    # assert(isinstance(signal, (sig.MonoSignal, sig.MultiSignal)))
    if isinstance(sig, sig.MonoSignal):
        if fs is not None:
            assert(signal.fs == fs)

    if type(signal) == sig.MonoSignal:
        data = signal.signal
        data_fs = signal.fs
    elif type(signal) in (sig.MultiSignal, sig.AmbiBSignal):
        data = signal.get_signals().T
        data_fs = signal.fs
    elif isinstance(signal, (np.ndarray, np.generic)):
        data = signal
        data_fs = fs
    else:
        raise NotImplementedError('Data type not supported.')

    sf.write(filename, data, data_fs, subtype=subtype)


def load_hrirs(fs, filename=None):
    """Convenience function to load 'HRTF.mat'.
    The file contains ['hrir_l', 'hrir_r', 'fs', 'azi', 'colat'].

    Parameters
    ----------
    fs : int
        fs(t).
    filename : string, optional
        HRTF.mat file or default set, or 'dummy' for debugging.

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
    if filename == 'dummy':
        azi, colat, _ = grids.gauss(15)
        grid = pd.DataFrame({'azi': azi, 'colat': colat})
        # Create diracs as dummy
        hrir_l = np.zeros([grid.shape[0], 256])
        hrir_l[:, 0] = np.ones(hrir_l.shape[0])
        hrir_r = np.zeros_like(hrir_l)
        hrir_r[:, 0] = np.ones(hrir_r.shape[0])
        hrir_fs = fs

    elif filename is None:
        # default
        if fs not in [44100, 48000, 96000]:
            raise NotImplementedError('44100, 48000, 96000'
                                      ' default available.')
        default_file = '../data/' + 'HRTF_default_' + str(fs) + '.mat'
        current_file_dir = os.path.dirname(__file__)
        filename = os.path.join(current_file_dir, default_file)

        try:
            mat = loadmat(filename)
        except FileNotFoundError:
            warn("No default hrirs. Generating them...")
            get_default_hrirs()
            mat = loadmat(filename)
    else:
        mat = loadmat(filename)

    if not filename == 'dummy':
        hrir_l = np.array(np.squeeze(mat['hrir_l']), dtype=float)
        hrir_r = np.array(np.squeeze(mat['hrir_r']), dtype=float)
        try:
            hrir_fs = int(mat['fs'])
        except KeyError:
            hrir_fs = int(mat['SamplingRate'])

        azi = np.array(np.squeeze(mat['azi']), dtype=float)
        colat = np.array(np.squeeze(mat['colat']), dtype=float)
        grid = pd.DataFrame({'azi': azi, 'colat': colat})

    HRIRs = sig.HRIRs(hrir_l, hrir_r, grid, hrir_fs)
    assert HRIRs.fs == fs
    return HRIRs


def get_default_hrirs(grid_azi=None, grid_colat=None):
    """Creates the default HRIRs loaded by load_hrirs() by inverse SHT.
    By default it renders onto a gauss grid of order N=35, and additionally
    resamples fs to 48kHz.

    Parameters
    ----------
    grid_azi : array_like, optional
    grid_colat : array_like, optional

    Notes
    -----
    HRTFs in SH domain obtained from
    http://dx.doi.org/10.14279/depositonce-5718.3

    """
    default_file = '../data/0 HRIRs neutral head orientation/' \
                   'SphericalHarmonics/FABIAN_DIR_measured_HATO_0.mat'
    current_file_dir = os.path.dirname(__file__)
    filename = os.path.join(current_file_dir, default_file)
    # %% Load HRTF
    try:
        file = loadmat(filename)

    except FileNotFoundError:
        import requests, zipfile, io
        print("Downloading from https://depositonce.tu-berlin.de/handle/"
              "11303/6153.3 ...")
        r = requests.get('https://depositonce.tu-berlin.de/bitstream/11303/'
                         '6153.3/8/FABIAN_HRTFs_NeutralHeadOrientation.zip')
        with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
            zip_ref.extractall(os.path.join(current_file_dir, '../data/'))
        file = loadmat(filename)
    # CTF already compensated in DIR
    # Extracting the data is a bit ugly here...
    SamplingRate = int(file['SamplingRate'])
    SH_l = file['SH'][0][0][0]
    SH_r = file['SH'][0][0][1]
    f = np.squeeze(file['SH'][0][0][5])

    # default grid:
    if (grid_azi is None) and (grid_colat is None):
        grid_azi, grid_colat, _ = grids.gauss(35)  # grid positions

    # %% Inverse SHT
    HRTF_l = sph.inverse_sht(SH_l, grid_azi, grid_colat, 'complex')
    HRTF_r = sph.inverse_sht(SH_r, grid_azi, grid_colat, 'complex')
    assert HRTF_l.shape == HRTF_r.shape
    # %%
    hrir_l = np.fft.irfft(HRTF_l)  # creates 256 samples(t)
    hrir_r = np.fft.irfft(HRTF_r)  # creates 256 samples(t)
    assert hrir_l.shape == hrir_r.shape

    # %% Resample
    fs_target = 48000
    hrir_l_48k, hrir_r_48k, _ = process.resample_hrirs(hrir_l, hrir_r,
                                                       SamplingRate,
                                                       fs_target)
    fs_target = 96000
    hrir_l_96k, hrir_r_96k, _ = process.resample_hrirs(hrir_l, hrir_r,
                                                       SamplingRate,
                                                       fs_target)

    savemat(os.path.join(current_file_dir, '../data/HRTF_default_44100.mat'),
            {'hrir_l': hrir_l,
             'hrir_r': hrir_r,
             'azi': grid_azi, 'colat': grid_colat,
             'fs': 44100})
    savemat(os.path.join(current_file_dir, '../data/HRTF_default_48000.mat'),
            {'hrir_l': hrir_l_48k,
             'hrir_r': hrir_r_48k,
             'azi': grid_azi, 'colat': grid_colat,
             'fs': 48000})
    savemat(os.path.join(current_file_dir, '../data/HRTF_default_96000.mat'),
            {'hrir_l': hrir_l_96k,
             'hrir_r': hrir_r_96k,
             'azi': grid_azi, 'colat': grid_colat,
             'fs': 96000})
    print("Saved new default HRIRs.")


def load_sdm(filename, init_nan=True):
    """Convenience function to load 'SDM.mat'.
    The file contains
    ['h_ref' or 'p', 'sdm_azi' or 'sdm_phi', 'sdm_colat' or 'sdm_theta', 'fs'].

    Parameters
    ----------
    filename : string
        SDM.mat file
    init_nan : bool, optional
        Initialize nan to [0, pi/2].

    Returns
    -------
    h : (n,) array_like
        p(t).
    sdm_azi : (n,) array_like
        Azimuth angle.
    sdm_colat : (n,) array_like
        Colatitude angle.
    fs : int
        fs(t).

    """
    mat = loadmat(filename)
    try:
        h = np.array(np.squeeze(mat['h_ref']), dtype=float)
    except KeyError:
        h = np.array(np.squeeze(mat['p']), dtype=float)
    try:
        sdm_azi = np.array(np.squeeze(mat['sdm_azi']), dtype=float)
    except KeyError:
        sdm_azi = np.array(np.squeeze(mat['sdm_phi']), dtype=float)
    try:
        sdm_colat = np.array(np.squeeze(mat['sdm_colat']), dtype=float)
    except KeyError:
        sdm_colat = np.array(np.squeeze(mat['sdm_theta']), dtype=float)

    if init_nan:
        sdm_azi[np.isnan(sdm_azi)] = 0.
        sdm_colat[np.isnan(sdm_colat)] = np.pi / 2
    fs = int(mat['fs'])
    return h, sdm_azi, sdm_colat, fs


def load_sofa_data(filename):
    """Load .sofa file into python dictionary that contains the data in
    numpy arrays."""
    with h5py.File(filename, 'r') as f:
        out_dict = {}
        for key, value in f.items():
            out_dict[key] = np.squeeze(value)
    return out_dict


def write_ssr_brirs_loudspeaker(filename, ls_irs, hull, fs, subtype='FLOAT',
                                hrirs=None, jobs_count=1):
    """Write binaural room impulse responses (BRIRs) and save as wav file.

    The azimuth resolution is one degree. The channels are interleaved and
    directly compatible to the SoundScape Renderer (SSR) ssr-brs.

    Parameters
    ----------
    filename : string
    ls_irs : (L, S) np.ndarray
        Impulse responses of L loudspeakers,
        e.g. by hull.loudspeaker_signals().
    hull : decoder.LoudspeakerSetup
    fs : int
    subtype : forwarded to sf.write(), optional
    hrirs : sig.HRIRs, optional
    jobs_count : int, optional
        [CPU Cores], Number of Processes, switches implementation for n > 1.

    """
    if hrirs is None:
        hrirs = load_hrirs(fs=fs)
    assert(hrirs.fs == fs)

    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()

    if not filename[-4:] == '.wav':
        filename = filename + '.wav'

    ssr_brirs = np.zeros((720, ls_irs.shape[1] + len(hrirs) - 1))

    if jobs_count == 1:
        for angle in range(0, 360):
            ir_l, ir_r = hull.binauralize(ls_irs, fs,
                                          orientation=(np.deg2rad(angle), 0),
                                          hrirs=hrirs)
            # left
            ssr_brirs[2 * angle, :] = ir_l
            # right
            ssr_brirs[2 * angle + 1, :] = ir_r

    elif jobs_count > 1:
        with multiprocessing.Pool(processes=jobs_count) as pool:
            results = pool.starmap(hull.binauralize,
                                   map(lambda a: (ls_irs, fs,
                                                  (np.deg2rad(a), 0),
                                                  hrirs),
                                       range(0, 360)))
        # extract
        ir_l = [ir[0] for ir in results]
        ir_r = [ir[1] for ir in results]
        for angle in range(0, 360):
            # left
            ssr_brirs[2 * angle, :] = ir_l[angle]
            # right
            ssr_brirs[2 * angle + 1, :] = ir_r[angle]

    # normalize
    if np.max(np.abs(ssr_brirs)) > 1:
        warn('Normalizing BRIRs')
        ssr_brirs = ssr_brirs / np.max(np.abs(ssr_brirs))

    # write to file
    save_audio(ssr_brirs.T, filename, fs, subtype=subtype)


def write_ssr_brirs_sdm(filename, sdm_p, sdm_phi, sdm_theta, fs,
                        subtype='FLOAT', hrirs=None):
    """Write binaural room impulse responses (BRIRs) and save as wav file.

    The azimuth resolution is one degree. The channels are interleaved and
    directly compatible to the SoundScape Renderer (SSR) ssr-brs.

    Parameters
    ----------
    filename : string
    sdm_p : (n,) array_like
        Pressure p(t).
    sdm_phi : (n,) array_like
        Azimuth phi(t).
    sdm_theta : (n,) array_like
        Colatitude theta(t).
    fs : int
    subtype : forwarded to sf.write(), optional
    hrirs : sig.HRIRs, optional

    """
    if hrirs is None:
        hrirs = load_hrirs(fs=fs)
    assert(hrirs.fs == fs)

    if not filename[-4:] == '.wav':
        filename = filename + '.wav'

    ssr_brirs = np.zeros((720, len(sdm_p) + len(hrirs) - 1))
    for angle in range(0, 360):
        sdm_phi_rot = sdm_phi - np.deg2rad(angle)
        ir_l, ir_r = sdm.render_bsdm(sdm_p, sdm_phi_rot, sdm_theta,
                                     hrirs=hrirs)
        # left
        ssr_brirs[2 * angle, :] = ir_l
        # right
        ssr_brirs[2 * angle + 1, :] = ir_r

    # normalize
    if np.max(np.abs(ssr_brirs)) > 1:
        warn('Normalizing BRIRs')
        ssr_brirs = ssr_brirs / np.max(np.abs(ssr_brirs))

    # write to file
    save_audio(ssr_brirs.T, filename, fs, subtype=subtype)


def load_layout(filename, N_kernel=50):
    """Load loudspeaker layout from json configuration file."""
    with open(filename, 'r') as f:
        in_data = json.load(f)

    layout = in_data['LoudspeakerLayout']
    ls_data = layout['Loudspeakers']

    azi = np.array([ls['Azimuth'] for ls in ls_data])
    ele = np.array([ls['Elevation'] for ls in ls_data])
    r = np.array([ls['Radius'] for ls in ls_data])
    try:
        # not actually used, yet
        gain = np.array([ls['Gain'] for ls in ls_data])
        if np.any(gain != 1.):
            warn('Additional gain handling not implemented.')
    except KeyError as e:
        warn('KeyError : {}, will return empty!'.format(e))
        gain = []
    try:
        isImaginary = np.array([ls['IsImaginary'] for ls in ls_data])
    except KeyError as e:
        warn('KeyError : {}, will return all False!'.format(e))
        isImaginary = np.full_like(azi, False, dtype=bool)

    # first extract real loudspeakers
    ls_x, ls_y, ls_z = utils.sph2cart(utils.deg2rad(azi[~isImaginary]),
                                      utils.deg2rad(90-ele[~isImaginary]),
                                      r[~isImaginary])

    ls_layout = decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
    # then add imaginary loudspeakers to ambisonics setup
    imag_x, imag_y, imag_z = utils.sph2cart(utils.deg2rad(azi[isImaginary]),
                                            utils.deg2rad(90-ele[isImaginary]),
                                            r[isImaginary])
    imag_pos = np.c_[imag_x, imag_y, imag_z]
    ls_layout.ambisonics_setup(N_kernel=N_kernel, update_hull=True,
                               imaginary_ls=imag_pos)
    return ls_layout


def save_layout(filename, ls_layout, name='unknown', description='unknown'):
    """Save loudspeaker layout to json configuration file."""
    if not ls_layout.ambisonics_hull:
        raise ValueError("No ambisonics_hull.")
    out_data = {}
    out_data['Name'] = name
    out_data['Description'] = 'This configuration file was created with ' +\
                              'spaudiopy (v-' + str(__version__) + '), ' + \
                              str(datetime.now())

    out_data['LoudspeakerLayout'] = {}
    out_data['LoudspeakerLayout']['Name'] = name
    out_data['LoudspeakerLayout']['Description'] = description
    out_data['LoudspeakerLayout']['Loudspeakers'] = []

    for ls_idx in range(ls_layout.ambisonics_hull.npoints):
        ls_dirs = utils.cart2sph(ls_layout.ambisonics_hull.x[ls_idx],
                                 ls_layout.ambisonics_hull.y[ls_idx],
                                 ls_layout.ambisonics_hull.z[ls_idx])
        ls_dict = {}
        ls_dict['Azimuth'] = round(float(utils.rad2deg(ls_dirs[0])), 2)
        ls_dict['Elevation'] = round(float(90 - utils.rad2deg(ls_dirs[1])), 2)
        ls_dict['Radius'] = round(float(ls_dirs[2]), 2)
        ls_dict['IsImaginary'] = ls_idx in np.asarray(
                                    ls_layout.ambisonics_hull.imaginary_ls_idx)
        ls_dict['Channel'] = ls_idx + 1
        ls_dict['Gain'] = 0. if ls_idx in np.asarray(
                            ls_layout.ambisonics_hull.imaginary_ls_idx) else 1.
        out_data['LoudspeakerLayout']['Loudspeakers'].append(ls_dict)

    with open(filename, 'w') as outfile:
        json.dump(out_data, outfile, indent=4)
