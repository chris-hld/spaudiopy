# -*- coding: utf-8 -*-
"""
@author: chris
"""
from itertools import repeat

import numpy as np
from joblib import Memory
import multiprocessing

from scipy import signal
from . import utils
from . import sig
from . import process as pcs


# Prepare Caching
cachedir = './__cache_dir'
memory = Memory(cachedir)


def render_stereo_sdm(sdm_p, sdm_phi, sdm_theta):
    """Stereophonic SDM Render IR.

    Parameters
    ----------
    sdm_p : (n,) array_like
        Pressure p(t).
    sdm_phi : (n,) array_like
        Azimuth phi(t).
    sdm_theta : (n,) array_like
        Colatitude theta(t).

    Returns
    -------
    ir_l : array_like
        Left impulse response.
    ir_r : array_like
        Right impulse response.
    """
    ir_l = np.zeros(len(sdm_p))
    ir_r = np.zeros_like(ir_l)

    for i, (p, phi, theta) in enumerate(zip(sdm_p, sdm_phi, sdm_theta)):
        h_l = 0.5*(1 + np.cos(phi - np.pi/2))
        h_r = 0.5*(1 + np.cos(phi + np.pi/2))
        # convolve
        ir_l[i] += p * h_l
        ir_r[i] += p * h_r
    return ir_l, ir_r


def _render_bsdm_sample(i, p, phi, theta, hrirs):
    h_l, h_r = hrirs[hrirs.nearest(phi, theta)]
    # global shared_array
    shared_array[i:i + len(h_l), 0] += p * h_l
    shared_array[i:i + len(h_l), 1] += p * h_r


@memory.cache
def render_bsdm(sdm_p, sdm_phi, sdm_theta, hrirs, jobs_count=None):
    """
    Binaural SDM Render.

    Parameters
    ----------
    sdm_p : (n,) array_like
        Pressure p(t).
    sdm_phi : (n,) array_like
        Azimuth phi(t).
    sdm_theta : (n,) array_like
        Colatitude theta(t).
    hrirs : sig.HRIRs
        'None' selects default hrir set.
    jobs_count : int
        Parallel jobs, switches implementation if > 1.

    Returns
    -------
    bsdm_l : array_like
        Left binaural impulse response.
    bsdm_r : array_like
        Right binaural impulse response.
    """
    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()

    bsdm_l = np.zeros(len(sdm_p) + len(hrirs) - 1)
    bsdm_r = np.zeros_like(bsdm_l)

    if jobs_count == 1:
        for i, (p, phi, theta) in enumerate(zip(sdm_p, sdm_phi, sdm_theta)):
            h_l, h_r = hrirs[hrirs.nearest(phi, theta)]
            # convolve
            bsdm_l[i:i + len(h_l)] += p * h_l
            bsdm_r[i:i + len(h_r)] += p * h_r

    else:
        _shared_array_shape = np.shape(np.c_[bsdm_l, bsdm_r])
        _arr_base = _create_shared_array(_shared_array_shape)
        _arg_itr = zip(range(len(sdm_p)), sdm_p, sdm_phi, sdm_theta,
                       repeat(hrirs))
        # execute
        with multiprocessing.Pool(processes=jobs_count,
                                  initializer=_init_shared_array,
                                  initargs=(_arr_base,
                                            _shared_array_shape,)) as pool:
            pool.starmap(_render_bsdm_sample, _arg_itr)
        # reshape
        _result = np.frombuffer(_arr_base.get_obj()).reshape(
                                _shared_array_shape)
        bsdm_l = _result[:, 0]
        bsdm_r = _result[:, 1]

    return bsdm_l, bsdm_r


def render_loudspeaker_sdm(sdm_p, ls_gains, ls_setup, hrirs):
    """
    Render sdm signal on loudspeaker setup as binaural synthesis.

    Parameters
    ----------
    sdm_p : (n,) array_like
        Pressure p(t).
    ls_gains : (n, l)
        Loudspeaker (l) gains.
    ls_setup : decoder.LoudspeakerSetup
    hrirs : sig.HRIRs

    Returns
    -------
    ir_l : array_like
        Left binaural impulse response.
    ir_r : array_like
        Right binaural impulse response.
    """
    n = len(sdm_p)
    ls_gains = np.atleast_2d(ls_gains)
    assert(n == ls_gains.shape[0])

    # render loudspeaker signals
    ls_sigs = sdm_p * ls_gains.T
    ir_l, ir_r = ls_setup.binauralize(ls_sigs, hrirs.fs, hrirs)
    return ir_l, ir_r


def post_equalization(ls_sigs, sdm_p, fs, blocksize=4096):
    """Post equalization to compensate spectral whitening.

    Parameters
    ----------
    ls_sigs : (L, S) np.ndarray
        Input loudspeaker signals.
    sdm_p : array_like
        Reference (sdm) pressure signal.
    fs : int
    blocksize : int

    Returns
    -------
    ls_sigs_compensated : (L, S) np.ndarray
        Compensated loudspeaker signals.

    References
    ----------
    TERVO, S., et. al. (2015).
    Spatial Analysis and Synthesis of Car Audio System and Car Cabin Acoustics
    with a Compact Microphone Array. Journal of the Audio Engineering Society
    (Vol. 63).
    """
    CHECK_SANITY = False

    hopsize = blocksize // 2
    win = np.hanning(blocksize + 1)[0: -1]

    # prepare Input
    pad = np.zeros([ls_sigs.shape[0], blocksize])
    x_padded = np.hstack([pad, ls_sigs, pad])
    p_padded = np.hstack([np.zeros(blocksize), sdm_p, np.zeros(blocksize)])
    ls_sigs_compensated = np.hstack([pad, np.zeros_like(x_padded), pad])
    assert(len(p_padded) == x_padded.shape[1])

    # prepare filterbank
    filter_gs, ff = pcs.frac_octave_filterbank(n=3, N_out=blocksize//2 + 1,
                                               fs=fs, f_low=100, f_high=12000)
    ntaps = 4096+1
    assert(ntaps % 2), "N does not produce uneven number of filter taps."
    irs = np.zeros([filter_gs.shape[0], ntaps])
    for ir_idx, g_b in enumerate(filter_gs):
        irs[ir_idx, :] = signal.firwin2(ntaps, np.linspace(0, 1, len(g_b)), g_b)

    band_gains_list = []
    start_idx = 0
    while (start_idx + blocksize) <= x_padded.shape[1]:
        if CHECK_SANITY:
            dirac = np.zeros_like(irs)
            dirac[:, blocksize//2] = np.sqrt(1/(irs.shape[0]))

        # blocks
        block_p = win * p_padded[start_idx: start_idx + blocksize]
        block_sdm = win[np.newaxis, :] * x_padded[:, start_idx:
                                                     start_idx + blocksize]

        # block mags
        p_mag = np.sqrt(np.abs(np.fft.rfft(block_p))**2)
        sdm_mag = np.sqrt(np.sum(np.abs(np.fft.rfft(block_sdm, axis=1))**2,
                                 axis=0))
        assert(len(p_mag) == len(sdm_mag))

        # get gains
        L_p = pcs.subband_levels(filter_gs * p_mag, ff[:, 2] - ff[:, 0], fs)
        L_sdm = pcs.subband_levels(filter_gs * sdm_mag, ff[:, 2] - ff[:, 0], fs)

        with np.errstate(divide='ignore', invalid='ignore'):
            band_gains = L_p / L_sdm
        band_gains[np.isnan(band_gains)] = 1
        # clip low shelf to 0
        band_gains[0] = 1 if band_gains[0] > 1 else band_gains[0]
        # gain smoothing over 4 blocks
        if len(band_gains_list) < 4:
            band_gains = band_gains
        else:
            band_gains = 1/4 * (band_gains_list[-3] +
                                band_gains_list[-2] +
                                band_gains_list[-1] +
                                band_gains)
        band_gains_list.append(band_gains)

        for ls_idx in range(ls_sigs.shape[0]):
            # prepare output
            X = np.zeros([irs.shape[0], blocksize + 2 * (irs.shape[1] - 1)])
            # Transform
            for band_idx in range(irs.shape[0]):
                if not CHECK_SANITY:
                    X[band_idx, :blocksize + irs.shape[1] - 1] = \
                        signal.convolve(block_sdm[ls_idx, :], irs[band_idx, :])
                else:
                    X[band_idx, :blocksize + irs.shape[1] - 1] = \
                        signal.convolve(block_sdm[ls_idx, :],
                                        dirac[band_idx, :])
            # Apply gains
            if not CHECK_SANITY:
                X = band_gains[:, np.newaxis] * X
            else:
                X = X

            # Inverse, with zero phase
            for band_idx in range(irs.shape[0]):
                if not CHECK_SANITY:
                    X[band_idx, :] = np.flip(signal.convolve(
                        np.flip(X[band_idx, :blocksize + irs.shape[1] - 1]),
                        irs[band_idx, :]))
                else:
                    X[band_idx, :] = np.flip(signal.convolve(
                        np.flip(X[band_idx, :blocksize + irs.shape[1] - 1]),
                        dirac[band_idx, :]))

            # overlap add
            ls_sigs_compensated[ls_idx,
                                start_idx + blocksize - (irs.shape[1] - 1):
                                start_idx + 2 * blocksize +
                                (irs.shape[1] - 1)] += np.sum(X, axis=0)

        # increase pointer
        start_idx += hopsize


    # restore shape
    if (np.sum(np.abs(ls_sigs_compensated[:, :2 * blocksize])) +
            np.sum(np.abs(ls_sigs_compensated[:, -(2 * blocksize)]))) > 10e-3:
        raise UserWarning('Truncated valid signal, consider more zero padding.')

    ls_sigs_compensated = ls_sigs_compensated[:,
                                              2 * blocksize: -(2 * blocksize)]
    return ls_sigs_compensated, band_gains_list


# Parallel worker stuff -->
def _create_shared_array(shared_array_shape):
    """Allocate ctypes array from shared memory with lock."""
    d_type = 'd'
    shared_array_base = multiprocessing.Array(d_type, shared_array_shape[0] *
                                              shared_array_shape[1])
    return shared_array_base


def _init_shared_array(shared_array_base, shared_array_shape):
    """Makes 'shared_array' available to child processes."""
    global shared_array
    shared_array = np.frombuffer(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shared_array_shape)
# < --Parallel worker stuff
