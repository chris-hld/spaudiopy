# -*- coding: utf-8 -*-
"""
@author: chris
"""
from itertools import repeat
from warnings import warn

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


def render_loudspeaker_sdm(sdm_p, ls_gains, ls_setup, hrirs,
                           orientation=(0, 0)):
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
    orientation : (azi, colat) tuple, optional
        Listener orientation offset (azimuth, colatitude) in rad.

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
    ir_l, ir_r = ls_setup.binauralize(ls_sigs, hrirs.fs,
                                      orientation=orientation, hrirs=hrirs)
    return ir_l, ir_r


def post_equalization(ls_sigs, sdm_p, fs, ls_distance=None, amp_decay=1,
                      soft_clip=True):
    """Post equalization to compensate spectral whitening.

    Parameters
    ----------
    ls_sigs : (L, S) np.ndarray
        Input loudspeaker signals.
    sdm_p : array_like
        Reference (sdm) pressure signal.
    fs : int
    ls_distance : (L,) array_like, optional
        Loudspeaker distances in m.
    amp_decay : float
        Distance attenuation exponent.
    soft_clip : bool, optional
        Limit the compensation boost to +6dB.

    Returns
    -------
    ls_sigs_compensated : (L, S) np.ndarray
        Compensated loudspeaker signals.

    References
    ----------
    Tervo, S., et. al. (2015).
    Spatial Analysis and Synthesis of Car Audio System and Car Cabin Acoustics
    with a Compact Microphone Array. Journal of the Audio Engineering Society.

    """
    if ls_distance is None:
        ls_distance = np.ones(ls_sigs.shape[0])
    a = amp_decay  # amplitude decay exponent

    CHECK_SANITY = False

    # prepare filterbank
    filter_gs, ff = pcs.frac_octave_filterbank(n=1, N_out=2**16,
                                               fs=fs, f_low=62.5, f_high=16000,
                                               mode='pressure')


    # band dependent block size
    band_blocksizes = np.zeros(ff.shape[0])
    # proposed by Tervo
    band_blocksizes[1:] = np.round(7 / ff[1:, 0] * fs)
    band_blocksizes[0] = np.round(7 / ff[0, 1] * fs)
    # make sure they are even
    band_blocksizes = (np.ceil(band_blocksizes / 2) * 2).astype(int)

    padsize = band_blocksizes.max()

    ntaps = padsize // 2 - 1
    assert(ntaps % 2), "N does not produce uneven number of filter taps."
    irs = np.zeros([filter_gs.shape[0], ntaps])
    for ir_idx, g_b in enumerate(filter_gs):
        irs[ir_idx, :] = signal.firwin2(ntaps, np.linspace(0, 1, len(g_b)),
                                        g_b)


    # prepare Input
    pad = np.zeros([ls_sigs.shape[0], padsize])
    x_padded = np.hstack([pad, ls_sigs, pad])
    p_padded = np.hstack([np.zeros(padsize), sdm_p, np.zeros(padsize)])
    ls_sigs_compensated = np.hstack([pad, np.zeros_like(x_padded), pad])
    ls_sigs_band = np.zeros([ls_sigs_compensated.shape[0],
                             ls_sigs_compensated.shape[1],
                             irs.shape[0]])
    assert(len(p_padded) == x_padded.shape[1])

    for band_idx in range(irs.shape[0]):
        blocksize = band_blocksizes[band_idx]
        hopsize = blocksize // 2
        win = np.hanning(blocksize + 1)[0: -1]
        start_idx = 0
        while (start_idx + blocksize) <= x_padded.shape[1]:
            if CHECK_SANITY:
                dirac = np.zeros_like(irs)
                dirac[:, blocksize // 2] = np.sqrt(1/(irs.shape[0]))

            # blocks
            block_p = win * p_padded[start_idx: start_idx + blocksize]
            block_sdm = win[np.newaxis, :] * x_padded[:, start_idx:
                                                      start_idx + blocksize]

            # block spectra
            nfft = blocksize + blocksize - 1

            H_p = np.fft.fft(block_p, nfft)
            H_sdm = np.fft.fft(block_sdm, nfft, axis=1)
            # distance
            spec_in_origin = np.diag(1 / ls_distance**a) @ H_sdm

            # magnitude difference by spectral division
            sdm_mag_incoherent = np.sqrt(np.sum(np.abs(spec_in_origin)**2,
                                                axis=0))
            sdm_mag_coherent = np.sum(np.abs(spec_in_origin), axis=0)

            # Coherent addition in the lows
            if band_idx == 0:
                mag_diff = np.abs(H_p) / \
                            np.clip(sdm_mag_coherent, 10e-10, None)
            elif band_idx == 1:
                mag_diff = np.abs(H_p) / \
                           (0.5 * np.clip(sdm_mag_coherent, 10e-10, None) +
                            0.5 * np.clip(sdm_mag_incoherent, 10e-10, None))
            elif band_idx == 2:
                mag_diff = np.abs(H_p) / \
                           (0.25 * np.clip(sdm_mag_coherent, 10e-10, None) +
                            0.75 * np.clip(sdm_mag_incoherent, 10e-10, None))
            else:
                mag_diff = np.abs(H_p) / np.clip(sdm_mag_incoherent, 10e-10,
                                                 None)
            # soft clip gain
            if soft_clip:
                mag_diff[mag_diff > 1] = 1 + \
                                          np.tanh(mag_diff[mag_diff > 1] - 1)

            # apply to ls input
            Y = H_sdm * mag_diff[np.newaxis, :]

            # inverse STFT
            X = np.real(np.fft.ifft(Y, axis=1))
            # Zero Phase
            assert(np.mod(X.shape[1], 2))
            # delay
            zp_delay = X.shape[1] // 2
            X = np.roll(X, zp_delay, axis=1)

            # overlap add
            ls_sigs_band[:, padsize + start_idx - zp_delay:
                            padsize + start_idx - zp_delay + nfft,
                         band_idx] += X

            # increase pointer
            start_idx += hopsize

        # apply filter
        for ls_idx in range(ls_sigs.shape[0]):
            ls_sigs_band[ls_idx, :, band_idx] = signal.convolve(ls_sigs_band[
                                                                ls_idx, :,
                                                                band_idx],
                                                                irs[band_idx],
                                                                mode='same')

    # sum over bands
    ls_sigs_compensated = np.sum(ls_sigs_band, axis=2)

    # restore shape
    out_start_idx = int(2 * padsize)
    out_end_idx = int(-(2 * padsize))
    if np.any(np.abs(ls_sigs_compensated[:, :out_start_idx]) > 10e-5) or \
            np.any(np.abs(ls_sigs_compensated[:, -out_end_idx]) > 10e-5):
        warn('Truncated valid signal, consider more zero padding.')

    ls_sigs_compensated = ls_sigs_compensated[:, out_start_idx: out_end_idx]
    assert(ls_sigs_compensated.shape == ls_sigs.shape)
    return ls_sigs_compensated


def post_equalization2(ls_sigs, sdm_p, fs, ls_distance=None, amp_decay=1,
                       blocksize=4096, smoothing_order=5):
    """Post equalization to compensate spectral whitening. This alternative
    version works on fixed blocksizes with octave band gain smoothing.
    Sonically, this is not the preferred version, but it can gain some insight
    through the band gains with are returned.

    Parameters
    ----------
    ls_sigs : (L, S) np.ndarray
        Input loudspeaker signals.
    sdm_p : array_like
        Reference (sdm) pressure signal.
    fs : int
    ls_distance : (L,) array_like, optional
        Loudspeaker distances in m.
    blocksize : int
    smoothing_order : int
        Block smoothing, increasing Hanning window up to this order.

    Returns
    -------
    ls_sigs_compensated : (L, S) np.ndarray
        Compensated loudspeaker signals.
    band_gains_list : list
        Each element contains the octave band gain applied as post eq.

    References
    ----------
    Tervo, S., et. al. (2015).
    Spatial Analysis and Synthesis of Car Audio System and Car Cabin Acoustics
    with a Compact Microphone Array. Journal of the Audio Engineering Society.
    (Vol. 63).

    """
    if ls_distance is None:
        ls_distance = np.ones(ls_sigs.shape[0])
    a = amp_decay  # amplitude decay exponent

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
    filter_gs, ff = pcs.frac_octave_filterbank(n=1, N_out=blocksize//2 + 1,
                                               fs=fs, f_low=62.5, f_high=16000)
    ntaps = blocksize+1
    assert(ntaps % 2), "N does not produce uneven number of filter taps."
    irs = np.zeros([filter_gs.shape[0], ntaps])
    for ir_idx, g_b in enumerate(filter_gs):
        irs[ir_idx, :] = signal.firwin2(ntaps, np.linspace(0, 1, len(g_b)),
                                        g_b)

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
        sdm_H = np.diag(1 / ls_distance**a) @ np.fft.rfft(block_sdm, axis=1)
        sdm_mag_incoherent = np.sqrt(np.sum(np.abs(sdm_H)**2, axis=0))
        sdm_mag_coherent = np.sum(np.abs(sdm_H), axis=0)
        assert(len(p_mag) == len(sdm_mag_incoherent) == len(sdm_mag_coherent))

        # get gains
        L_p = pcs.subband_levels(filter_gs * p_mag, ff[:, 2] - ff[:, 0], fs)
        L_sdm_incoherent = pcs.subband_levels(filter_gs * sdm_mag_incoherent,
                                              ff[:, 2] - ff[:, 0], fs)
        L_sdm_coherent = pcs.subband_levels(filter_gs * sdm_mag_coherent,
                                            ff[:, 2] - ff[:, 0], fs)
        with np.errstate(divide='ignore', invalid='ignore'):
            band_gains_incoherent = L_p / L_sdm_incoherent
            band_gains_coherent = L_p / L_sdm_coherent

        band_gains_incoherent[np.isnan(band_gains_incoherent)] = 1
        band_gains_coherent[np.isnan(band_gains_coherent)] = 1
        # clip gains
        gain_clip = 1
        band_gains_incoherent = np.clip(band_gains_incoherent, None, gain_clip)
        band_gains_coherent = np.clip(band_gains_coherent, None, gain_clip)

        # attenuate lows (coherent)
        band_gains = np.zeros_like(band_gains_coherent)
        band_gains[0] = band_gains_coherent[0]
        band_gains[1] = 0.5 * band_gains_coherent[1] + \
                        0.5 * band_gains_incoherent[1]
        band_gains[2] = 0.25 * band_gains_coherent[2] + \
                        0.75 * band_gains_incoherent[2]
        band_gains[3:] = band_gains_incoherent[3:]

        # gain smoothing over blocks
        if len(band_gains_list) > 0:
            # half-sided window, increasing in size
            current_order = min(smoothing_order, len(band_gains_list))
            w = np.hanning(current_order * 2 + 1)[-(current_order + 1): -1]
            # normalize
            w = w / w.sum()
            band_gains_smoothed = w[0] * band_gains  # current
            for order_idx in range(1, current_order):
                band_gains_smoothed += w[order_idx] * \
                                       band_gains_list[-order_idx]
        else:
            band_gains_smoothed = band_gains

        band_gains_list.append(band_gains_smoothed)

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
    out_start_idx = 2 * blocksize
    out_end_idx = -(2 * blocksize)
    if (np.sum(np.abs(ls_sigs_compensated[:, :out_start_idx])) +
            np.sum(np.abs(ls_sigs_compensated[:, -out_end_idx]))) > 10e-3:
        warn('Truncated valid signal, consider more zero padding.')

    ls_sigs_compensated = ls_sigs_compensated[:, out_start_idx: out_end_idx]
    assert(ls_sigs_compensated.shape == ls_sigs.shape)
    return ls_sigs_compensated, band_gains_list[2:-2]


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
