# -*- coding: utf-8 -*-
"""Parametric Spatial Audio (PARSA).

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

    N_sph = 3
    # Three sources
    x_nm = spa.sph.src_to_sh(np.random.randn(3, 10000),
                            [np.pi/2, -np.pi/4, np.pi/3],
                            [np.pi/3, np.pi/2, 2/3 * np.pi], N_sph)
    # Diffuse noise
    x_nm += np.sqrt(16/(4*np.pi)) * np.random.randn(16, 10000)
    spa.plot.sh_rms_map(x_nm, title="Input SHD Signal")

**Memory cached functions**

.. autofunction:: spaudiopy.parsa.pseudo_intensity(ambi_b, win_len=33, f_bp=None, smoothing_order=5, jobs_count=1)
.. autofunction:: spaudiopy.parsa.render_bsdm(sdm_p, sdm_phi, sdm_theta, hrirs, jobs_count=None)

"""

from itertools import repeat
from warnings import warn
import logging

import numpy as np
from joblib import Memory
import multiprocessing

from scipy import signal
from . import utils, sph
from . import process as pcs


# Prepare Caching
cachedir = './.spa_cache_dir'
memory = Memory(cachedir)
shared_array = None
lock = multiprocessing.RLock()


def sh_beamformer_from_pattern(pattern, N_sph, azi_steer, zen_steer):
    """Get spherical harmonics domain (SHD) beamformer coefficients.

    Parameters
    ----------
    pattern : string , or (N+1, ) array_like
        Pattern description, e.g. `'cardioid'` or modal weights.
    N_sph : int
        SH order.
    azi_steer : (J,) array_like
        Azimuth steering directions.
    zen_steer : (J,) array_like
        Zenith/colatitude steering directions.

    Returns
    -------
    w_nm : (J, (N+1)**2) numpy.ndarray
        SHD Beamformer weights.

    Examples
    --------
    See :py:func:`spaudiopy.parsa.sh_beamform`.

    """
    if isinstance(pattern, str):
        if pattern.lower() in ['hypercardioid', 'max_di']:
            c_n = sph.hypercardioid_modal_weights(N_sph)
        elif pattern.lower() in ['cardioid', 'inphase']:
            c_n = sph.cardioid_modal_weights(N_sph)
        elif pattern.lower() in ['max_re', 'maxre']:
            c_n = sph.maxre_modal_weights(N_sph)
        else:
            raise ValueError("Pattern not available: " + pattern)
    else:
        c_n = utils.asarray_1d(pattern)
    assert len(c_n) == (N_sph+1), "Input not matching:" + str(c_n)

    w_nm = sph.repeat_per_order(c_n)
    Y_steer = sph.sh_matrix(N_sph, azi_steer, zen_steer, sh_type='real')
    return w_nm * Y_steer


def sh_beamform(w_nm, sig_nm):
    """Apply spherical harmonics domain (SHD) beamformer.

    Parameters
    ----------
    w_nm : ((N+1)**2,) array_like, or (J, (N+1)**2) np.ndarray
        SHD beamformer weights (for `J` beamformers)
    sig_nm : ((N+1)**2, l) np.ndarray
        SHD signal of length l.

    Returns
    -------
    y : (J, l) np.ndarray
        Beamformer output signals.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, _ = spa.grids.load_maxDet(50)
        dirs = spa.utils.vec2dir(vecs)
        w_nm = spa.parsa.sh_beamformer_from_pattern('cardioid', N_sph,
                                                    dirs[:,0], dirs[:,1])
        y = spa.parsa.sh_beamform(w_nm, x_nm)
        spa.plot.spherical_function_map(spa.utils.rms(y), dirs[:,0], dirs[:,1],
                                        TODB=True, title="Output RMS")

    """
    W = np.atleast_2d(w_nm)
    sig_nm = np.asarray(sig_nm)
    if sig_nm.ndim == 1:
        sig_nm = sig_nm[:, np.newaxis]  # upgrade to handle 1D arrays
    return W @ sig_nm


def estimate_num_sources(cov_x, a=None, w=None):
    """Active source count estimate from signal covariance.

    Based on the relation of consecutive eigenvalues.

    Parameters
    ----------
    cov_x : (L, L) numpy.2darray
        Signal covariance.
    a : float, optional
        Threshold condition (ratio), defaults to `1 + 2/len(cov_x)`
    w : (L,) array_like, optional
        Eigenvalues in ascending order, not using `cov_x` if available.

    Returns
    -------
    num_src_est : int
        Number of active sources estimate.

    Examples
    --------
    See :py:func:`spaudiopy.parsa.sh_music`.

    """
    if w is None:
        w = np.linalg.eigvalsh(cov_x)
    else:
        w = utils.asarray_1d(w)
    if a is None:
        a = 1 + 2/len(w)
    if np.var(w) < a:
        num_src_est = 0
    else:
        c = w[1:] / (w[:-1] + 10e-8)
        cn = np.argmax(c > a)
        num_src_est = len(w)-1 - cn
    return num_src_est


def separate_cov(cov_x, num_cut=None):
    """Separate Covariance matrix in signal and noise components.

    Parameters
    ----------
    S_xx : (L, L) numpy.2darray
        Covariance.
    num_cut : int, optional
        Split point of Eigenvalues, default: `parsa.estimate_num_sources()`.

    Returns
    -------
    S_pp : (L, L) numpy.2darray
        Signal covariance.
    S_nn : (L, L) numpy.2darray
        Noise (residual) covariance.

    Notes
    -----
    Signal model is :math:`S_x = S_p + S_n` .

    Examples
    --------
    .. plot::
        :context: close-figs

        S_xx = x_nm @ x_nm.T
        S_pp, S_nn = spa.parsa.separate_cov(S_xx, num_cut=3)
        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        axs[0].matshow(S_xx)
        axs[0].set_title("X")
        axs[1].matshow(S_pp)
        axs[1].set_title("S")
        axs[2].matshow(S_nn)
        axs[2].set_title("N")

    """
    assert (cov_x.shape[0] == cov_x.shape[1])
    w, v = np.linalg.eigh(cov_x)
    if num_cut is None:
        num_cut = estimate_num_sources([], w=w)

    w_nn = 1. * w
    w_r = w[-(num_cut+1)]
    w_nn[-num_cut:] = w_r
    S_nn = v @ np.diag(w_nn) @ v.T
    S_pp = v[:, -num_cut:] @ (np.diag(w[-num_cut:] - w_r)) @ v[:, -num_cut:].T
    return S_pp, S_nn


def sh_music(cov_x, num_src, dirs_azi, dirs_zen):
    """SH domain / Eigenbeam Multiple Signal Classification (EB-MUSIC).

    Parameters
    ----------
    cov_x : (L, L) numpy.2darray
        SH signal covariance.
    num_src : int
        Number of sources.
    dirs_azi : (g,) array_like
    dirs_zen : (g,) array_like

    Returns
    -------
    P_music : (g,) array_like
        MUSIC (psuedo-) spectrum.

    Examples
    --------
    .. plot::
        :context: close-figs

        S_xx = x_nm @ x_nm.T
        num_src_est = spa.parsa.estimate_num_sources(S_xx)

        vecs, _ = spa.grids.load_maxDet(50)
        dirs = spa.utils.vec2dir(vecs)
        P_music = spa.parsa.sh_music(S_xx, num_src_est, dirs[:,0], dirs[:,1])
        spa.plot.spherical_function_map(P_music, dirs[:,0], dirs[:,1],
                                        TODB=True, title="MUSIC spectrum")

    """
    assert (cov_x.shape[0] == cov_x.shape[1])
    N_sph = int(np.sqrt(cov_x.shape[0]) - 1)
    dirs_azi = utils.asarray_1d(dirs_azi)
    dirs_zen = utils.asarray_1d(dirs_zen)
    Y = sph.sh_matrix(N_sph, dirs_azi, dirs_zen, sh_type='real')
    _, v = np.linalg.eigh(cov_x)
    Qn = v[:, :-num_src]
    a = (Qn.T @ Y.T)
    P_music = 1 / (np.sum(a * a, 0) + 10e-12)
    return P_music


def sh_mvdr(cov_x, dirs_azi, dirs_zen):
    """Spherical Harmonics domain MVDR beamformer.
    SH / Eigenbeam domain minimum variance distortionless response (EB-MVDR).
    Often employed on signal `cov_x = S_xx`, instead of noise `cov_x = S_nn`,
    then called minimum power distortionless response (MPDR) beamformer.

    Parameters
    ----------
    cov_x : (L, L) numpy.2darray
        SH signal (noise) covariance.
    dirs_azi : (g,) array_like
    dirs_zen : (g,) array_like

    Returns
    -------
    W_nm : (g, L) numpy.2darray
        MVDR beampattern weights.

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
    ch. 7.2.

    Examples
    --------
    .. plot::
        :context: close-figs

        S_xx = x_nm @ x_nm.T
        num_src_est = spa.parsa.estimate_num_sources(S_xx)
        _, S_nn = spa.parsa.separate_cov(S_xx, num_cut=num_src_est)

        vecs, _ = spa.grids.load_maxDet(50)
        dirs = spa.utils.vec2dir(vecs)
        W_nm = spa.parsa.sh_mvdr(S_nn, dirs[:,0], dirs[:,1])
        y = spa.parsa.sh_beamform(W_nm, x_nm)
        spa.plot.spherical_function_map(spa.utils.rms(y), dirs[:,0], dirs[:,1],
                                        TODB=True, title="MVDR output RMS")

    """
    assert (cov_x.shape[0] == cov_x.shape[1])
    N_sph = int(np.sqrt(cov_x.shape[0]) - 1)
    dirs_azi = utils.asarray_1d(dirs_azi)
    dirs_zen = utils.asarray_1d(dirs_zen)
    Y_steer = sph.sh_matrix(N_sph, dirs_azi, dirs_zen, sh_type='real')
    S_inv = np.linalg.inv(cov_x)
    c = Y_steer @ S_inv
    a = Y_steer @ S_inv @ Y_steer.conj().T
    W_nm = c.T / np.diag(a)
    return W_nm.T


def sh_lcmv(cov_x, dirs_azi_c, dirs_zen_c, c_gain):
    """Spherical Harmonics domain LCMV beamformer.
    SH / Eigenbeam domain Linearly Constrained Minimum Variance (LCMV)
    beamformer.
    Often employed on signal `cov_x = S_xx`, instead of noise `cov_x = S_nn`,
    then called linearly constrained minimum power (LCMP) beamformer.

    Parameters
    ----------
    cov_x : (L, L) numpy.2darray
        SH signal (noise) covariance.
    dirs_azi : (g,) array_like
    dirs_zen : (g,) array_like
    c_gain : (g,) array_like
        Constraints (gain) on points `[dirs_azi, dirs_zen]`.

    Returns
    -------
    w_nm : (L,) array_like
        LCMV beampattern weights.

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
    ch. 7.5.

    Examples
    --------
    .. plot::
        :context: close-figs

        S_xx = x_nm @ x_nm.T
        num_src_est = spa.parsa.estimate_num_sources(S_xx)
        _, S_nn = spa.parsa.separate_cov(S_xx, num_cut=num_src_est)

        dirs_azi_c = [np.pi/2, 0., np.pi]
        dirs_zen_c = [np.pi/2, np.pi/2, np.pi/4]
        c = [1, 0.5, 0]
        w_nm = spa.parsa.sh_lcmv(S_nn, dirs_azi_c, dirs_zen_c, c)
        spa.plot.sh_coeffs(w_nm)

    """
    assert (cov_x.shape[0] == cov_x.shape[1])
    dirs_azi_c = utils.asarray_1d(dirs_azi_c)
    dirs_zen_c = utils.asarray_1d(dirs_zen_c)
    c_gain = utils.asarray_1d(c_gain)

    assert (len(dirs_azi_c) == len(dirs_zen_c))
    assert (len(dirs_azi_c) == len(c_gain))
    N_sph = int(np.sqrt(cov_x.shape[0]) - 1)
    V = sph.sh_matrix(N_sph, dirs_azi_c, dirs_zen_c, sh_type='real').T
    S_inv = np.linalg.inv(cov_x)
    w_nm = c_gain.T @ np.linalg.inv(V.T @ S_inv @ V) @ V.T @ S_inv
    return w_nm


def sh_sector_beamformer(A_nm):
    """
    Get sector pressure and intensity beamformers.

    Parameters
    ----------
    A_nm : (J, (N+1)**2), np.ndarray
        SH beamformer matrix, see spa.sph.design_sph_filterbank().

    Returns
    -------
    A_wxyz : ((4*J), (N+2)**2)
        SH sector pattern beamformers.

    """
    num_sec = A_nm.shape[0]
    A_wxyz = np.zeros((4*num_sec, int(np.sqrt(A_nm.shape[1])+1)**2))

    w_nm = np.sqrt(4*np.pi) * np.array([1, 0, 0, 0])
    x_nm = np.sqrt(4/3*np.pi) * np.array([0, 0, 0, 1])
    y_nm = np.sqrt(4/3*np.pi) * np.array([0, 1, 0, 0])
    z_nm = np.sqrt(4/3*np.pi) * np.array([0, 0, 1, 0])
    for idx_s in range(num_sec):
        A_wxyz[idx_s*4+0, :] = sph.sh_mult(w_nm, A_nm[idx_s, :], 'real')
        A_wxyz[idx_s*4+1, :] = sph.sh_mult(x_nm, A_nm[idx_s, :], 'real')
        A_wxyz[idx_s*4+2, :] = sph.sh_mult(y_nm, A_nm[idx_s, :], 'real')
        A_wxyz[idx_s*4+3, :] = sph.sh_mult(z_nm, A_nm[idx_s, :], 'real')
    return A_wxyz


# part of parallel pseudo_intensity:
def _intensity_sample(i, W, X, Y, Z, win):
    buf = len(win)
    # global shared_array
    shared_array[int(i + buf // 2), :] = np.asarray(
        [np.trapz(win * W[i:i + buf] * X[i:i + buf]),
         np.trapz(win * W[i:i + buf] * Y[i:i + buf]),
         np.trapz(win * W[i:i + buf] * Z[i:i + buf])])


@memory.cache
def pseudo_intensity(ambi_b, win_len=33, f_bp=None, smoothing_order=5,
                     jobs_count=1):
    """Direction of arrival (DOA) for each time sample from pseudo-intensity.

    Parameters
    ----------
    ambi_b : sig.AmbiBSignal
        Input signal, B-format.
    win_len : int optional
        Sliding window length.
    f_bp : tuple(f_lo, f_hi), optional
        Cutoff frequencies for bandpass, 'None' to disable.
    smoothing_order : int, optional
        Apply hanning(smoothing_order) smoothing to output.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    I_azi, I_zen, I_r : array_like
        Pseudo intensity vector for each time sample.

    """
    # WIP
    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()

    assert (win_len % 2)
    win = np.hanning(win_len)
    fs = ambi_b.fs
    # Z_0 = 413.3
    # T_int = 1/fs * win_len
    # a = 1 / (np.sqrt(2) * T_int * Z_0)

    # get first order signals
    W = utils.asarray_1d(ambi_b.W)
    X = utils.asarray_1d(ambi_b.X)
    Y = utils.asarray_1d(ambi_b.Y)
    Z = utils.asarray_1d(ambi_b.Z)

    # Bandpass signals
    if f_bp is not None:
        f_lo = f_bp[0]
        f_hi = f_bp[1]
        b, a = signal.butter(N=2, Wn=(f_lo / (fs / 2), f_hi / (fs / 2)),
                             btype='bandpass')
        W = signal.filtfilt(b, a, W)
        X = signal.filtfilt(b, a, X)
        Y = signal.filtfilt(b, a, Y)
        Z = signal.filtfilt(b, a, Z)

    # Initialize intensity vector
    I_vec = np.c_[np.zeros(len(ambi_b)),
                  np.zeros(len(ambi_b)), np.zeros(len(ambi_b))]

    if jobs_count == 1:
        # I = p*v for each sample
        for i in range(len(ambi_b) - win_len):
            I_vec[int(i + win_len // 2), :] = np.asarray(
                [np.trapz(win * W[i:i + win_len] * X[i:i + win_len]),
                 np.trapz(win * W[i:i + win_len] * Y[i:i + win_len]),
                 np.trapz(win * W[i:i + win_len] * Z[i:i + win_len])])
    else:
        logging.info("Using %i processes..." % jobs_count)
        # preparation
        shared_array_shape = np.shape(I_vec)
        _arr_base = _create_shared_array(shared_array_shape)
        _arg_itr = zip(range(len(ambi_b) - win_len),
                       repeat(W), repeat(X), repeat(Y), repeat(Z),
                       repeat(win))
        # execute
        with multiprocessing.Pool(processes=jobs_count,
                                  initializer=_init_shared_array,
                                  initargs=(_arr_base,
                                            shared_array_shape,)) as pool:
            pool.starmap(_intensity_sample, _arg_itr)
        # reshape
        I_vec = np.frombuffer(_arr_base.get_obj()).reshape(
            shared_array_shape)

    if smoothing_order > 0:
        assert (smoothing_order % 2)
        I_vec = np.apply_along_axis(signal.convolve, 0, I_vec,
                                    np.hanning(smoothing_order), 'same')
    I_azi, I_zen, I_r = utils.cart2sph(I_vec[:, 0], I_vec[:, 1],
                                       I_vec[:, 2], steady_zen=True)
    return I_azi, I_zen, I_r


def render_stereo_sdm(sdm_p, sdm_phi, sdm_theta):
    """Stereophonic SDM Render IR, with a cos(phi) pannign law.
    This is only meant for quick testing.

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


# part of parallel render_bsdm:
def _render_bsdm_sample(i, p, phi, theta, hrirs):
    h_l, h_r = hrirs.nearest_hrirs(phi, theta)
    # global shared_array
    with lock:  # synchronize access, operator += needs lock!
        shared_array[i:i + len(h_l), 0] += p * h_l
        shared_array[i:i + len(h_r), 1] += p * h_r


@memory.cache
def render_bsdm(sdm_p, sdm_phi, sdm_theta, hrirs, jobs_count=1):
    """Binaural SDM Render.
    Convolves each sample with corresponding hrir. No Post-EQ.

    Parameters
    ----------
    sdm_p : (n,) array_like
        Pressure p(t).
    sdm_phi : (n,) array_like
        Azimuth phi(t).
    sdm_theta : (n,) array_like
        Colatitude theta(t).
    hrirs : sig.HRIRs
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

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
            h_l, h_r = hrirs.nearest_hrirs(phi, theta)
            # convolve
            bsdm_l[i:i + len(h_l)] += p * h_l
            bsdm_r[i:i + len(h_r)] += p * h_r

    else:
        logging.info("Using %i processes..." % jobs_count)
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


def render_binaural_loudspeaker_sdm(sdm_p, ls_gains, ls_setup, fs,
                                    post_eq_func='default', **kwargs):
    """Render sdm signal on loudspeaker setup as binaural synthesis.

    Parameters
    ----------
    sdm_p : (n,) array_like
        Pressure p(t).
    ls_gains : (n, l)
        Loudspeaker (l) gains.
    ls_setup : decoder.LoudspeakerSetup
    fs : int
    post_eq_func : None, 'default' or function
        Post EQ applied to the loudspeaker signals. 'default' calls
        'parsa.post_equalization', 'None' disables (not recommended).
        You can also provide your custom post-eq-function with the signature
        `post_eq_func(ls_sigs, sdm_p, fs, ls_setup, **kwargs)`.

    Returns
    -------
    ir_l : array_like
        Left binaural impulse response.
    ir_r : array_like
        Right binaural impulse response.

    """
    n = len(sdm_p)
    ls_gains = np.atleast_2d(ls_gains)
    assert (n == ls_gains.shape[0])

    # render loudspeaker signals
    ls_sigs = ls_setup.loudspeaker_signals(ls_gains=ls_gains, sig_in=sdm_p)

    # post EQ
    if post_eq_func is not None:
        if post_eq_func == 'default':
            ls_sigs = sdm_post_equalization(ls_sigs, sdm_p, fs, ls_setup,
                                            **kwargs)
        else:  # user defined function
            ls_sigs = post_eq_func(ls_sigs, sdm_p, fs, ls_setup, **kwargs)
    else:
        warn("No post EQ applied!")

    ir_l, ir_r = ls_setup.binauralize(ls_sigs, fs)
    return ir_l, ir_r


def sdm_post_equalization(ls_sigs, sdm_p, fs, ls_setup, soft_clip=True):
    """Post equalization to compensate spectral whitening.

    Parameters
    ----------
    ls_sigs : (L, S) np.ndarray
        Input loudspeaker signals.
    sdm_p : array_like
        Reference (sdm) pressure signal.
    fs : int
    ls_setup : decoder.LoudspeakerSetup
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
    ls_distance = ls_setup.d  # ls distance
    a = ls_setup.a  # distance attenuation exponent

    CHECK_SANITY = False

    # prepare filterbank
    filter_gs, ff = pcs.frac_octave_filterbank(n=1, N_out=2**16,
                                               fs=fs, f_low=62.5, f_high=16000,
                                               mode='amplitude')

    # band dependent block size
    band_blocksizes = np.zeros(ff.shape[0])
    # proposed by Tervo
    band_blocksizes[1:] = np.round(7 / ff[1:, 0] * fs)
    band_blocksizes[0] = np.round(7 / ff[0, 1] * fs)
    # make sure they are even
    band_blocksizes = (np.ceil(band_blocksizes / 2) * 2).astype(int)

    padsize = band_blocksizes.max()

    ntaps = padsize // 2 - 1
    assert (ntaps % 2), "N does not produce uneven number of filter taps."
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
    assert (len(p_padded) == x_padded.shape[1])

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
                mag_diff = pcs.gain_clipping(mag_diff, 1)

            # apply to ls input
            Y = H_sdm * mag_diff[np.newaxis, :]

            # inverse STFT
            X = np.real(np.fft.ifft(Y, axis=1))
            # Zero Phase
            assert (np.mod(X.shape[1], 2))
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
    assert (ls_sigs_compensated.shape == ls_sigs.shape)
    return ls_sigs_compensated


def sdm_post_equalization2(ls_sigs, sdm_p, fs, ls_setup,
                       blocksize=4096, smoothing_order=5):
    """Post equalization to compensate spectral whitening. This alternative
    version works on fixed blocksizes with octave band gain smoothing.
    Sonically, this seems not the preferred version, but it can gain some
    insight through the band gains which are returned.

    Parameters
    ----------
    ls_sigs : (L, S) np.ndarray
        Input loudspeaker signals.
    sdm_p : array_like
        Reference (sdm) pressure signal.
    fs : int
    ls_setup : decoder.LoudspeakerSetup
    blocksize : int
    smoothing_order : int
        Block smoothing, increasing Hanning window up to this order.

    Returns
    -------
    ls_sigs_compensated : (L, S) np.ndarray
        Compensated loudspeaker signals.
    band_gains_list : list
        Each element contains the octave band gain applied as post eq.

    """
    ls_distance = ls_setup.d  # ls distance
    a = ls_setup.a  # distance attenuation exponent

    CHECK_SANITY = False

    hopsize = blocksize // 2
    win = np.hanning(blocksize + 1)[0: -1]

    # prepare Input
    pad = np.zeros([ls_sigs.shape[0], blocksize])
    x_padded = np.hstack([pad, ls_sigs, pad])
    p_padded = np.hstack([np.zeros(blocksize), sdm_p, np.zeros(blocksize)])
    ls_sigs_compensated = np.hstack([pad, np.zeros_like(x_padded), pad])
    assert (len(p_padded) == x_padded.shape[1])

    # prepare filterbank
    filter_gs, ff = pcs.frac_octave_filterbank(n=1, N_out=blocksize//2 + 1,
                                               fs=fs, f_low=62.5, f_high=16000)
    ntaps = blocksize+1
    assert (ntaps % 2), "N does not produce uneven number of filter taps."
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
        assert (len(p_mag) == len(sdm_mag_incoherent) == len(sdm_mag_coherent))

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
    assert (ls_sigs_compensated.shape == ls_sigs.shape)
    return ls_sigs_compensated, band_gains_list[2:-2]


# Parallel worker stuff -->
def _create_shared_array(shared_array_shape, d_type='d'):
    """Allocate ctypes array from shared memory with lock."""
    shared_array_base = multiprocessing.Array(d_type, shared_array_shape[0] *
                                              shared_array_shape[1])
    return shared_array_base


def _init_shared_array(shared_array_base, shared_array_shape):
    """Make 'shared_array' available to child processes."""
    global shared_array
    shared_array = np.frombuffer(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shared_array_shape)
# < --Parallel worker stuff
