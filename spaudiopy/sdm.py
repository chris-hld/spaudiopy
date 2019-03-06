# -*- coding: utf-8 -*-
"""
@author: chris
"""
from itertools import repeat

import numpy as np
from joblib import Memory
import multiprocessing

from . import utils
from . import sig
from . import process as pcs


# Prepare Caching
cachedir = './__cache_dir'
memory = Memory(cachedir)


def render_stereoSDM(sdm_p, sdm_phi, sdm_theta):
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


def _render_BSDM_sample(i, p, phi, theta, hrirs):
    h_l, h_r = hrirs[hrirs.nearest(phi, theta)]
    # global shared_array
    shared_array[i:i + len(h_l), 0] += p * h_l
    shared_array[i:i + len(h_l), 1] += p * h_r


@memory.cache
def render_BSDM(sdm_p, sdm_phi, sdm_theta, hrirs, jobs_count=None):
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
        Left impulse response.
    bsdm_r : array_like
        Right impulse response.
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
            pool.starmap(_render_BSDM_sample, _arg_itr)
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
        Left impulse response.
    ir_r : array_like
        Right impulse response.
    """
    n = len(sdm_p)
    ls_gains = np.atleast_2d(ls_gains)
    assert(n == ls_gains.shape[0])

    # render loudspeaker signals
    ls_sigs = sdm_p * ls_gains.T
    ir_l, ir_r = ls_setup.binauralize(ls_sigs, hrirs.fs, hrirs)
    return ir_l, ir_r


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
