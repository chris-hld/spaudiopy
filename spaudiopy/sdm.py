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
    """Stereophonic SDM Render IR."""
    bsdm_l = np.zeros(len(sdm_p))
    bsdm_r = np.zeros_like(bsdm_l)

    for i, (p, phi, theta) in enumerate(zip(sdm_p, sdm_phi, sdm_theta)):
        h_l = 0.5*(1 + np.cos(phi - np.pi/2))
        h_r = 0.5*(1 + np.cos(phi + np.pi/2))
        # convolve
        bsdm_l[i] += p * h_l
        bsdm_r[i] += p * h_r
    return bsdm_l, bsdm_r


def _BSDM_sample(i, p, phi, theta, hrir_l, hrir_r, grid_phi, grid_theta):
    h_l, h_r = pcs.select_hrtf(hrir_l, hrir_r, grid_phi, grid_theta,
                               phi, theta)
    # global shared_array
    shared_array[i:i + len(h_l), 0] += p * h_l
    shared_array[i:i + len(h_l), 1] += p * h_r


@memory.cache
def render_BSDM(sdm_p, sdm_phi, sdm_theta, hrirs, n_jobs=None):
    """Binaural SDM Render.

    Parameters
    ----------
    sdm_p : array_like
    sdm_phi : array_like
    sdm_theta : array_like
    hrirs: sig.HRIRs
    n_jobs=None

    Returns:
    bsdm_l : array_like
    bsdm_r : array_like
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    hrir_l = hrirs.left
    hrir_r = hrirs.right
    grid = hrirs.grid
    bsdm_l = np.zeros(len(sdm_p) + hrir_l.shape[1] - 1)
    bsdm_r = np.zeros_like(bsdm_l)
    grid_phi = np.array(grid['az'])
    grid_theta = np.array(grid['el'])

    if n_jobs == 1:
        for i, (p, phi, theta) in enumerate(zip(sdm_p, sdm_phi, sdm_theta)):
            h_l, h_r = pcs.select_hrtf(hrir_l, hrir_r, grid_phi, grid_theta,
                                       phi, theta)
            # convolve
            bsdm_l[i:i + len(h_l)] += p * h_l
            bsdm_r[i:i + len(h_r)] += p * h_r

    else:
        bsdm_sigs = np.c_[bsdm_l, bsdm_r]
        shared_array_shape = np.shape(bsdm_sigs)
        _arr_base = create_shared_array(shared_array_shape)
        _arg_itr = zip(range(len(sdm_p)), sdm_p, sdm_phi, sdm_theta,
                       repeat(hrir_l), repeat(hrir_r),
                       repeat(grid_phi), repeat(grid_theta))
        # execute
        with multiprocessing.Pool(processes=n_jobs,
                                  initializer=init_shared_array,
                                  initargs=(_arr_base,
                                            shared_array_shape,)) as pool:
            pool.starmap(_BSDM_sample, _arg_itr)
        # reshape
        _result = np.frombuffer(_arr_base.get_obj()).reshape(
                                shared_array_shape)
        bsdm_l = _result[:, 0]
        bsdm_r = _result[:, 1]

    return bsdm_l, bsdm_r


# Parallel worker stuff -->
def create_shared_array(shared_array_shape):
    """Allocate ctypes array from shared memory with lock."""
    d_type = 'd'
    shared_array_base = multiprocessing.Array(d_type, shared_array_shape[0] *
                                              shared_array_shape[1])
    return shared_array_base


def init_shared_array(shared_array_base, shared_array_shape):
    """Makes 'shared_array' available to child processes."""
    global shared_array
    shared_array = np.frombuffer(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shared_array_shape)
# < --Parallel worker stuff
