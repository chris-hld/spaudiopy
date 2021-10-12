# -*- coding: utf-8 -*-
"""
pytest
@author: chris

Test parallel computing.
"""
import os
import sys
import pytest

import numpy as np
from numpy.testing import assert_allclose

import spaudiopy as spa

#current_file_dir = os.path.dirname(__file__)
#sys.path.insert(0, os.path.abspath(os.path.join(
#                current_file_dir, '..')))

JOB_COUNTS = [1, 2, None]

@pytest.mark.parametrize('test_jobs', JOB_COUNTS)
def test_pseudo_intensity(test_jobs):
    fs = 44100
    n_samples = 10000
    ambi_b = spa.sig.AmbiBSignal([np.random.randn(n_samples),
                                  np.random.randn(n_samples),
                                  np.random.randn(n_samples),
                                  np.random.randn(n_samples)], fs=fs)
    azi_r, colat_r, r_r = spa.sdm.pseudo_intensity(ambi_b, jobs_count=1)
    azi_t, colat_t, r_t = spa.sdm.pseudo_intensity(ambi_b,
                                                   jobs_count=test_jobs)
    assert_allclose([azi_t, colat_t, r_t], [azi_r, colat_r, r_r])


@pytest.mark.parametrize('test_jobs', JOB_COUNTS)
def test_vbap(test_jobs):
    vecs = spa.grids.load_t_design(degree=5)
    hull = spa.decoder.LoudspeakerSetup(*vecs.T)
    src = np.random.randn(1000, 3)
    gains_r = spa.decoder.vbap(src, hull, jobs_count=1)
    gains_t = spa.decoder.vbap(src, hull, jobs_count=test_jobs)
    assert_allclose(gains_t, gains_r)


@pytest.mark.parametrize('test_jobs', JOB_COUNTS)
def test_allrap(test_jobs):
    vecs = spa.grids.load_t_design(degree=5)
    hull = spa.decoder.LoudspeakerSetup(*vecs.T)
    hull.ambisonics_setup(update_hull=False)
    src = np.random.randn(1000, 3)
    gains_r = spa.decoder.allrap(src, hull, jobs_count=1)
    gains_t = spa.decoder.allrap(src, hull, jobs_count=test_jobs)
    assert_allclose(gains_t, gains_r)


@pytest.mark.parametrize('test_jobs', JOB_COUNTS)
def test_allrap2(test_jobs):
    vecs = spa.grids.load_t_design(degree=5)
    hull = spa.decoder.LoudspeakerSetup(*vecs.T)
    hull.ambisonics_setup(update_hull=False)
    src = np.random.randn(1000, 3)
    gains_r = spa.decoder.allrap2(src, hull, jobs_count=1)
    gains_t = spa.decoder.allrap2(src, hull, jobs_count=test_jobs)
    assert_allclose(gains_t, gains_r)


#@pytest.mark.parametrize('test_jobs', JOB_COUNTS)
#def test_render_bsdm(test_jobs):
#    sdm_p, sdm_phi, sdm_theta = [*np.random.randn(3, 1000)]
#    hrirs = spa.IO.load_hrirs(fs=44100, filename='dummy')
#    bsdm_l_r, bsdm_r_r = spa.sdm.render_bsdm(sdm_p, sdm_phi, sdm_theta, hrirs,
#                                             jobs_count=1)
#    bsdm_l_t, bsdm_r_t = spa.sdm.render_bsdm(sdm_p, sdm_phi, sdm_theta, hrirs,
#                                             jobs_count=test_jobs)
#    assert_allclose([bsdm_l_t, bsdm_r_t], [bsdm_l_r, bsdm_r_r])


@pytest.mark.parametrize('test_jobs', JOB_COUNTS)
def test_resample_hrirs(test_jobs):
    hrirs = spa.IO.load_hrirs(fs=44100, filename='dummy')
    hrir_l_rsmp_r, hrir_r_rsmp_r, _ = spa.process.resample_hrirs(hrirs.left,
                                                                 hrirs.right,
                                                                 44100, 48000,
                                                                 jobs_count=1)
    hrir_l_rsmp_t, hrir_r_rsmp_t, _ = spa.process.resample_hrirs(hrirs.left,
                                                                 hrirs.right,
                                                                 44100, 48000,
                                                                 jobs_count=
                                                                 test_jobs)
    assert_allclose([hrir_l_rsmp_t, hrir_r_rsmp_t],
                    [hrir_l_rsmp_r, hrir_r_rsmp_r])

