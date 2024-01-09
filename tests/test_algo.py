# -*- coding: utf-8 -*-
"""
pytest
@author: chris

Test algorithms.
"""
import pytest

import numpy as np
from numpy.testing import assert_allclose

import spaudiopy as spa


# SH Order
N_SPHS = [1, 3, 5]  # higher orders might need more tolerance


@pytest.mark.parametrize('test_n_sph', N_SPHS)
def test_sph_filter_bank(test_n_sph):
    N_sph = test_n_sph
    sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design(2*N_sph).T)
    c_n = spa.sph.maxre_modal_weights(N_sph)
    [A, B] = spa.sph.design_sph_filterbank(N_sph, sec_dirs[0], sec_dirs[1],
                                           c_n, 'real', 'perfect')

    # diffuse SH signal
    in_nm = np.random.randn((N_sph+1)**2, 1000)
    # Sector signals (Analysis)
    s_sec = A @ in_nm
    # Reconstruction to SH domain
    out_nm = B @ s_sec

    # Perfect Reconstruction
    assert_allclose(in_nm, out_nm)


@pytest.mark.parametrize('test_n_sph', N_SPHS)
def test_calculate_grid_weights(test_n_sph):
    N_sph = test_n_sph
    vecs = spa.grids.load_t_design(degree=2*N_sph)
    azi, zen, _ = spa.utils.cart2sph(*vecs.T)

    q_weights_t = spa.grids.calculate_grid_weights(azi, zen)
    q_weights = 4*np.pi / len(q_weights_t) * np.ones_like(q_weights_t)

    # Perfect Reconstruction
    assert_allclose(q_weights_t, q_weights)


@pytest.mark.parametrize('test_n_sph', N_SPHS)
def test_rotation(test_n_sph):
    test_yaw, test_pitch, test_roll = (0, np.pi/2, 5), (0, 3, 5), (0, -3, 5)

    tgrid = spa.grids.load_t_design(degree=2*test_n_sph)
    tazi, tzen, _ = spa.utils.cart2sph(*tgrid.T)

    for yaw in test_yaw:
        for pitch in test_pitch:
            for roll in test_roll:
                print(yaw, pitch, roll)
                R = spa.utils.rotation_euler(yaw, pitch, roll)
                tgrid_rot = (R @ tgrid.T).T
                tazi_rot, tzen_rot, _ = spa.utils.cart2sph(*tgrid_rot.T)

                shmat = spa.sph.sh_matrix(test_n_sph, tazi, tzen, 'real')
                shmat_ref = spa.sph.sh_matrix(test_n_sph, tazi_rot, tzen_rot)

                R = spa.sph.sh_rotation_matrix(test_n_sph, yaw, pitch, roll,
                                               sh_type='real')

                shmat_rot = (R @ shmat.T).T
                assert_allclose(shmat_ref, shmat_rot, rtol=1e-3)

                shmat_rotate_sh = spa.sph.rotate_sh(shmat, yaw, pitch, roll)
                assert_allclose(shmat_ref, shmat_rotate_sh, rtol=1e-3)
