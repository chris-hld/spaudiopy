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



# use integer angles here (these definitely will not lead to special cases in
# which a rotation matches by chance)
@pytest.mark.parametrize(('test_n_sph', 'test_yaw', 'test_pitch', 'test_roll'),
                         (N_SPHS, [1, 3, 5], [1, 3, 5], [1, 3, 5]))
def test_rotation(test_n_sph, test_yaw, test_pitch, test_roll):
    for N_sph in test_n_sph:
        for yaw in test_yaw:
            for pitch in test_pitch:
                for roll in test_roll:
                    tgrid = spa.grids.load_t_design(degree=2*N_sph)                
                    tx = tgrid[:, 0]
                    ty = tgrid[:, 1]
                    tz = tgrid[:, 2]
                    tazi, tcolat, _ = spa.utils.cart2sph(tx, ty, tz)

                    R = spa.utils.rotation_euler(yaw, pitch, roll)
                    tgrid_rot = (R @ tgrid.T).T             
                    tx_rot = tgrid_rot[:, 0]
                    ty_rot = tgrid_rot[:, 1]
                    tz_rot = tgrid_rot[:, 2]
                    tazi_rot, tcolat_rot, _ = spa.utils.cart2sph(
                        tx_rot, ty_rot, tz_rot)
                    
                    shmat = spa.sph.sh_matrix(N_sph, tazi, tcolat, 'real')
                    shmat_rotref = spa.sph.sh_matrix(tazi_rot, tcolat_rot)

                    R = spa.sph.rotation_matrix(N_sph, yaw, pitch, roll, 'real',
                                                False)

                    shmat_rot = R @ shmat
                    assert_allclose(shmat_rotref, shmat_rot)

                    shmat_rot = spa.sph.rotate_sh(shmat)
                    assert_allclose(shmat_rotref, shmat_rot)
                    