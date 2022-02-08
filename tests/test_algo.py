# -*- coding: utf-8 -*-
"""
pytest
@author: chris

Test algorithms.
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

# SH Order
N_SPHS = [1, 3, 5]  # higher orders might need more tolerance

@pytest.mark.parametrize('test_n_sph', N_SPHS)
def test_spat_filter_bank(test_n_sph):
    N_sph = test_n_sph
    sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design(2*N_sph).T)
    c_n = spa.sph.maxre_modal_weights(N_sph)
    [A, B] = spa.sph.design_spat_filterbank(N_sph, sec_dirs[0], sec_dirs[1],
                                            c_n, 'real', 'perfect')

    # diffuse SH signal
    in_nm = np.random.randn((N_sph+1)**2, 1000)
    # Sector signals (Analysis)
    s_sec = A @ in_nm
    # Reconstruction to SH domain
    out_nm = B.conj().T @ s_sec

    # Perfect Reconstruction
    assert_allclose(in_nm, out_nm)

