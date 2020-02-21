# -*- coding: utf-8 -*-
"""
pytest
@author: chris

Test values against references.
"""
import os
import sys
import pytest

import numpy as np
from numpy.testing import assert_allclose

from scipy.io import loadmat

import spaudiopy as spa

current_file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(
                current_file_dir, '..')))

# SH Order
N = 8
# dict with results from reference implementations
ref_struct = loadmat(os.path.join(current_file_dir, 'reference.mat'))

# More data to compare against
cart_sph_data = [
    ((1, 1, 1), (np.pi / 4, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, 1), (3 / 4 * np.pi, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((1, -1, 1), (-np.pi / 4, np.arccos(1 / np.sqrt(3)),
                  np.sqrt(3))),
    ((-1, -1, 1), (-3 / 4 * np.pi, np.arccos(1 / np.sqrt(3)),
                   np.sqrt(3))),
    ((1, 1, -1), (np.pi / 4, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, -1), (3 / 4 * np.pi, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((1, -1, -1), (-np.pi / 4, np.arccos(-1 / np.sqrt(3)),
                   np.sqrt(3))),
    ((-1, -1, -1), (-3 / 4 * np.pi, np.arccos(-1 / np.sqrt(3)),
                    np.sqrt(3))),
]


# Makes N globally accessible in test_ functions
@pytest.fixture(autouse=True)
def myglobal(request):
    request.function.__globals__['N'] = N


# SH Tests
@pytest.mark.parametrize('expected_dirs', [ref_struct['dirs'], ])
def test_tDesign(expected_dirs):
    vecs = spa.grids.load_t_design(2*N)
    dirs = spa.utils.vecs2dirs(vecs)
    assert(np.allclose(dirs, expected_dirs))


@pytest.mark.parametrize('expected_Ynm', [ref_struct['Y_N_r'], ])
def test_realSH(expected_Ynm):
    vecs = spa.grids.load_t_design(2*N)
    dirs = spa.utils.vecs2dirs(vecs)
    Y_nm = spa.sph.sh_matrix(N, dirs[:, 0], dirs[:, 1], SH_type='real')
    assert(np.allclose(Y_nm, expected_Ynm))


@pytest.mark.parametrize('expected_Ynm', [ref_struct['Y_N_c'], ])
def test_cpxSH(expected_Ynm):
    vecs = spa.grids.load_t_design(2*N)
    dirs = spa.utils.vecs2dirs(vecs)
    Y_nm = spa.sph.sh_matrix(N, dirs[:, 0], dirs[:, 1], SH_type='complex')
    assert(np.allclose(Y_nm, expected_Ynm))


# Coordinate system conversion
@pytest.mark.parametrize('coord, polar', cart_sph_data)
def test_cart2sph(coord, polar):
    x, y, z = coord
    a = spa.utils.cart2sph(x, y, z)
    a_steady = spa.utils.cart2sph(x, y, z, steady_colat=True)
    assert_allclose(a_steady, a)
    assert_allclose(np.squeeze(a), polar)


@pytest.mark.parametrize('coord, polar', cart_sph_data)
def test_sph2cart(coord, polar):
    alpha, beta, r = polar
    b = spa.utils.sph2cart(azi=alpha, colat=beta, r=r)
    assert_allclose(np.squeeze(b), coord)
