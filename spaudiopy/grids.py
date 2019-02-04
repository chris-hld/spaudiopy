# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:34:33 2018

@author: chris
"""

import os
import numpy as np
from warnings import warn
from scipy.io import loadmat
try:
    import quadpy  # only for grid_lebedev()
except ImportError:
    pass


def load_t_design(degree):
    """Return the coordinates of minimal T-designs."""
    if degree > 21:
        raise ValueError('Designs of order > 21 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir, 't_designs_1_21.mat')
    mat = loadmat(file_path)
    t_designs_obj = mat['t_designs']
    t_designs = t_designs_obj[0].tolist()
    # degree t>=2N should be used for SH
    vecs = t_designs[degree - 1]
    return vecs


def load_Fliege_Maier_nodes(grid_order):
    """Return Fliege-Maier grid nodes with associated weights."""
    if grid_order > 30:
        raise ValueError('Designs of order > 30 are not implemented.')
    elif grid_order < 2:
        raise ValueError('Order should be at least 2.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir, 'fliegeMaierNodes_1_30.mat')
    mat = loadmat(file_path)
    fliege_maier_nodes = np.squeeze(mat['fliegeNodes'])
    # grid_order >= N+1 should be used for SH
    vecs = fliege_maier_nodes[grid_order - 1][:, :-1]
    weights = fliege_maier_nodes[grid_order - 1][:, -1]
    return vecs, weights


def equal_angle(n):
    """Equi-angular sampling points on a sphere.

    According to (cf. Rafaely book, sec.3.2)

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    colat, d_colat = np.linspace(0, np.pi, 2*n+2, endpoint=False, retstep=True)
    colat += d_colat/2

    weights = np.zeros_like(colat)
    p = np.arange(1, 2*n+2, 2)
    for i, theta in enumerate(colat):
        weights[i] = 2*np.pi/(n+1) * np.sin(theta) * np.sum(np.sin(p*theta)/p)

    azi = np.tile(azi, 2*n+2)
    colat = np.repeat(colat, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights /= n+1     # sum(weights) == 4pi
    return azi, colat, weights


def gauss(n):
    """ Gauss-Legendre sampling points on sphere.

    According to (cf. Rafaely book, sec.3.3)

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    x, weights = np.polynomial.legendre.leggauss(n+1)
    colat = np.arccos(x)
    azi = np.tile(azi, n+1)
    colat = np.repeat(colat, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights *= np.pi / (n+1)      # sum(weights) == 4pi
    return azi, colat, weights


def equal_polar_angle(n):
    """Equi-angular sampling points on a circle.

    Parameters
    ----------
    n : int
        Maximum order

    Returns
    -------
    pol : array_like
        Polar angle.
    weights : array_like
        Weights.
    """
    num_mic = 2*n+1
    pol = np.linspace(0, 2*np.pi, num=num_mic, endpoint=False)
    weights = 1/num_mic * np.ones(num_mic)
    return pol, weights


def lebedev(n):
    """Lebedev sampling points on sphere.

    (Maximum n is 65. We use what is available in quadpy, some n may not be
    tight, others produce negative weights.

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.

    """
    def available_quadrature(d):
        """Get smallest availabe quadrature of of degree d.

        see:
        https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
        """
        l = list(range(1, 32, 2)) + list(range(35, 132, 6))
        matches = [x for x in l if x >= d]
        return matches[0]

    if n > 65:
        raise ValueError("Maximum available Lebedev grid order is 65. "
                         "(requested: {})".format(n))

    # this needs https://pypi.python.org/pypi/quadpy
    q = quadpy.sphere.Lebedev(degree=available_quadrature(2*n))
    if np.any(q.weights < 0):
        warn("Lebedev grid of order {} has negative weights.".format(n))
    azi = q.azimuthal_polar[:, 0]
    colat = q.azimuthal_polar[:, 1]
    return azi, colat, 4*np.pi*q.weights
