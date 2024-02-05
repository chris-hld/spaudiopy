# -*- coding: utf-8 -*-
"""Sampling grids.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

"""

import os
import numpy as np
from warnings import warn
from scipy.io import loadmat
from . import sph


def calculate_grid_weights(azi, zen, order=None):
    """Approximate quadrature weights by pseudo-inverse.

    Parameters
    ----------
    azi : (Q,) array_like
        Azimuth.
    zen : (Q,) array_like
        Zenith / Colatitude.
    order : int, optional
        Supported order N, searched if not provided.

    Returns
    -------
    weights : (Q,) array_like
        Grid / Quadrature weights.

    References
    ----------
    Fornberg, B., & Martel, J. M. (2014). On spherical harmonics based
    numerical quadrature over the surface of a sphere.
    Advances in Computational Mathematics.

    """
    if order is None:  # search for max supported SHT order
        for itOrder in range(1, 100):
            cond = sph.check_cond_sht(itOrder, azi, zen, 'real', np.inf)
            if cond > 2*(itOrder+1):  # experimental condition
                order = itOrder-1
                break
    assert (order > 0)
    Y = sph.sh_matrix(order, azi, zen, 'real')
    P_leftinv = np.linalg.pinv(Y)
    weights = np.sqrt(4*np.pi) * P_leftinv[0, :]
    if (np.abs(np.sum(weights) - 4*np.pi) > 0.01) or np.any(weights < 0):
        warn('Could not calculate weights')
    return weights


def load_t_design(degree):
    """Return the unit coordinates of minimal T-designs.

    Parameters
    ----------
    degree : int
        T-design degree between 1 and 21.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.

    Notes
    -----
    Degree must be >= 2 * SH_order for spherical harmonic transform (SHT).

    References
    ----------
    The designs have been copied from:
    http://neilsloane.com/sphdesigns/
    and should be referenced as:

        "McLaren's Improved Snub Cube and Other New Spherical Designs in
        Three Dimensions", R. H. Hardin and N. J. A. Sloane, Discrete and
        Computational Geometry, 15 (1996), pp. 429-441.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs = spa.grids.load_t_design(degree=2*5)
        spa.plot.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 21:
        raise ValueError('Designs of order > 21 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/t_designs_1_21.mat')
    mat = loadmat(file_path)
    t_designs_obj = mat['t_designs']
    t_designs = t_designs_obj[0].tolist()
    # degree t>=2N should be used for SHT
    vecs = t_designs[degree - 1]
    return vecs


def load_n_design(degree):
    """Return the unit coordinates of spherical N-design
    (Chebyshev-type quadrature rules). Seem to be equivalent but more
    modern t-designs.

    Parameters
    ----------
    degree : int
       Degree of exactness N between 1 and 124.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.

    References
    ----------
    The designs have been copied from:
    https://homepage.univie.ac.at/manuel.graef/quadrature.php

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs = spa.grids.load_n_design(degree=2*5)
        spa.plot.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 124:
        raise ValueError('Designs of order > 124 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/n_designs_1_124.mat')
    mat = loadmat(file_path)
    try:
        n_design = mat['N' + f'{degree:03}']
    except KeyError:
        warn(f"Degree {degree} not defined, trying {degree+1} ...")
        n_design = load_n_design(degree + 1)
    return n_design


def load_lebedev(degree):
    """Return the unit coordinates of Lebedev grid.

    Parameters
    ----------
    degree : int
       Degree of precision p between 3 and 131.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.
    weights : array_like
        Quadrature weights.

    References
    ----------
    The designs have been copied from:
    https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, weights = spa.grids.load_lebedev(degree=2*5)
        spa.plot.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 131:
        raise ValueError('Designs of order > 131 are not implemented.')
    elif degree < 3:
        raise ValueError('Order should be at least 3.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/lebedevQuadratures_3_131.mat')
    mat = loadmat(file_path)
    try:
        design = mat['lebedev_' + f'{degree:03}']
        vecs = design[:, :3]
        weights = 4*np.pi * design[:, 3]
        if np.any(weights < 0):
            warn(f"Lebedev grid {degree} has negative weights.")
    except KeyError:
        warn(f"Degree {degree} not defined, trying {degree+1} ...")
        vecs, weights = load_lebedev(degree + 1)
    return vecs, weights


def load_Fliege_Maier_nodes(grid_order):
    """Return Fliege-Maier grid nodes with associated weights.

    Parameters
    ----------
    grid_order : int
        Grid order between 2 and 30

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.
    weights : array_like
        Quadrature weights.

    References
    ----------
    The designs have been copied from:
    http://www.personal.soton.ac.uk/jf1w07/nodes/nodes.html
    and should be referenced as:

        "A two-stage approach for computing cubature formulae for the sphere.",
        Jorg Fliege and Ulrike Maier, Mathematik 139T, Universitat Dortmund,
        Fachbereich Mathematik, Universitat Dortmund, 44221. 1996.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, weights = spa.grids.load_Fliege_Maier_nodes(grid_order=5)
        spa.plot.hull(spa.decoder.get_hull(*vecs.T))

    """
    if grid_order > 30:
        raise ValueError('Designs of order > 30 are not implemented.')
    elif grid_order < 2:
        raise ValueError('Order should be at least 2.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/fliegeMaierNodes_1_30.mat')
    mat = loadmat(file_path)
    fliege_maier_nodes = np.squeeze(mat['fliegeNodes'])
    # grid_order >= N+1 should be used for SHT
    vecs = fliege_maier_nodes[grid_order - 1][:, :-1]
    weights = fliege_maier_nodes[grid_order - 1][:, -1]  # sum(weights) == 4pi
    return vecs, weights


def load_maxDet(degree):
    """Return Maximum Determinant (Fekete, Extremal) points on the sphere.

    Parameters
    ----------
    degree : int
       Degree between 1 and 200.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.
    weights : array_like
        Quadrature weights.

    References
    ----------
    The designs have been copied from:
    https://web.maths.unsw.edu.au/~rsw/Sphere/MaxDet/

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, weights = spa.grids.load_maxDet(degree=5)
        spa.plot.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 200:
        raise ValueError('Designs of order > 200 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/maxDetPoints_1_200.mat')
    mat = loadmat(file_path)
    try:
        design = mat['maxDet_' + f'{degree:03}']
        vecs = design[:, :3]
        weights = design[:, 3]
        if np.any(weights < 0):
            warn(f"Grid {degree} has negative weights.")
    except KeyError:
        warn(f"Degree {degree} not defined, trying {degree+1} ...")
        vecs, weights = load_maxDet(degree + 1)
    return vecs, weights


def equal_angle(n):
    """Equi-angular sampling points on a sphere.

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    zen : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing., sec.3.2

    Examples
    --------
    .. plot::
        :context: close-figs

        azi, zen, weights = spa.grids.equal_angle(n=5)
        spa.plot.hull(spa.decoder.get_hull(*spa.utils.sph2cart(azi, zen)))

    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    zen, d_zen = np.linspace(0, np.pi, 2*n+2, endpoint=False, retstep=True)
    zen += d_zen/2

    weights = np.zeros_like(zen)
    p = np.arange(1, 2*n+2, 2)
    for i, theta in enumerate(zen):
        weights[i] = 2*np.pi/(n+1) * np.sin(theta) * np.sum(np.sin(p*theta)/p)

    azi = np.tile(azi, 2*n+2)
    zen = np.repeat(zen, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights /= n+1     # sum(weights) == 4pi
    return azi, zen, weights


def gauss(n):
    """Gauss-Legendre sampling points on sphere.

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    zen : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing., sec.3.3

    Examples
    --------
    .. plot::
        :context: close-figs

        azi, zen, weights = spa.grids.gauss(n=5)
        spa.plot.hull(spa.decoder.get_hull(*spa.utils.sph2cart(azi, zen)))

    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    x, weights = np.polynomial.legendre.leggauss(n+1)
    zen = np.arccos(x)
    azi = np.tile(azi, n+1)
    zen = np.repeat(zen, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights *= np.pi / (n+1)      # sum(weights) == 4pi
    return azi, zen, weights


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
