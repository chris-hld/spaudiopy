# -*- coding: utf-8 -*-
"""
A few helpers.
"""
import numpy as np


def asarray_1d(a, **kwargs):
    """Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.
    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def deg2rad(deg):
    """Convert from degree [0, 360) to radiant [0, 2*pi)."""
    return deg % 360 / 180 * np.pi


def rad2deg(rad):
    """Convert from radiant [0, 2*pi) to degree [0, 360)."""
    return rad / np.pi * 180 % 360


def cart2sph(x, y, z):
    """Vectorized conversion of cartesian to spherical coordinates."""
    x = asarray_1d(x)
    y = asarray_1d(y)
    z = asarray_1d(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    azi = np.arctan2(y, x)
    colat = np.arccos(z / r)
    return azi, colat, r


def sph2cart(azi, colat, r=1):
    """Vectorized conversion of spherical to cartesian coordinates."""
    azi = asarray_1d(azi)
    colat = asarray_1d(colat)
    r = asarray_1d(r)
    x = r * np.cos(azi) * np.sin(colat)
    y = r * np.sin(azi) * np.sin(colat)
    z = r * np.cos(colat)
    return x, y, z


def matlab_sph2cart(az, elev, r):
    """Matlab port with ELEVATION."""
    z = r * np.sin(elev)
    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev * np.sin(az)
    return x, y, z


def vecs2dirs(vecs, positive_azi=True):
    """Helper to convert [x, y, z] to [azi, colat]."""
    azi, colat, _ = cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    if positive_azi:
        azi = azi % (2 * np.pi)  # [-pi, pi] -> [0, 2pi)
    return np.c_[azi, colat]


def angle_between(v1, v2, vi=None):
    """Angle between point v1 and v2(s) with initial point vi."""
    v1 = asarray_1d(v1)
    if vi is not None:
        v1 = v1 - vi
        v2 = v2 - vi

    a = np.dot(v1, v2.T) / (np.linalg.norm(v1.T, axis=0) *
                            np.linalg.norm(v2.T, axis=0))
    return np.arccos(np.clip(a, -1.0, 1.0))


def area_triangle(p1, p2, p3):
    """calculate area of any triangle given coordinates of its corners p."""
    return 0.5 * np.linalg.norm(np.cross((p2 - p1), (p3 - p1)))


def db(x, power=False):
    """Convert *x* to decibel.
    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.
    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def rms(x, axis=-1):
    """RMS (root-mean-squared) along given axis.

    Parameters
    ----------
    x : array_like
        Input data.
    axis : int, optional
        Axis along which RMS is calculated
    """
    return np.sqrt(np.mean(x * np.conj(x), axis=axis))


def stack(vector_1, vector_2):
    """Stack two 2D vectors along the same-sized or the smaller dimension."""
    vector_1, vector_2 = np.atleast_2d(vector_1, vector_2)
    M1, N1 = vector_1.shape
    M2, N2 = vector_2.shape

    if (M1 == M2 and (M1 < N1 or M2 < N2)):
        out = np.vstack([vector_1, vector_2])
    elif (N1 == N2 and (N1 < M1 or N2 < M2)):
        out = np.hstack([vector_1, vector_2])
    else:
        raise ValueError('vector_1 and vector_2 dont have a common dimension.')
    return np.squeeze(out)


def test_diff(v1, v2, msg=None, VERBOSE=True):
    """Test if the cumulative element-wise difference between v1 and v2
    is greater 10-e8.
    """
    d = np.sum(np.abs(v1.ravel() - v2.ravel()))
    if VERBOSE:
        if msg is not None:
            print(msg, '--', end=' ')
        if np.any(d > 10e-8):
            print('Diff: ', d)
        else:
            print('Close enough')
    return d


def interleave_channels(left_channel, right_channel, style=None):
    """Interleave left and right channels (Nchannel x Nsamples).
    Style = 'SSR' checks if we total 360 channels.

    """
    if not left_channel.shape == right_channel.shape:
        raise ValueError('left_channel and right_channel '
                         'have to be of same dimensions!')

    if style == 'SSR':
        if not (left_channel.shape[0] == 360):
            raise ValueError('Provided arrays to have 360 channels '
                             '(Nchannel x Nsamples).')

    output_data = np.repeat(left_channel, 2, axis=0)
    output_data[1::2, :] = right_channel

    return output_data
