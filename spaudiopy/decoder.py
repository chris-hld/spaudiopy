# -*- coding: utf-8 -*-

import copy
import multiprocessing
from itertools import repeat

import numpy as np
import scipy.spatial as scyspat
from scipy import signal

from spaudiopy import utils, sph, IO, plots, grids


class LoudspeakerSetup:
    def __init__(self, x, y, z, listener_position=None):
        """Constructor"""
        self.x = utils.asarray_1d(x)
        self.y = utils.asarray_1d(y)
        self.z = utils.asarray_1d(z)
        if listener_position is None:
            listener_position = [0, 0, 0]
        self.listener_position = np.asarray(listener_position)

        # Listener position as origin
        self.x -= listener_position[0]
        self.y -= listener_position[1]
        self.z -= listener_position[2]
        # TODO: Better handling of this:
        self.listener_position -= self.listener_position
        _, _, self.d = utils.cart2sph(self.x, self.y, self.z)

        # amplitude decay exponent
        self.a = 1

        # Triangulation of points
        _hull = get_hull(self.x, self.y, self.z)
        self.points = _hull.points
        self.npoints = _hull.npoints
        self.nsimplex = _hull.nsimplex
        self.vertices = _hull.vertices
        self.simplices = _hull.simplices
        self.simplices = sort_vertices(self.simplices)
        self.centroids = calculate_centroids(self)
        self.face_areas = calculate_face_areas(self)
        self.face_normals = calculate_face_normals(self)
        self.vertex_normals = calculate_vertex_normals(self)
        self.barycenter = calculate_barycenter(self)

        # All simplices enclosing listener valid per default for rendering
        self._encloses_listener = check_listener_inside(self,
                                                        self.listener_position)
        self.valid_simplices = self._encloses_listener

        # see 'setup_for_ambisonic()'
        self.ambisonics_hull = []
        self.kernel_hull = []
        self.characteristic_order = None

    def is_simplex_valid(self, simplex):
        """Tests if simplex is in valid simplices (independent of orientation).
        """
        # find face in all faces
        in_s = np.isin(self.valid_simplices, simplex).sum(axis=-1) == 3
        return np.any(in_s)

    def pop_triangles(self, normal_limit=None, aperture_limit=None,
                      opening_limit=None, blacklist=None):
        """Refine triangulation by removing them from valid simplices."""
        if normal_limit is not None:
            self.valid_simplices = check_normals(self, normal_limit)
        if aperture_limit is not None:
            self.valid_simplices = check_aperture(self, aperture_limit)
        if opening_limit is not None:
            self.valid_simplices = check_opening(self, opening_limit)
        if blacklist is not None:
            self.valid_simplices = apply_blacklist(self, blacklist)

    def get_characteristic_order(self):
        N_e = characteristic_ambisonic_order(self)
        if N_e < 1:
            raise ValueError
        return N_e

    def setup_for_ambisonic(self, N_kernel=None):
        """Prepare loudspeaker hull for ambisonic rendering."""
        self.characteristic_order = self.get_characteristic_order()
        if N_kernel is None:
            print('Setting Ambisonics order =', self.characteristic_order)
            N_kernel = self.characteristic_order

        ls = self.points
        imaginary_loudspeaker_coordinates = find_imaginary_loudspeaker(self)
        # add imaginary speaker to hull
        new_ls = np.vstack([ls, imaginary_loudspeaker_coordinates])
        # This new triangulation is now the rendering setup
        ambisonics_hull = LoudspeakerSetup(new_ls[:, 0],
                                           new_ls[:, 1],
                                           new_ls[:, 2])
        # mark imaginary speaker (last one)
        ambisonics_hull.imaginary_speaker = new_ls.shape[0] - 1
        # virtual optimal loudspeaker arrangement
        virtual_speakers = grids.load_t_design(2 * N_kernel + 1)
        kernel_hull = LoudspeakerSetup(virtual_speakers[:, 0],
                                       virtual_speakers[:, 1],
                                       virtual_speakers[:, 2])

        del ambisonics_hull.ambisonics_hull
        del ambisonics_hull.kernel_hull
        self.ambisonics_hull = ambisonics_hull
        del kernel_hull.ambisonics_hull
        del kernel_hull.kernel_hull
        self.kernel_hull = kernel_hull

    def binauralize(self, ls_signals, fs, orientation=(0, 0), hrirs=None):
        """Create binaural signals that the loudspeaker signals produce on this
        setup (no delays).

        Parameters
        ----------
        ls_signals : (L, S) np.ndarray
            Loudspeaker signals.
        fs : int
        orientation : (azi, colat) tuple, optional
            Listener orientation offset (azimuth, colatitude) in rad.
        hrirs : sig.HRIRs, optional

        Returns
        -------
        l_sig : array_like
        r_sig : array_like

        """
        if hrirs is None:
            hrirs = IO.load_hrirs(fs)
        assert(hrirs.fs == fs)
        ls_signals = np.atleast_2d(ls_signals)
        assert ls_signals.shape[0] == self.npoints, \
            'Provide signal per loudspeaker!'
        # distance attenuation
        relative_position = self.points - \
                            self.listener_position
        ls_azi, ls_colat, ls_r = utils.cart2sph(*relative_position.T)
        ls_signals = np.diag(1 / ls_r ** self.a) @ ls_signals
        # convolve with hrir
        l_sig = np.zeros(ls_signals.shape[1] + len(hrirs) - 1)
        r_sig = np.zeros_like(l_sig)
        for ch, ls_sig in enumerate(ls_signals):
            if any(abs(ls_sig) > 10e-6):  # Gate at -100dB
                hrir_l, hrir_r = hrirs.nearest_hrirs(ls_azi[ch] -
                                                     orientation[0],
                                                     ls_colat[ch] -
                                                     orientation[1])
                # sum all loudspeakers
                l_sig += signal.convolve(ls_sig, hrir_l)
                r_sig += signal.convolve(ls_sig, hrir_r)
        return l_sig, r_sig

    def loudspeaker_signals(self, ls_gains, sig_in=None):
        """Render loudspeaker signals.

        Parameters
        ----------
        ls_gains : (S, L) np.ndarray
        sig_in : (S,) array like, optional

        Returns
        -------
        sig_out : (L, S) np.ndarray

        """
        ls_gains = np.atleast_2d(ls_gains)
        if sig_in is None:
            sig_in = np.ones(ls_gains.shape[0])
        sig_in = utils.asarray_1d(sig_in)
        assert(ls_gains.shape[1] == len(self.points)), \
            'Provide gain per speaker!'
        return (sig_in[:, np.newaxis] * ls_gains).T

    def show(self):
        plots.hull(self, title='Loudspeaker Setup')


def get_hull(x, y, z):
    """Wrapper for scipy.spatial.ConvexHull."""
    return scyspat.ConvexHull(np.c_[x, y, z], incremental=False)


def calculate_centroids(hull):
    """Calculate centroid for each simplex."""
    centroids = np.zeros((len(hull.simplices), 3))
    for face_i, face in enumerate(hull.simplices):
        # extract vertices face
        v = hull.points[face, :]
        centroids[face_i, :] = np.mean(v, axis=0)
    return centroids


def calculate_face_areas(hull):
    """Calculate area for each simplex."""
    face_areas = np.zeros(len(hull.simplices))
    for face_i, face in enumerate(hull.simplices):
        v = hull.points[face, :]
        face_areas[face_i] = utils.area_triangle(v[0, :], v[1, :], v[2, :])
    return face_areas


def calculate_face_normals(hull, eps=10e-6, normalize=False):
    """Calculate outwards pointing normal for each simplex."""
    face_normals = np.zeros((len(hull.simplices), 3))
    barycenter = np.mean(hull.points, axis=0)
    for face_i, face in enumerate(hull.simplices):
        # extract vertices face
        v = hull.points[face, :]
        centroid = np.mean(v, axis=0)
        # normal vector is cross product of two sides, initial point v0
        v_n = np.cross(v[1, :] - v[0, :], v[2, :] - v[0, :])
        # compare of face normal points in direction of barycenter
        criterion = np.dot(centroid + v_n - centroid, barycenter - centroid)
        # Make normal vector pointing outwards
        if criterion > eps:
            v_n = -v_n
        if normalize:
            v_n = v_n / np.linalg.norm(v_n)
        face_normals[face_i, :] = v_n
    return face_normals


def calculate_vertex_normals(hull, normalize=False):
    """Calculate normal for each vertex from simplices normals."""
    faces = hull.simplices
    vertex_normals = np.zeros([hull.npoints, 3])
    for p in hull.vertices:
        is_in_face = [p in row for row in faces]
        a = hull.face_areas[is_in_face]
        N = hull.face_normals[is_in_face]
        # weighted sum
        N_w = a[:, np.newaxis] * N
        vertex_n = np.sum(N_w, axis=0)
        if normalize:
            vertex_n = vertex_n / np.linalg.norm(vertex_n)
        vertex_normals[p, :] = vertex_n
    return vertex_normals


def calculate_barycenter(hull):
    """Barycenter of hull object."""
    return np.mean(hull.points, axis=0)


def check_listener_inside(hull, listener_position=None):
    """Return valid simplices for which the listener is inside the hull."""
    if listener_position is None:
        listener_position = hull.listener_position
    listener_position = np.asarray(listener_position)
    valid_simplices = []
    for face, centroid in zip(hull.simplices, hull.centroids):
        # centroid to listener
        v1 = listener_position - centroid
        # centroid to barycenter
        v2 = hull.barycenter - centroid
        # listener inside if both point in the same direction
        if np.dot(v1, v2) < 0:
            print("Listener not inside:", face)
        else:
            valid_simplices.append(face)
    return np.array(valid_simplices)


def check_normals(hull, normal_limit=85, listener_position=None):
    """Return valid simplices that point towards listener."""
    if listener_position is None:
        listener_position = hull.listener_position
    valid_simplices = []
    for face, v_n, c in zip(hull.simplices, hull.face_normals,
                            hull.centroids):
        if hull.is_simplex_valid(face):
            if utils.angle_between(c, v_n, vi=listener_position) > \
                    utils.deg2rad(normal_limit):
                print("Face not pointing towards listener: " + str(face))
            else:
                valid_simplices.append(face)
    return np.array(valid_simplices)


def check_aperture(hull, aperture_limit=90, listener_position=None):
    """Return valid simplices, where the aperture form the listener is small.
    """
    if listener_position is None:
        listener_position = hull.listener_position
    valid_simplices = []
    for face in hull.valid_simplices:
        # extract vertices face
        v = hull.points[face, :]
        a = utils.angle_between(v[0, :], v[1, :], vi=listener_position)
        b = utils.angle_between(v[1, :], v[2, :], vi=listener_position)
        c = utils.angle_between(v[0, :], v[2, :], vi=listener_position)
        if np.any(np.r_[a, b, c] > utils.deg2rad(aperture_limit)):
            print("Face produces critically large aperture: " + str(face))
        else:
            valid_simplices.append(face)
    return np.array(valid_simplices)


def check_opening(hull, opening_limit=135):
    """Return valid simplices with all opening angles within simplex > limit.
    """
    valid_simplices = []
    for face in hull.valid_simplices:
        # extract vertices face
        v = hull.points[face, :]
        a = utils.angle_between(v[1, :], v[2, :], vi=v[0, :])
        b = utils.angle_between(v[0, :], v[2, :], vi=v[1, :])
        c = utils.angle_between(v[0, :], v[1, :], vi=v[2, :])
        if np.any(np.r_[a, b, c] > utils.deg2rad(opening_limit)):
            print("Found large opening angle in face: " + str(face))
        else:
            valid_simplices.append(face)
    return np.array(valid_simplices)


def apply_blacklist(hull, blacklist=None):
    """Specify a blacklist to exclude simplcies from valid simplices."""
    if blacklist is not None:
        valid_simplices = []
        for face in hull.valid_simplices:
            if not all(elem in face for elem in blacklist):
                valid_simplices.append(face)
    return np.array(valid_simplices)


def sort_vertices(simplices):
    """Start the simplices with smallest vertex entry."""
    out = np.zeros_like(simplices)
    for i, face in enumerate(simplices):
        face = face[::-1]
        out[i, :] = np.roll(face, -np.argmin(face))
    return out


def find_imaginary_loudspeaker(hull):
    """Find imaginary loudspeaker coordinates according to
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, Sec. 1.1.
    """
    # detect edges
    rim_edges = []
    for face in hull.valid_simplices:
        # extract the three edges
        edges = np.zeros([3, 2])
        edges[0, :] = face[[0, 1]]
        edges[1, :] = face[[1, 2]]
        edges[2, :] = face[[0, 2]]

        # detect if edge occurs once
        for edge in edges:
            occured_once = np.count_nonzero(np.isin(
                            hull.valid_simplices, edge).sum(axis=1) == 2) == 1
            if occured_once:
                rim_edges.append(edge)
    # Check that all rim vertices are connected
    unique, counts = np.unique(rim_edges, return_counts=True)
    if not (counts >= 2).all():
        raise NotImplementedError("More than one rim found.")

    # Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    # Journal of Audio Engineering Society, sec. 1.1
    av_valid_n = np.zeros([1, 3])
    for face in hull.valid_simplices:
        # find valid face in all faces
        mask = np.isin(hull.simplices, face).sum(axis=-1) == 3
        av_valid_n += hull.face_areas[mask] * hull.face_normals[mask]
    # r**3 seems necessary
    imaginary_loudspeaker_coordinates = -av_valid_n / np.mean(hull.d)**3
    return imaginary_loudspeaker_coordinates


def _invert_triplets(simplices, points):
    """Invert loudspeaker triplets."""
    inverted_ls_triplets = []
    for face in simplices:
        # extract vertices face (valid LS positions)
        v = points[face, :]
        v_inv = np.linalg.lstsq(v, np.eye(3), rcond=None)[0]
        inverted_ls_triplets.append(v_inv.T)
    return inverted_ls_triplets


def _vbap_gains_single_source(src_idx, src, inverted_ls_triplets,
                              valid_simplices):
    for face_idx, ls_base in enumerate(inverted_ls_triplets):
        # projecting src onto loudspeakers
        projection = np.dot(ls_base, src[src_idx, :])
        # normalization
        projection /= np.sqrt(np.sum(projection**2))
        if np.all(projection > -10e-6):
            assert(np.count_nonzero(projection) <= 3)
            # print(f"Source {src_idx}: Gains {projection}")
            shared_array[src_idx, valid_simplices[face_idx]] = projection
            break  # found valid gains


def vbap(src, hull, valid_simplices=None, retain_outside=False,
         jobs_count=None):
    """Loudspeaker gains for Vector Base Amplitude Panning decoding.
    Pulkki, V. (1997). Virtual Sound Source Positioning Using Vector Base
    Amplitude Panning. AES, 144(5), 357–360.

    Parameters
    ----------
    src : (n, 3) numpy.ndarray
        Cartesian coordinates of n sources to be rendered.
    hull : LoudspeakerSetup
    valid_simplices : (nsimplex, 3) numpy.ndarray
        Valid simplices employed for rendering, defaults hull.valid_simplices.
    jobs_count : int, optional
        [CPU Cores], Number of Processes, switches implementation for n > 1.

    Returns
    -------
    gains : (n, L) numpy.ndarray
        Panning gains for L loudspeakers to render n sources.

    """
    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()
    if retain_outside:
        assert(valid_simplices is None)
        if hull.ambisonics_hull:
            hull = hull.ambisonics_hull
        else:
            raise ValueError('Run hull.setup_for_ambisonic() first!')

    if valid_simplices is None:
        valid_simplices = hull.valid_simplices

    src = np.atleast_2d(src)
    assert(src.shape[1] == 3)
    src_count = src.shape[0]

    ls_count = valid_simplices.max() + 1  # includes zero
    # Base
    inverted_ls_triplets = _invert_triplets(valid_simplices, hull.points)

    gains = np.zeros([src_count, ls_count])

    if (jobs_count == 1) or (src_count < 10):
        for src_idx in range(src_count):
            for face_idx, ls_base in enumerate(inverted_ls_triplets):
                # projecting src onto loudspeakers
                projection = np.dot(ls_base, src[src_idx, :])
                # normalization
                projection /= np.sqrt(np.sum(projection**2))
                if np.all(projection > -10e-6):
                    assert(np.count_nonzero(projection) <= 3)
                    # print(f"Source {src_idx}: Gains {projection}")
                    gains[src_idx, valid_simplices[face_idx]] = projection
                    break  # found valid gains
    else:
        # preparation
        shared_array_shape = np.shape(gains)
        _arr_base = _create_shared_array(shared_array_shape)
        _arg_itr = zip(range(src_count),
                       repeat(src), repeat(inverted_ls_triplets),
                       repeat(valid_simplices))
        # execute
        with multiprocessing.Pool(processes=jobs_count,
                                  initializer=_init_shared_array,
                                  initargs=(_arr_base,
                                            shared_array_shape,)) as pool:
            pool.starmap(_vbap_gains_single_source, _arg_itr)
        # reshape
        gains = np.frombuffer(_arr_base.get_obj()).reshape(
                                shared_array_shape)

    # Distance compensation
    gains = (hull.d[np.newaxis, :] ** hull.a) * gains
    if retain_outside:
        # remove imaginary loudspeaker
        gains = np.delete(gains, hull.imaginary_speaker, axis=1)
    return gains


def characteristic_ambisonic_order(hull):
    """Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and
    Decoding. Journal of Audio Engineering Society, Sec. 7.
    """
    _hull = copy.copy(hull)
    # projection for loudspeakers not on unit sphere
    _xp, _yp, _zp = sph.project_on_sphere(_hull.points[:, 0],
                                          _hull.points[:, 1],
                                          _hull.points[:, 2],)
    _hull.points = np.c_[_xp, _yp, _zp]
    # project centroids (for non-uniform setups)
    src = np.asarray(sph.project_on_sphere(hull.centroids[:, 0],
                                           hull.centroids[:, 1],
                                           hull.centroids[:, 2])).T
    # all loudspeaker triplets enclosing listener
    gains = vbap(src, _hull, valid_simplices=_hull._encloses_listener)
    # Energy vector of each center
    rE, rE_mag = sph.r_E(_hull.points, gains)
    # eq. (16)
    spread = 2 * np.arccos(rE_mag) * (180 / np.pi)
    N_e = 2 * 137.9 / np.average(spread) - 1.51
    # ceil might be optimistic
    return int(np.ceil(N_e))


def allrap(src, hull, N_sph=None):
    """Loudspeaker gains for All-Round Ambisonic Panning.
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, Sec. 4.

    Parameters
    ----------
    src : (N, 3)
        Cartesian coordinates of N sources to be rendered.
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    gains : (N, L) numpy.ndarray
        Panning gains for L loudspeakers to render N sources.

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    src = np.atleast_2d(src)
    src_count = src.shape[0]
    ls_count = ambisonics_hull.valid_simplices.max() + 1

    # normalize direction
    src = src / np.linalg.norm(src, axis=1)[:, np.newaxis]
    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources
    G = vbap(src=kernel_hull.points, hull=ambisonics_hull)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)

    gains = np.zeros([src_count, ls_count])
    for src_idx in range(src_count):
        # discretize panning function
        d = utils.angle_between(src[src_idx, :], kernel_hull.points)
        g_l = sph.bandlimited_dirac(N_sph, d, a_n)
        gains[src_idx, :] = 4 * np.pi / J * G.T @ g_l
    # remove imaginary loudspeaker
    gains = np.delete(gains, ambisonics_hull.imaginary_speaker, axis=1)
    return gains


def allrap2(src, hull, N_sph=None):
    """Loudspeaker gains for All-Round Ambisonic Panning 2.
    Zotter, F., & Frank, M. (2018). Ambisonic decoding with panning-invariant
    loudness on small layouts (AllRAD2). In 144th AES Convention.

    Parameters
    ----------
    src : (N, 3)
        Cartesian coordinates of N sources to be rendered.
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    gains : (N, L) numpy.ndarray
        Panning gains for L loudspeakers to render N sources.

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    src = np.atleast_2d(src)
    src_count = src.shape[0]
    # includes imaginary loudspeakers
    ls_count = ambisonics_hull.valid_simplices.max() + 1

    # normalize direction
    src = src / np.linalg.norm(src, axis=1)[:, np.newaxis]
    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources
    G = vbap(src=kernel_hull.points, hull=ambisonics_hull)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)
    # sqrt(E) normalization (eq.6)
    a_w = np.sqrt(np.sum((2 * (np.arange(N_sph + 1) + 1)) / (4 * np.pi) *
                         a_n**2))
    a_n /= a_w

    gains = np.zeros([src_count, ls_count])
    for src_idx in range(src_count):
        # discretize panning function
        d = utils.angle_between(src[src_idx, :], kernel_hull.points)
        g_l = sph.bandlimited_dirac(N_sph, d, a_n)
        gains[src_idx, :] = np.sqrt(4 * np.pi / J * G.T**2 @ g_l**2)
    # remove imaginary loudspeaker
    gains = np.delete(gains, ambisonics_hull.imaginary_speaker, axis=1)
    return gains


def allrad(F_nm, hull, N_sph=None):
    """Loudspeaker gains for All-Round Ambisonic Decoder.
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, Sec. 6.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources
    G = vbap(src=kernel_hull.points, hull=ambisonics_hull)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)
    a_n = sph.repeat_order_coeffs(a_n)

    # virtual Ambisonic decoder
    _t_azi, _t_colat, _t_r = utils.cart2sph(kernel_hull.points[:, 0],
                                            kernel_hull.points[:, 1],
                                            kernel_hull.points[:, 2])
    Y_td = sph.sh_matrix(N_sph, _t_azi, _t_colat, SH_type='real')
    # ALLRAD Decoder
    D = 4 * np.pi / J * G.T @ Y_td
    # apply tapering to decoder matrix
    D = D @ np.diag(a_n)
    # loudspeaker output signals
    ls_sig = D @ F_nm
    # remove imaginary loudspeakers
    ls_sig = np.delete(ls_sig, ambisonics_hull.imaginary_speaker, axis=0)
    return ls_sig


def allrad2(F_nm, hull, N_sph=None):
    """Loudspeaker gains for All-Round Ambisonic Decoder 2.
    Zotter, F., & Frank, M. (2018). Ambisonic decoding with panning-invariant
    loudness on small layouts (AllRAD2). In 144th AES Convention.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run hull.setup_for_ambisonic() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources
    G = vbap(src=kernel_hull.points, hull=ambisonics_hull)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)
    # sqrt(E) normalization (eq.6)
    a_w = np.sqrt(np.sum((2 * (np.arange(N_sph + 1) + 1)) / (4 * np.pi) *
                         a_n**2))
    a_n /= a_w
    a_n = sph.repeat_order_coeffs(a_n)

    # virtual Ambisonic decoder
    _t_azi, _t_colat, _t_r = utils.cart2sph(kernel_hull.points[:, 0],
                                            kernel_hull.points[:, 1],
                                            kernel_hull.points[:, 2])
    Y_td = sph.sh_matrix(N_sph, _t_azi, _t_colat, SH_type='real')

    # tapered dirac
    Y_k = Y_td @ np.diag(a_n) @ F_nm
    # ALLRAD2 Decoder
    ls_sig = np.sqrt(4 * np.pi / J * np.square(G.T) @ np.square(Y_k))

    # remove imaginary loudspeakers
    ls_sig = np.delete(ls_sig, ambisonics_hull.imaginary_speaker, axis=0)
    return ls_sig


def nearest_loudspeaker(src, hull):
    """Loudspeaker gains for nearest loudspeaker decoding, based on euclidean
    distance.

    Parameters
    ----------
    src : (N, 3)
        Cartesian coordinates of N sources to be rendered.
    hull : LoudspeakerSetup

    Returns
    -------
    gains : (N, L) numpy.ndarray
        Panning gains for L loudspeakers to render N sources.

    """
    src = np.atleast_2d(src)
    src_count = src.shape[0]
    gains = np.zeros([src_count, hull.npoints])

    # Loudspeakers projected on unit sphere
    ls_points = hull.points / hull.d[:, np.newaxis]
    p = np.inner(src, ls_points)
    idx = np.argmax(p, axis=1)
    for g, i in zip(gains, idx):
        g[i] = 1.0 * (hull.d[i] ** hull.a)
    return gains


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
