# -*- coding: utf-8 -*-
"""Plotting helpers.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, tri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.spatial import ConvexHull

from . import utils, sph, decoder, process, grids


def spectrum(x, fs, scale_mag=False, **kwargs):
    """Positive (single sided) magnitude spectrum of time signal x.
    kwargs are forwarded to plot.freq_resp().

    Parameters
    ----------
    x : np.array, list of np.array
        Time domain signal.
    fs : int
        Sampling frequency.

    """
    if not isinstance(x, (list, tuple)):
        x = [x]
    bins = len(x[0])
    freq = np.fft.rfftfreq(bins, d=1. / fs)

    specs = []
    for s in x:
        assert (s.ndim == 1)
        # rfft returns half sided spectrum
        mag = np.abs(np.fft.rfft(s))
        # Scale the amplitude (factor two for mirrored frequencies)
        mag = mag / bins
        if bins % 2:
            # odd
            mag[1:] *= 2.
        else:
            # even
            # should be spec[1:-1] *= 2., but this looks "correct" for plotting
            mag[1:] *= 2.
        # scale by factor bins/2
        if scale_mag:
            mag = mag * bins/2
        specs.append(mag)

    freq_resp(freq, specs, **kwargs)


def freq_resp(freq, mag, TODB=True, smoothing_n=None, xlim=(20, 24000),
              ylim=30, title=None, labels=None, ax=None):
    """ Plot magnitude of frequency response over time frequency f.

    Parameters
    ----------
    f : frequency array
    mag : array_like, list of array_like
    TODB : bool
        Plot in dB.
    smoothing_n : int
        Forwarded to process.frac_octave_smoothing()

    Examples
    --------
    See :py:func:`spaudiopy.sph.binaural_coloration_compensation`

    """
    if not isinstance(mag, (list, tuple)):
        mag = [mag]
    if labels is not None:
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

    assert (all(len(a) == len(freq) for a in mag))

    if TODB:
        # Avoid zeros in spec for dB
        mag = [utils.db(np.clip(np.abs(a), 10e-15, None)) for a in mag]

    if smoothing_n is not None:
        smoothed = []
        for a in mag:
            smoothed.append(process.frac_octave_smoothing(a, smoothing_n,
                                                          WEIGHTED=True))
        mag = smoothed

    if ax is None:
        fig, ax = plt.subplots()
    [ax.semilogx(freq, a.flat) for a in mag]

    if smoothing_n is not None:
        if labels is None:
            labels = [None] * len(mag)
        # fake line for extra legend entry
        ax.plot([], [], '*', color='black')
        labels.append(r"$\frac{%d}{8}$ octave smoothing" % smoothing_n)
    if labels is not None:
        ax.legend(labels)
    
    # int specifies range from max
    if isinstance(ylim, int):
        ymax = np.max(mag)
        ylim = (ymax - ylim, ymax)

    ax.set_xlabel('Frequency in Hz')
    ax.set_ylabel('Magnitude in dB')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    if title is not None:
        plt.title(title)


def transfer_function(freq, H, title=None, xlim=(10, 25000)):
    """Plot transfer function H (magnitude and phase) over time frequency f."""
    fig, ax1 = plt.subplots()
    H = np.clip(H, 10e-15, None)
    ax1.semilogx(freq, utils.db(H),
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                 label='Magnitude')
    ax1.set_xlabel('Frequency in Hz')
    ax1.set_ylabel('Magnitude in dB')
    ax1.set_xlim(xlim)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.semilogx(freq, np.unwrap(np.angle(H)),
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                 label='Phase', zorder=0)
    ax2.set_ylabel('Phase in rad')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
    if title is not None:
        plt.title(title)


def zeropole(b, a, zPlane=False, title=None):
    """Plot Zero Pole diagram from a and b coefficients."""
    z = np.roots(b)
    p = np.roots(a)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
    ax.scatter(np.real(p), np.imag(p), marker='x', label='Poles')

    if zPlane:
        circle1 = plt.Circle([0, 0], 1, facecolor='none', edgecolor='r')
        ax.add_artist(circle1)
        plt.xlim(np.min([-1.1, plt.xlim()[0]]), np.max([1.1, plt.xlim()[1]]))
        plt.ylim(np.min([-1.1, plt.ylim()[0]]), np.max([1.1, plt.ylim()[1]]))
        plt.xlabel(r'$\Re(z)$')
        plt.ylabel(r'$\Im(z)$')
    else:
        plt.axhline(0, color='black', zorder=0)
        plt.axvline(0, color='red', zorder=0)
        plt.xlabel(r'$\Re(s)$')
        plt.ylabel(r'$\Im(s)$')

    plt.legend(loc=2)
    plt.grid(True)
    if title is not None:
        plt.title(title)


def compare_ambi(Ambi_A, Ambi_B):
    """Compare A and B format signals."""
    t = np.linspace(0, (len(Ambi_B)-1)/Ambi_B.fs, len(Ambi_B))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, Ambi_A.channel[0].signal,
             t, Ambi_A.channel[1].signal,
             t, Ambi_A.channel[2].signal,
             t, Ambi_A.channel[3].signal, alpha=0.3)
    plt.legend(['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4'])
    plt.title('A-format')
    plt.subplot(2, 1, 2)
    plt.plot(t, Ambi_B.channel[0].signal,
             t, Ambi_B.channel[1].signal,
             t, Ambi_B.channel[2].signal,
             t, Ambi_B.channel[3].signal, alpha=0.3)
    plt.legend(['Channel W', 'Channel X', 'Channel Y', 'Channel Z'])
    plt.xlabel('t in s')
    plt.title('B-format')


def spherical_function(f, azi, zen, title=None, ax=None):
    """Plot function 1D vector f over azi and zen."""
    f = utils.asarray_1d(np.real_if_close(f))
    azi = utils.asarray_1d(azi)
    zen = utils.asarray_1d(zen)
    x, y, z = utils.sph2cart(azi, zen, r=abs(f))

    # triangulate in the underlying parametrization
    chull = ConvexHull(np.column_stack((x, y, z)))
    triang = tri.Triangulation(zen, azi, triangles=chull.simplices)

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()
    ax.view_init(25, 230)

    p_tri = ax.plot_trisurf(x, y, z,
                            cmap=plt.cm.coolwarm,
                            # antialiased=False,
                            triangles=triang.triangles, shade=True,
                            edgecolor='none', linewidth=0.06, alpha=0.25)

    # Draw axis lines
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])
    for i in range(3):
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k',
                alpha=0.3)

    # overlay data points, radius as color
    p_sc = ax.scatter(x, y, z, c=f, cmap=plt.cm.viridis,
                      vmin=np.min([0, f.min()]),
                      vmax=np.max([1, f.max()]))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(nbins=5)

    cbar = plt.colorbar(p_sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(r"$f\,(\Omega)$")

    plt.grid(True)
    set_aspect_equal3d(ax)
    if title is not None:
        plt.title(title)


def sh_coeffs(F_nm, sh_type=None, azi_steps=5, el_steps=3, title=None,
              ax=None, cbar=True):
    """Plot spherical harmonics coefficients as function on the sphere.
    Evaluates the inverse SHT.

    Examples
    --------
    See :py:mod:`spaudiopy.sph`

    """
    F_nm = utils.asarray_1d(F_nm)
    F_nm = F_nm[:, np.newaxis]
    if sh_type is None:
        sh_type = 'complex' if np.iscomplexobj(F_nm) else 'real'

    phi_plot, theta_plot = np.meshgrid(np.linspace(0., 2 * np.pi,
                                                   int(360 / azi_steps)),
                                       np.linspace(10e-8, np.pi - 10e-8,
                                                   int(180 / el_steps)))

    f_plot = sph.inverse_sht(F_nm, phi_plot.ravel(), theta_plot.ravel(),
                             sh_type)
    f_r = np.abs(f_plot)
    f_ang = np.angle(f_plot)

    x_plot, y_plot, z_plot = utils.sph2cart(phi_plot.ravel(),
                                            theta_plot.ravel(),
                                            f_r.ravel())

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()
    ax.view_init(25, 230)

    m = cm.ScalarMappable(cmap=cm.hsv,
                          norm=colors.Normalize(vmin=-np.pi, vmax=np.pi))
    m.set_array(f_ang)
    c = m.to_rgba(f_ang.reshape(phi_plot.shape))

    ax.plot_surface(x_plot.reshape(phi_plot.shape),
                    y_plot.reshape(phi_plot.shape),
                    z_plot.reshape(phi_plot.shape),
                    facecolors=c,
                    edgecolor='none', linewidth=0.06, alpha=0.68, shade=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Draw axis lines
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])
    for i in range(3):
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k',
                alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(nbins=5)

    if cbar:
        cb = plt.colorbar(mappable=m, ax=ax, ticks=[-np.pi, 0, np.pi],
                          shrink=0.5, aspect=10)
        cb.set_label("Phase in rad")
        cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    plt.grid(True)
    set_aspect_equal3d(ax)
    if title is not None:
        plt.title(title)


def sh_coeffs_subplot(F_nm_list, titles=None, fig=None, **kwargs):
    """Plot spherical harmonics coefficients list as function on the sphere.
    `kwargs` are forwarded to :py:mod:`spaudiopy.plt.sh_coeffs`.

    Examples
    --------
    See :py:mod:`spaudiopy.sph`

    """
    num_plots = len(F_nm_list)

    if fig is None:
        fig = plt.figure(figsize=plt.figaspect(1 / num_plots),
                         constrained_layout=True)
    axs = kwargs.pop('ax', None)
    if axs is None:
        axs = fig.subplots(1, num_plots, subplot_kw={'projection': '3d'})
    sub_cbar = kwargs.pop('cbar', True)
    plt.suptitle(kwargs.pop('title', None))

    for idx_p, ax in enumerate(axs):
        sh_coeffs(F_nm_list[idx_p], ax=ax, cbar=False, **kwargs)

        ax.locator_params(nbins=3)
        if titles is not None:
            ax.set_title(titles[idx_p])

    if sub_cbar:
        m = cm.ScalarMappable(cmap=cm.hsv,
                              norm=colors.Normalize(vmin=-np.pi, vmax=np.pi))
        cb = plt.colorbar(m, ax=axs, shrink=0.5, aspect=10*num_plots, pad=0.1,
                          orientation='horizontal', anchor='S')
        cb.set_label("Phase in rad")
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])


def sh_coeffs_overlay(F_nm_list, sh_type=None, azi_steps=5, el_steps=3,
                      title=None, ax=None):
    """Overlay spherical harmonics coefficients plot.

    Examples
    --------
    See :py:mod:`spaudiopy.plot.sh_coeffs`

    """
    phi_plot, theta_plot = np.meshgrid(np.linspace(0., 2 * np.pi,
                                                   int(360 / azi_steps)),
                                       np.linspace(10e-8, np.pi - 10e-8,
                                                   int(180 / el_steps)))

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    ax.view_init(25, 230)

    # m = cm.ScalarMappable(cmap=cm.hsv,
    #                      norm=colors.Normalize(vmin=-np.pi, vmax=np.pi))
    # m.set_array(f_ang)
    # c = m.to_rgba(f_ang.reshape(phi_plot.shape))
    # ax.set_prop_cycle('color',
    # plt.cm.Spectral(np.linspace(0,1,len(F_nm_list))))
    cols = plt.cm.get_cmap('Set1')(np.arange(len(F_nm_list)))

    for idx, F_nm in enumerate(F_nm_list):
        F_nm = utils.asarray_1d(F_nm)
        F_nm = F_nm[:, np.newaxis]
        if sh_type is None:
            sh_type = 'complex' if np.iscomplexobj(F_nm) else 'real'
        f_plot = sph.inverse_sht(F_nm, phi_plot.ravel(), theta_plot.ravel(),
                                 sh_type)
        f_r = np.abs(f_plot)
        f_ang = np.angle(f_plot)

        x_plot, y_plot, z_plot = utils.sph2cart(phi_plot.ravel(),
                                                theta_plot.ravel(),
                                                f_r.ravel())
        ax.plot_surface(x_plot.reshape(phi_plot.shape),
                        y_plot.reshape(phi_plot.shape),
                        z_plot.reshape(phi_plot.shape),
                        # facecolors=c,
                        color=cols[idx, :],
                        edgecolor=(0.6, 0.6, 0.6, 0.6), linewidth=0.06,
                        alpha=0.38, shade=True)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Draw axis lines
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])
    for i in range(3):
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k',
                alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(nbins=5)

    # cb = plt.colorbar(m, ticks=[-np.pi, 0, np.pi], shrink=0.5, aspect=10)
    # cb.set_label("Phase in rad")
    # cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    plt.grid(True)
    set_aspect_equal3d(ax)
    if title is not None:
        plt.title(title)


def sh_rms_map(F_nm, TODB=False, w_n=None, sh_type=None, n_plot=50, title=None,
               clim=[None, None], ax=None):
    """Plot spherical harmonic signal RMS as function on the sphere.
    Evaluates the maxDI beamformer, if w_n is None.

    Parameters
    ----------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients, Ambisonic signal.
    TODB : bool
        Plot in dB.
    w_n : array_like
        Modal weighting of beamformers that are evaluated on the grid.
    sh_type :  'complex' or 'real' spherical harmonics.
    n_plot : int
        Plotting precision (grid degree).

    Examples
    --------
    See :py:mod:`spaudiopy.sph.src_to_sh`

    """
    F_nm = np.atleast_2d(F_nm)
    assert (F_nm.ndim == 2)
    if sh_type is None:
        sh_type = 'complex' if np.iscomplexobj(F_nm) else 'real'
    N_sph = int(np.sqrt(F_nm.shape[0]) - 1)

    vp = grids.load_n_design(n_plot)
    azi_plot, zen_plot, _ = utils.cart2sph(*vp.T)
    azi_plot = np.concatenate((azi_plot,
                               [np.pi, 0, -np.pi, np.pi, 0, -np.pi]))
    zen_plot = np.concatenate((zen_plot,
                               [0, 0, 0, np.pi, np.pi, np.pi]))

    Y_smp = sph.sh_matrix(N_sph, azi_plot.ravel(), zen_plot.ravel(), sh_type)
    if w_n is None:
        w_n = sph.hypercardioid_modal_weights(N_sph)

    mem_block = 2**16
    if F_nm.shape[1] > mem_block:
        rms_d_list = []
        start_idx = 0
        while start_idx + mem_block <= F_nm.shape[1]:
            f_d = Y_smp @ np.diag(sph.repeat_per_order(w_n)) @ F_nm[:, start_idx:start_idx+mem_block]
            rms_d_list.append(np.abs(utils.rms(f_d, axis=1)))
            start_idx += mem_block
        rms_d = np.sqrt(np.square(np.array(rms_d_list)).mean(axis=0))
    else:
        f_d = Y_smp @ np.diag(sph.repeat_per_order(w_n)) @ F_nm
        rms_d = np.abs(utils.rms(f_d, axis=1))

    if TODB:
        rms_d = utils.db(rms_d)

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal')

    if clim[0] is None:
        clim[0]=0 if not TODB else None
    p = ax.tricontourf(azi_plot, zen_plot, rms_d, levels=100,
                       vmin=clim[0], vmax=clim[1])
    ax.grid(True)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 3))
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Zenith')

    ax.axvline(x=0, color='grey', linestyle=':')
    ax.axhline(y=np.pi/2, color='grey', linestyle=':')

    ax.set_xticks([np.pi, np.pi/2, 0, -np.pi/2, -np.pi],
               labels=[r"$\pi$", r"$\pi/2$", r"$0$", r"$-\pi/2$", r"$-\pi$"])
    ax.set_yticks([0, np.pi/2, np.pi],
               labels=[r"$0$", r"$\pi/2$", r"$\pi$", ])

    cb = fig.colorbar(p, ax=ax, shrink=0.5)
    cb.set_label("RMS in dB" if TODB else "RMS")
    if title is not None:
        ax.set_title(title)


def spherical_function_map(f, azi, zen, TODB=False, title=None,
                           clim=(None, None), ax=None):
    """Plot function 1D vector f over azi and zen, can also convert to dB.

    Examples
    --------
    See :py:mod:`spaudiopy.parsa.sh_beamform`

    """
    f = utils.asarray_1d(np.real_if_close(f))
    azi = utils.asarray_1d(azi)
    zen = utils.asarray_1d(zen)
    azi[azi > np.pi] = azi[azi > np.pi] - 2*np.pi

    if TODB:
        f = utils.db(f)

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal')

    p = ax.tricontourf(azi, zen, f, levels=100, alpha=0.25, vmin=clim[0],
                       vmax=clim[1])
    p = ax.scatter(azi, zen, c=f, alpha=0.8, edgecolor='none', vmin=clim[0],
                   vmax=clim[1])
    ax.grid(True)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 3))
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Zenith')

    ax.axvline(x=0, color='grey', linestyle=':')
    ax.axhline(y=np.pi/2, color='grey', linestyle=':')

    ax.set_xticks([np.pi, np.pi/2, 0, -np.pi/2, -np.pi],
                  labels=[r"$\pi$", r"$\pi/2$", r"$0$", r"$-\pi/2$", r"$-\pi$"])
    ax.set_yticks([0, np.pi/2, np.pi],
                  labels=[r"$0$", r"$\pi/2$", r"$\pi$", ])

    cb = fig.colorbar(p, ax=ax, shrink=0.5)
    cb.set_label("in dB" if TODB else None)
    if title is not None:
        ax.set_title(title)


def sh_bar(x_nm, TODB=True, centered=False, num_groups=1, s=250, vf=4,
           clim=None, xticklabels=None, title=None, ax=None):
    """
    Barplot over SH channels.

    Parameters
    ----------
    x_nm : array_like
        C x L.
    TODB : TYPE, optional
        DESCRIPTION. The default is True.
    centered : TYPE, optional
        DESCRIPTION. The default is False.
    num_groups : TYPE, optional
        Plot gourps. The default is 1.
    s : TYPE, optional
        Scatter plot size. The default is 250.
    vf : TYPE, optional
        Vertical ratio. The default is 4.
    clim : TYPE, optional
        DESCRIPTION. The default is None.
    xticklabels : TYPE, optional
        DESCRIPTION. The default is None.
    title : TYPE, optional
        DESCRIPTION. The default is None.
    fig : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()

    val_plot = utils.db(np.asarray(x_nm)) if TODB else np.asarray(x_nm)
    val_plot = np.atleast_2d(val_plot)
    if centered:
        norm = colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=val_plot.min(), vmax=val_plot.max())
    mapper = cm.ScalarMappable(norm=norm, cmap='RdYlBu_r')
    mapper.set_array(val_plot)
    if clim is None:
        clim = (val_plot.min(), val_plot.max())
    mapper.set_clim(vmin=clim[0], vmax=clim[1])
    cols = mapper.to_rgba(val_plot)

    vf = vf
    hf = 1.
    verts = [[-vf, -hf], [vf, -hf], [vf, hf], [-vf, hf], [-vf, -hf]]

    L = val_plot.shape[1]
    fidx = 0
    xticks = []
    for idx1 in range(val_plot.shape[0] // num_groups):
        for idx2 in np.linspace(-0.25, 0.25, num_groups):
            xtick = idx1 + idx2
            for l_idx in range(L):
                ax.scatter(xtick, l_idx, color=cols[fidx, l_idx],
                           edgecolors='grey', linewidth=0.5, marker=verts, s=s)
            fidx += 1
            xticks.append(xtick)

    ax.set_xticks(np.array(xticks))
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_yticks([(n)**2 for n in range(int(np.sqrt(L)-1)+1)])
    ax.set_ylabel("L")
    ax.spines[['right', 'top']].set_visible(False)
    cb = fig.colorbar(mapper, ax=ax, aspect=40, extend='both'
                      if np.max(np.abs(clim)) < np.max(np.abs(val_plot))
                      else 'neither')
    cb.set_label("in dB" if TODB else None)
    ax.grid(True)


def hull(hull, simplices=None, mark_invalid=True, title=None, draw_ls=True,
         ax_lim=None, color=None, clim=None, ax=None):
    """Plot loudspeaker setup and valid simplices from its hull object.

    Parameters
    ----------
    hull : decoder.LoudspeakerSetup
    simplices : optional
    mark_invalid : bool, optional
        mark invalid simplices from hull object.
    title : string, optional
    draw_ls : bool, optional
    ax_lim : float, optional
        Axis limits in m.
    color : array_like, optional
        Custom colors for simplices.
    clim : (2,), optional
        `vmin` and `vmax` for colors.

    Examples
    --------
    See :py:mod:`spaudiopy.decoder`

    """
    if simplices is None:
        simplices = hull.simplices

    if mark_invalid:
        if not hasattr(hull, 'valid_simplices'):
            # Can happen when not a LoudspeakerSetup but generic hull object
            mark_invalid = False
            valid_s = simplices
        else:
            is_valid_s = np.array([hull.is_simplex_valid(s)
                                   for s in simplices])
            valid_s = simplices[is_valid_s]
            invalid_s = simplices[~is_valid_s]
            if np.all(is_valid_s):
                # nothing invalid to show
                mark_invalid = False
    else:
        valid_s = simplices

    if color is not None:
        if clim is None:
            clim = (None, None)
        color = utils.asarray_1d(color)
        assert (len(color) == simplices.shape[0])
        m = cm.ScalarMappable(cmap=cm.Spectral,
                              norm=colors.Normalize(vmin=clim[0],
                                                    vmax=clim[1]))
        m.set_array(color)
        colset = m.to_rgba(color)
    else:
        colset = None

    # extract data
    x = hull.points[:, 0]
    y = hull.points[:, 1]
    z = hull.points[:, 2]

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()
    ax.view_init(25, 230)

    # valid
    polyc = ax.plot_trisurf(x, y, z,
                            triangles=valid_s,
                            cmap=cm.Spectral if color is None else None,
                            edgecolor='grey', linewidth=0.1, alpha=0.6,
                            zorder=2)
    # apply colors if given
    polyc.set_facecolors(colset)
    # invalid
    if mark_invalid:
        ax.plot_trisurf(x, y, z,
                        triangles=invalid_s, linestyle='--',
                        edgecolor='grey', linewidth=0.1,
                        color=(0., 0., 0., 0.), alpha=0.1, zorder=2)
    # loudspeaker no
    if draw_ls:
        for s, co in enumerate(np.c_[x, y, z]):
            ax.scatter(co[0], co[1], co[2], c='black', s=20, alpha=0.5,
                       zorder=2)
            ax.text(co[0], co[1], co[2], s, zorder=2)

    # origin
    ax.scatter(0, 0, 0, s=30, c='darkgray', marker='+')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(tight=True, nbins=5)

    if ax_lim is None:
        ax_lim = np.max(np.array([abs(x), abs(y), abs(z)])).T
        ax_lim = 1.1*np.max([ax_lim, 1.0])  # 1.1 looks good
    set_aspect_equal3d(ax, XYZlim=(-ax_lim, ax_lim))

    if color is not None:
        fig.colorbar(m, ax=ax, fraction=0.024, pad=0.04)
    if title is not None:
        plt.title(title)


def hull_normals(hull, plot_face_normals=True, plot_vertex_normals=True):
    """Plot loudspeaker setup with vertex and face normals."""
    x = hull.points[:, 0]
    y = hull.points[:, 1]
    z = hull.points[:, 2]

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(25, 230)
    ax.plot_trisurf(x, y, z,
                    triangles=hull.simplices,
                    edgecolor='black', linewidth=0.5,
                    color=(0., 0., 0., 0.), alpha=0.15, zorder=3)

    for s, co in enumerate(np.c_[x, y, z]):
        ax.text(co[0], co[1], co[2], s, zorder=2)

    if plot_vertex_normals:
        ax.quiver(x, y, z,
                  hull.vertex_normals[:, 0],
                  hull.vertex_normals[:, 1],
                  hull.vertex_normals[:, 2],
                  arrow_length_ratio=0.1,
                  colors=(0.52941176, 0.56862745, 0.7372549, 0.8),
                  alpha=0.6, zorder=0, label='vertex normal')

    if plot_face_normals:
        c_x = hull.centroids[:, 0]
        c_y = hull.centroids[:, 1]
        c_z = hull.centroids[:, 2]
        ax.quiver(c_x, c_y, c_z,
                  hull.face_normals[:, 0],
                  hull.face_normals[:, 1],
                  hull.face_normals[:, 2],
                  arrow_length_ratio=0.1,
                  colors=(0.16078431, 0.64705882, 0.17647059, 1),
                  alpha=0.6, zorder=1, label='simplex normal')

    ax.scatter(hull.listener_position[0], hull.listener_position[1],
               hull.listener_position[2],
               s=48, marker='+', label='listener')

    ax.scatter(hull.barycenter[0], hull.barycenter[1], hull.barycenter[2],
               s=48, marker='*', label='barycenter')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(tight=True, nbins=5)

    set_aspect_equal3d(ax, XYZlim=[-1, 1])
    plt.legend(loc='best')


def polar(theta, r, TODB=True, rlim=None, title=None, ax=None):
    """Polar plot (in dB) that allows negative values for `r`.

    Examples
    --------
    See :py:func:`spaudiopy.sph.bandlimited_dirac`
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')

    # Split in pos and neg part and set rest NaN for plots
    rpos = np.copy(r)
    rpos[r < 0] = np.nan
    rneg = np.copy(r)
    rneg[r >= 0] = np.nan
    if TODB:
        if not rlim:
            rlim = (-40, 0)
        ax.plot(theta, utils.db(rpos), label='$+$')
        ax.plot(theta, utils.db(abs(rneg)), label='$-$')
        ax.text(6.5/8 * 2*np.pi, 3.3, 'dB', horizontalalignment='left')
    else:
        if not rlim:
            rlim = (0, 1)
        ax.plot(theta, rpos, label='$+$')
        ax.plot(theta, np.abs(rneg), label='$-$')

    ax.set_theta_offset(np.pi/2)
    ax.set_rmin(rlim[0])
    ax.set_rmax(rlim[1] + (0.5 if TODB else 0.03))
    ax.set_rticks(np.linspace(rlim[0], rlim[1], 5))
    ax.set_rlabel_position(6.5/8 * 360)
    ax.legend(loc='lower right', bbox_to_anchor=(1.175, 0.15))
    if title is not None:
        ax.set_title(title)


def decoder_performance(hull, renderer_type, azi_steps=5, ele_steps=3,
                        show_ls=True, title=None, **kwargs):
    """Shows amplitude, energy, spread and angular error measures on grid.
    For renderer_type={'VBAP', 'VBIP', 'ALLRAP', 'NLS'},
    as well as {'ALLRAD', 'ALLRAD2', 'EPAD', 'MAD', 'SAD'}.
    All kwargs are forwarded to the decoder function.

    References
    ----------
    Zotter, F., & Frank, M. (2019). Ambisonics.
    Springer Topics in Signal Processing.

    Examples
    --------
    See :py:mod:`spaudiopy.decoder`
    """
    azi_steps = np.deg2rad(azi_steps)
    ele_steps = np.deg2rad(ele_steps)
    phi_vec = np.arange(-np.pi, np.pi + azi_steps, azi_steps)
    theta_vec = np.arange(0., np.pi + ele_steps, ele_steps)
    phi_plot, theta_plot = np.meshgrid(phi_vec, theta_vec)
    _grid_x, _grid_y, grid_z = utils.sph2cart(phi_plot.ravel(),
                                              theta_plot.ravel())

    # Prepare for SH based rendering
    if renderer_type.lower() in ['allrad', 'allrad2', 'epad', 'mad', 'sad']:
        if 'N_sph' in kwargs:
            N_sph = kwargs.pop('N_sph')
        else:
            N_sph = hull.get_characteristic_order()
        Y_in = sph.sh_matrix(N_sph, phi_plot.flatten(), theta_plot.flatten(),
                             sh_type='real').T

    # Switch renderer
    if renderer_type.lower() == 'vbap':
        G = decoder.vbap(np.c_[_grid_x, _grid_y, grid_z], hull, **kwargs)
    elif renderer_type.lower() == 'vbip':
        G = decoder.vbip(np.c_[_grid_x, _grid_y, grid_z], hull, **kwargs)
    elif renderer_type.lower() == 'allrap':
        G = decoder.allrap(np.c_[_grid_x, _grid_y, grid_z], hull,
                           **kwargs)
    elif renderer_type.lower() == 'allrap2':
        G = decoder.allrap2(np.c_[_grid_x, _grid_y, grid_z], hull,
                            **kwargs)
    elif renderer_type.lower() == 'nls':
        G = decoder.nearest_loudspeaker(np.c_[_grid_x, _grid_y, grid_z], hull,
                                        **kwargs)
    elif renderer_type.lower() == 'allrad':
        G = decoder.allrad(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'allrad2':
        G = decoder.allrad2(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'epad':
        G = decoder.epad(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'mad':
        G = decoder.mad(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'sad':
        G = decoder.sad(Y_in, hull, N_sph=N_sph, **kwargs).T
    else:
        raise ValueError('Unknown renderer_type')

    # Measures
    # Amplitude
    A = np.sum(G, axis=1)
    # Energy
    E = np.sum(G**2, axis=1)  # * (4 * np.pi / G.shape[1])  # (eq. 15)
    # project points onto unit sphere
    ls_points = hull.points / hull.d[:, np.newaxis]
    rE, rE_mag = sph.r_E(ls_points, G / hull.d[np.newaxis, :] ** hull.a)
    # Zotter book (eq. 2.11) adds factor 5/8
    spread = 2 * np.arccos(np.clip(rE_mag, 0, 1)) * 180 / np.pi
    # angular error
    col_dot = np.einsum('ij,ij->i', np.array([_grid_x, _grid_y, grid_z]).T,
                        (rE / (np.clip(rE_mag, 10e-15, None)[:, np.newaxis])))
    ang_error = np.rad2deg(np.arccos(np.clip(col_dot, -1.0, 1.0)))

    # Show them
    fig, axes = plt.subplots(2, 2, sharex='all', sharey='all')
    axes = axes.ravel()
    for ip, _data in enumerate([A, E, spread, ang_error]):
        _data = _data.reshape(phi_plot.shape)
        ax = axes[ip]
        ax.set_aspect('equal')
        # draw mesh, value corresponds to center of mesh
        p = ax.pcolormesh(phi_plot, theta_plot, _data,
                          shading='gouraud', vmin=0,
                          vmax=np.max([1.0, np.max(_data)])
                          if ip in [0, 1] else 90)

        if show_ls:
            for s, co in enumerate(ls_points):
                # map to pixels grid
                _azi_ls, _zen_ls, _ = utils.cart2sph(*co)
                ax.plot(_azi_ls, _zen_ls, marker='2', color='grey')
                # ax.text(_x_plot, _y_plot, s)  # LS Number
        # Labeling etc
        ax.grid(True)
        ax.set_xlim([-np.pi - azi_steps/2, np.pi + azi_steps/2])
        ax.set_ylim([0 - ele_steps/2, np.pi + ele_steps/2])
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                            r'$\pi/2$', r'$\pi$'])
        ax.set_yticks(np.linspace(0, np.pi, 3))
        ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar = fig.colorbar(p, ax=ax, fraction=0.024, pad=0.04)
        cbar.outline.set_linewidth(0.5)
        if ip == 0:
            ax.set_title(r'$A$')
            cbar.set_ticks([0, 0.5, np.max([1.0, np.max(_data)])])
        if ip == 1:
            ax.set_title(r'$E$')
            cbar.set_ticks([0, 0.5, np.max([1.0, np.max(_data)])])
        elif ip == 2:
            ax.set_xlabel('Azimuth')
            ax.set_ylabel('Zenith')

            ax.set_title(r'$\sigma_E$')
            cbar.set_ticks([0, 45, 90])
            cbar.set_ticklabels([r'$0^{\circ}$', r'$45^{\circ}$',
                                 r'$90^{\circ}$'])
        elif ip == 3:
            ax.set_title(r'$\Delta \angle$')
            cbar.set_ticks([0, 45, 90])
            cbar.set_ticklabels([r'$0^{\circ}$', r'$45^{\circ}$',
                                 r'$90^{\circ}$'])

    if title is None:
        title = renderer_type
    else:
        title = renderer_type + ', ' + str(title)
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.25)


def doa(azi, zen, p=None, size=250, c=None, alpha=None, fs=None, title=None,
        ltitle=None, ax=None):
    """Direction of Arrival, with optional p(t) scaling the size.

    Examples
    --------
    .. plot::
        :context: close-figs

        n = 300
        fs = 44100
        t_ms = np.linspace(0, n/fs, n, endpoint=False) * 1000  # t in ms

        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)
        azi, zen, r = spa.utils.cart2sph(x, y, z)

        ps = 1 / np.exp(np.linspace(0, 3, n))
        spa.plot.doa(azi, zen, ps, fs=fs, ltitle="p(t)")

    """
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')

    # shift azi to [np.pi, np.pi]
    azi[azi > np.pi] = azi[azi > np.pi] % -np.pi

    if p is not None:
        s_plot = np.clip(p / np.max(p), 10e-15, None)
    else:
        s_plot = np.ones_like(azi)
    # scale
    s_plot *= size

    if alpha is None:
        alpha = 0.35 * np.ones_like(azi)

    # plot in reverse order so that first reflections are on top
    if c is None and fs is not None:  # t in ms
        t_ms = np.linspace(0, len(azi) / fs, len(azi), endpoint=False) * 1000
        p = ax.scatter(azi[::-1], zen[::-1], s=s_plot[::-1], c=t_ms[::-1],
                       alpha=alpha[::-1])
    else:
        if c is None:
            c = np.ones_like(azi)
        p = ax.scatter(azi[::-1], zen[::-1], s=s_plot[::-1], c=c[::-1],
                       alpha=alpha[::-1])
    ax.set_xlim([-(np.pi+0.03), (np.pi+0.03)])
    ax.set_ylim([-(0.03), (np.pi+0.03)])

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 3))
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Zenith')

    ax.axvline(x=0, color='grey', linestyle=':')
    ax.axhline(y=np.pi/2, color='grey', linestyle=':')

    ax.set_xticks([np.pi, np.pi/2, 0, -np.pi/2, -np.pi],
        labels=[r"$\pi$", r"$\pi/2$", r"$0$", r"$-\pi/2$", r"$-\pi$"])
    ax.set_yticks([0, np.pi/2, np.pi],
        labels=[r"$0$", r"$\pi/2$", r"$\pi$", ])
    ax.grid(True)

    if fs is not None:
        # show t as colorbar
        cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
        cbar.set_label("t in ms")

    if ltitle is not None:
        try:
            # produce a legend with a cross section of sizes from the scatter
            handles, labels = p.legend_elements(prop="sizes", alpha=0.3, num=5,
                                                func=lambda x: x/size)
            ax.legend(handles, labels, loc="upper right", title=ltitle)
        except AttributeError:  # mpl < 3.3.0
            pass

    if title is not None:
        ax.set_title(title)


def hrirs_ild_itd(hrirs, plevels=50, pclims=(None, None), title=None,
                  fig=None):
    """Plot ILDs and ITDs of HRIRs.

    Parameters
    ----------
    hrirs : sig.HRIRs
    plevels : int, optional
        Contour levels. The default is 50.
    pclims : (2,), optional
        Set the plot color limits for ild and itd, e.g. (20, 0.75)
    title : string, optional.
    fig : plt.figure, optional

    Returns
    -------
    None.

    See Also
    --------
    spaudiopy.process.ilds_from_hrirs : Calculating ILDs with defaults (in dB).
    spaudiopy.process.itds_from_hrirs : Calculating ITDs with defaults.

    Examples
    --------
    .. plot::
        :context: close-figs

        dummy_hrirs = spa.io.load_hrirs(48000, 'dummy')
        spa.plot.hrirs_ild_itd(dummy_hrirs)

    """
    ilds = process.ilds_from_hrirs(hrirs, TODB=True)
    itds = process.itds_from_hrirs(hrirs)

    pazi = np.fmod(hrirs.azi + np.pi, 2*np.pi) - np.pi
    pzen = hrirs.zen
    if fig is None:
        fig = plt.figure(constrained_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    if pclims[0] is None:
        pclim_ild = max(abs(ilds))
    else:
        pclim_ild = pclims[0]
    p1 = ax1.tricontourf(pazi, pzen, ilds, levels=plevels,
                         cmap='RdYlBu', vmin=-pclim_ild, vmax=pclim_ild)
    ax1.set_title("ILD")

    if pclims[1] is None:
        pclim_itd = max(abs(1000 * itds))
    else:
        pclim_itd = pclims[1]
    p2 = ax2.tricontourf(pazi, pzen, 1000 * itds, levels=plevels,
                         cmap='RdYlBu', vmin=-pclim_itd, vmax=pclim_itd)
    ax2.set_title("ITD")

    for axit in [ax1, ax2]:
        axit.set_aspect('equal')
        axit.grid(True)

        axit.set_xticks(np.linspace(-np.pi, np.pi, 5))
        axit.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                             r'$\pi/2$', r'$\pi$'])
        axit.set_yticks(np.linspace(0, np.pi, 3))
        axit.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
        axit.set_ylabel('Zenith')

        axit.axvline(x=0, color='grey', linestyle=':')
        axit.axhline(y=np.pi/2, color='grey', linestyle=':')

        axit.set_xticks([np.pi, np.pi/2, 0, -np.pi/2, -np.pi],
                labels=[r"$\pi$", r"$\pi/2$", "$0$", r"$-\pi/2$", r"$-\pi$"])
        axit.set_yticks([0, np.pi/2, np.pi],
                        labels=[r"$0$", r"$\pi/2$", r"$\pi$", ])

    ax2.set_xlabel('Azimuth')
    cb1 = fig.colorbar(p1, ax=ax1)
    cb1.set_label("ILD in dB")
    cb2 = fig.colorbar(p2, ax=ax2)
    cb2.set_label("ITD in ms")
    if title is not None:
        fig.suptitle(title)


def set_aspect_equal3d(ax=None, XYZlim=None):
    """Set 3D axis to equal aspect.

    Parameters
    ----------
    ax : axis, optional
        ax object. The default is None.
    XYZlim : [min, max], optional
        min and max in m. The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        ax = plt.gca()

    if XYZlim is None:
        xyzlim = np.array([ax.get_xlim3d(),
                           ax.get_ylim3d(),
                           ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]

    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    try:  # mpl < 3.1.0
        ax.set_aspect('equal')
    except NotImplementedError:  # mpl >= 3.3.0
        ax.set_box_aspect((1, 1, 1))
