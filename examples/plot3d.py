#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:21:36 2020

@author: holdc1
"""

import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.gridspec import GridSpec



from spaudiopy import utils, IO, sph, plots, grids
plt.close('all')


def sph_coeffs(F_nm, SH_type=None, azi_steps=5, el_steps=3, title=None, ax=None):
    """Plot spherical harmonics coefficients as function on the sphere."""
    F_nm = utils.asarray_1d(F_nm)
    F_nm = F_nm[:, np.newaxis]
    if SH_type is None:
        SH_type = 'complex' if np.iscomplexobj(F_nm) else 'real'

    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(10e-3, np.pi + el_steps,
                                                 el_steps))

    f_plot = sph.inverse_sht(F_nm, phi_plot.ravel(), theta_plot.ravel(),
                             SH_type)
    f_r = np.abs(f_plot)
    f_ang = np.angle(f_plot)

    x_plot, y_plot, z_plot = utils.sph2cart(phi_plot.ravel(),
                                            theta_plot.ravel(),
                                            f_r.ravel())

    # Color mapping
    cmap = cm.hsv
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(f_ang)
    c = m.to_rgba(f_ang.reshape(phi_plot.shape))
    
    # Handle if ax is provided
    if ax is None:     
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        cb = plt.colorbar(m, ticks=[-np.pi, 0, np.pi], shrink=0.3, aspect=8)
        cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

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

    plt.grid(True)
    
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        pass

    ax.view_init(25, 230)
    if title is not None:
        plt.title(title)
    # fig.tight_layout()


def subplot_sph_coeffs(F_l, SH_type=None, azi_steps=5, el_steps=3, title=None):
    """Plot spherical harmonics coefficients as function on the sphere."""
    nplots = len(F_l)
    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(10e-3, np.pi + el_steps,
                                                 el_steps))

    
    fig = plt.figure(figsize=plt.figaspect(1.066 / (nplots)))
    # gs = GridSpec(2, nplots, height_ratios=[15, 1])
    # axs = []
    # for idx in range(nplots):
    #     ax = fig.add_subplot(gs[0, idx], projection='3d')
    #     try:
    #         ax.set_aspect('equal')
    #     except NotImplementedError:
    #         pass
    #     axs.append(ax)
    
    
    
        
    for idx, F in enumerate(F_l):
        ax = plt.subplot2grid((2, nplots), (0, idx), projection='3d')
        sph_coeffs(F, ax=ax)

    axcol = plt.subplot2grid((2, nplots), (1, 0), colspan=nplots)  # draw colorbar here
    # this colorbar:
    cmap = cm.hsv
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(F_l[0])
    cb = fig.colorbar(m, cax=axcol, ticks=[-np.pi, 0, np.pi], aspect=8,
                      orientation='horizontal')
    cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])





    
sph_coeffs(np.sqrt(2) * np.array([1, 0, 0, -1]), 'complex',
                 title="sqrt(2) * [1, 0, 0, -1] complex coeffs 3D")

subplot_sph_coeffs([np.array([1, 0, 0, 0]),
                          np.array([0, 1, 0, 0]),
                          np.array([0, 0, 1, 0]),
                          np.array([0, 0, 0, 1])],
                         title=["0", "1, -1", "1, 0", "1, 1 3D"])

ax = plt.gca()
