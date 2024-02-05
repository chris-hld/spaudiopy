#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import spaudiopy as spa

LISTEN = True

# 5.0+4 Setup
ls_dirs = np.array([[0, -30, 30, 110, -110, -30, 30, 110, -110],
                    [0, 0, 0, 0, 0, 45, 45, 45, 45]])
ls_x, ls_y, ls_z = spa.utils.sph2cart(spa.utils.deg2rad(ls_dirs[0, :]),
                                      spa.utils.deg2rad(90 - ls_dirs[1, :]))

ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
ls_setup.show()

# Load SH impulse response
ambi_ir = spa.sig.MultiSignal.from_file('../data/IR_Gewandhaus_SH1.wav')
# convert to B-format
ambi_ir = spa.sig.AmbiBSignal.sh_to_b(ambi_ir)

fs = ambi_ir.fs

# - SDM Encoding:
sdm_p = ambi_ir.W
sdm_azi, sdm_zen, _ = spa.parsa.pseudo_intensity(ambi_ir, f_bp=(100, 5000))

# Show first 10000 samples DOA
spa.plot.doa(sdm_azi[:10000], sdm_zen[:10000], fs=fs, p=sdm_p[:10000])


# - SDM Decoding:
# very quick stereo SDM decoding. This is only for testing!
ir_st_l, ir_st_r = spa.parsa.render_stereo_sdm(sdm_p, sdm_azi, sdm_zen)

# Loudspeaker decoding
s_pos = np.array(spa.utils.sph2cart(sdm_azi, sdm_zen)).T
ls_gains = spa.decoder.nearest_loudspeaker(s_pos, ls_setup)
assert len(ls_gains) == len(sdm_p)
ir_ls_l, ir_ls_r = spa.parsa.render_binaural_loudspeaker_sdm(sdm_p, ls_gains,
                                                             ls_setup, fs)

# Render some examples
s_in = spa.sig.MonoSignal.from_file('../data/piano_mono.flac', fs)
s_in.trim(2.6, 6)

# Convolve with the omnidirectional IR
s_out_p = s_in.copy()
s_out_p.conv(sdm_p)

# Convolve with the stereo SDM IR
s_out_SDM_stereo = spa.sig.MultiSignal([s_in.signal, s_in.signal], fs=fs)
s_out_SDM_stereo.conv([ir_st_l, ir_st_r])

# Convolve with the loudspeaker SDM IR
s_out_SDM_ls = spa.sig.MultiSignal([s_in.signal, s_in.signal], fs=fs)
s_out_SDM_ls.conv([ir_ls_l, ir_ls_r])


if LISTEN:
    print("input")
    s_in.play()
    print("output: Omni")
    s_out_p.play(gain=0.5/np.max(sdm_p))
    print("output: Stereo SDM")
    s_out_SDM_stereo.play(gain=0.5/np.max(sdm_p))
    print("output: Binaural Loudspeaker SDM")
    s_out_SDM_ls.play(gain=0.5/np.max(sdm_p))
