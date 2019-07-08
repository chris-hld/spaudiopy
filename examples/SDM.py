#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

from spaudiopy import sig, sdm, process, plots, utils

# Load SH impulse response
ambi_ir = sig.MultiSignal.from_file('../data/IR_Gewandhaus_SH1.wav')
# convert to B-format
ambi_ir = sig.AmbiBSignal.sh_to_b(ambi_ir)

fs = ambi_ir.fs

# SDM Encoding
sdm_p = ambi_ir.W
sdm_azi, sdm_colat, _ = sdm.pseudo_intensity(ambi_ir, f_bp=(100, 5000))

# Show first 10000 samples DOA
plots.doa(sdm_azi[:10000], sdm_colat[:10000], fs, p=sdm_p[:10000])


# very quick stereo SDM decoding. This is only for testing!
ir_l, ir_r = sdm.render_stereo_sdm(sdm_p, sdm_azi, sdm_colat)


# Render some examples
s_in = sig.MonoSignal.from_file('../data/piano_mono.flac', fs)
s_in.trim(2.6, 6)

# Convolve with the omnidirectional IR
s_out_p = s_in.copy()
s_out_p.filter(sdm_p)

# Convolve with the stereo SDM IR
s_out_SDM = sig.MultiSignal([s_in.signal, s_in.signal], fs=fs)
s_out_SDM.filter([ir_l, ir_r])


LISTEN = True
if LISTEN:
    print("input")
    s_in.play()
    print("output: Omni")
    s_out_p.play(gain=0.5/np.max([ir_l, ir_r]))
    print("output: SDM")
    s_out_SDM.play(gain=0.5/np.max([ir_l, ir_r]), wait=False)
