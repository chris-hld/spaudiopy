#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

from spaudiopy import sig, sdm, process, plots, utils

# Load (A-format) SH impulse response
ambi_ir = sig.AmbiBSignal.from_file('../data/IR_Gewandhaus_SH1.wav')
# convert to B-format
ambi_ir.sh_to_b()

fs = ambi_ir.fs

# SDM Encoding
sdm_p = ambi_ir.W
sdm_azi, sdm_colat, _ = sdm.pseudo_intensity(ambi_ir, f_bp=(100, 5000))

# Show first 10000 samples DOA
plots.doa(sdm_azi[:10000], sdm_colat[:10000], fs, p=sdm_p[:10000])


# SDM Decoding
ir_l, ir_r = sdm.render_stereo_sdm(sdm_p, sdm_azi, sdm_colat)


# Render some examples
s_in = sig.MonoSignal.from_file('../data/piano_mono.flac', fs)
s_in.trim(2.6, 6)

s_out = sig.MultiSignal([s_in.signal, s_in.signal], fs=fs)
s_out.filter([ir_l, ir_r])


LISTEN = True
if LISTEN:
    print("input")
    s_in.play()
    print("output")
    s_out.play(wait=False, gain=0.5/np.max([ir_l, ir_r]))

