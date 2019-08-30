#!/usr/bin/env python2
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""
:platform: Unix, Mac
:synopsis: This module processes the beacon receiver data in DigitalRF format to
get geophysical parameters and saves the output to digital metadata. Derived from
jitter code from Juha Vierinen.


"""
from __future__ import absolute_import, division, print_function

import argparse
import datetime as dt
import math
import os
import shutil
import sys

import digital_rf as drf
import ephem
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as md
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as s_const
import scipy.fftpack as scfft
import scipy.signal as sig
from mpl_toolkits.basemap import Basemap


TLE_def = (
    "CASSIOPE",
    "1 39265U 13055A   17015.93102647 +.00001768 +00000-0 +52149-4 0  9992",
    "2 39265 080.9701 172.4450 0693099 333.2926 023.4045 14.21489964169876",
)
debug_plot = False

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 100  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#" * block + "-" * (barLength - block), progress * 100, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def ephem_doponly(maindir, tleoff=10.0):
    """
        This function will output a dictionary that can be used to remove the
        frequency offset.

        Args:
            maindir (:obj:'str'): Directory that holds the digital rf and metadata.
            tleoff (:obj:'float'): Offset of the tle from the actual data.

        Returns:
            outdict (dict[str, obj]): Output data dictionary::

                {
                        't': Time in posix,
                        'dop1': Doppler frequency of 150 MHz channel from TLE ,
                        'dop2': Doppler frequency of 400 MHz channel from TLE ,
                }
    """

    #%% Get Ephem info
    # Assuming this will stay the same
    ut0 = 25567.5
    e2p = 3600.0 * 24  # ephem day to utc seconds

    sitepath = os.path.expanduser(os.path.join(maindir, "metadata/config/site"))
    sitemeta = drf.DigitalMetadataReader(sitepath)
    sdict = sitemeta.read_latest()
    sdict1 = list(sdict.values())[0]

    infopath = os.path.expanduser(os.path.join(maindir, "metadata/info"))
    infometa = drf.DigitalMetadataReader(infopath)
    idict = infometa.read_latest()
    idict1 = list(idict.values())[0]

    passpath = os.path.expanduser(os.path.join(maindir, "metadata/pass/"))
    passmeta = drf.DigitalMetadataReader(passpath)
    pdict = passmeta.read_latest()
    pdict1 = list(pdict.values())[0]
    rtime = (pdict1["rise_time"] - ut0) * e2p
    tsave = list(pdict.keys())[0]

    Dop_bw = pdict1["doppler_bandwidth"]
    t = sp.arange(0, (Dop_bw.shape[0] + 1) * 10, 10.0) + rtime
    t = t.astype(float)

    obsLoc = ephem.Observer()
    obsLoc.lat = sdict1["latitude"]
    obsLoc.long = sdict1["longitude"]

    satObj = ephem.readtle(idict1["name"], idict1["tle1"][1:-1], idict1["tle2"][1:-1])
    tephem = (t - rtime) * ephem.second + pdict1["rise_time"]

    sublat = sp.zeros_like(tephem)
    sublon = sp.zeros_like(tephem)
    for i, itime in enumerate(tephem):
        obsLoc.date = itime
        satObj.compute(obsLoc)
        sublat[i] = sp.rad2deg(satObj.sublat)
        sublon[i] = sp.rad2deg(satObj.sublong)

    # XXX Extend t vector because for the most part the velocities at the edge
    # are not changing much so to avoid having the interpolation extrapolate.
    # If extrapolation used then error messages that the time was off
    t[-1] = t[-1] + 600
    t[-2] = t[-2] + 500
    t[0] = t[0] - 240

    tdop = (t[0 : (len(t) - 1)] + t[1 : len(t)]) / 2.0
    tdop[0] = tdop[0] - 35.0
    # XXX Used this to line up inital TLE times
    tdop = tdop - tleoff

    tephem = (tdop - rtime) * ephem.second + pdict1["rise_time"]

    sublat = sp.zeros_like(tephem)
    sublon = sp.zeros_like(tephem)
    for i, itime in enumerate(tephem):
        obsLoc.date = itime
        satObj.compute(obsLoc)
        sublat[i] = sp.rad2deg(satObj.sublat)
        sublon[i] = sp.rad2deg(satObj.sublong)
    return {
        "t": t,
        "tsave": tsave,
        "dop1": sp.interpolate.interp1d(tdop, Dop_bw[:, 0], kind="cubic"),
        "dop2": sp.interpolate.interp1d(tdop, Dop_bw[:, 1], kind="cubic"),
        "sublat": sp.interpolate.interp1d(tdop, sublat, kind="cubic"),
        "sublon": sp.interpolate.interp1d(tdop, sublon, kind="cubic"),
        "site_latitude": float(sdict1["latitude"]),
        "site_longitude": float(sdict1["longitude"]),
    }


def outlier_removed_fit(m, w=None, n_iter=10, polyord=7):
    """
    Remove outliers using fited data.

    Args:
        m (:obj:`numpy array`): Phase curve.
        n_iter (:obj:'int'): Number of iteration outlier removal
        polyorder (:obj:'int'): Order of polynomial used.

    Returns:
        fit (:obj:'numpy array'): Curve with outliers removed
    """
    if w is None:
        w = sp.ones_like(m)
    W = sp.diag(sp.sqrt(w))
    m2 = sp.copy(m)
    tv = sp.linspace(-1, 1, num=len(m))
    A = sp.zeros([len(m), polyord])
    for j in range(polyord):
        A[:, j] = tv ** (float(j))
    A2 = sp.dot(W, A)
    m2w = sp.dot(m2, W)
    fit = None
    for i in range(n_iter):
        xhat = sp.linalg.lstsq(A2, m2w)[0]
        fit = sp.dot(A, xhat)
        # use gradient for central finite differences which keeps order
        resid = sp.gradient(fit - m2)
        std = sp.std(resid)
        bidx = sp.where(sp.absolute(resid) > 2.0 * std)[0]
        for bi in bidx:
            A2[bi, :] = 0.0
            m2[bi] = 0.0
            m2w[bi] = 0.0
    if debug_plot:
        plt.plot(m2, label="outlier removed")
        plt.plot(m, label="original")
        plt.plot(fit, label="fit")
        plt.legend()
        plt.ylim([sp.minimum(fit) - std * 3.0, sp.maximum(fit) + std * 3.0])
        plt.show()
    return fit


def open_file(maindir):
    """
        Creates the digital RF reading object.

        Args:
            maindir (:obj:'str'): The directory where the data is located.

        Returns:
            drfObj (obj:"DigitalRFReader"): Digital RF Reader object.
            chandict (obj:"dict"): Dictionary that holds info for the channels.
            start_indx (obj:'long'): Start index in samples.
            end_indx (obj:'long'): End index in samples.
    """
    mainpath = os.path.expanduser(maindir)
    drfObj = drf.DigitalRFReader(mainpath)
    chans = drfObj.get_channels()

    chandict = {}

    start_indx, end_indx = [0, sp.inf]
    # Get channel info
    for ichan in chans:
        curdict = {}

        curdict["sind"], curdict["eind"] = drfObj.get_bounds(ichan)
        # determine the read boundrys assuming the sampling is the same.
        start_indx = sp.maximum(curdict["sind"], start_indx)
        end_indx = sp.minimum(curdict["eind"], end_indx)

        dmetadict = drfObj.read_metadata(start_indx, end_indx, ichan)
        dmetakeys = list(dmetadict.keys())

        curdict["sps"] = dmetadict[dmetakeys[0]]["samples_per_second"]
        curdict["fo"] = dmetadict[dmetakeys[0]]["center_frequencies"].ravel()[0]
        chandict[ichan] = curdict

    return (drfObj, chandict, start_indx, end_indx)


def corr_tle_rf(
    maindir, e=None, window=2 ** 18, n_measure=100, timewin=[0, 0], tleoff=0
):
    """
        Coorelates the tle derived frequency and the rf frequency. A flag is output
        that specifies if the two frequencies correlate. A time offset between
        the two is also calculated if there is a correlation.

    """
    if e is None:
        e = ephem_doponly(maindir, tleoff)

    bw_search0 = 8e3
    bw_search1 = 20e3
    drfObj, chandict, start_indx, end_indx = open_file(maindir)

    chans = list(chandict.keys())
    sps = chandict[chans[0]]["sps"]
    start_indx = start_indx + timewin[0] * sps
    end_indx = end_indx - timewin[1] * sps
    mid_indx = (end_indx + start_indx) / 2
    start_vec = sp.linspace(-150 * sps, 150 * sps, n_measure) + mid_indx
    tvec = start_vec / sps
    del_f = sps / window
    fvec = sp.arange(-window / 2.0, window / 2.0) * del_f
    bw_indx0 = int(bw_search0 / del_f)
    bw_indx1 = int(bw_search1 / del_f)

    fidx0 = sp.arange(-bw_indx0 // 2, bw_indx0 // 2, dtype=int) + window // 2
    fidx1 = sp.arange(-bw_indx1 // 2, bw_indx1 // 2, dtype=int) + window // 2

    res0 = sp.zeros([n_measure, window], dtype=float)
    res1 = sp.zeros([n_measure, window], dtype=float)

    freqm0 = sp.zeros([n_measure])
    snr0 = sp.zeros([n_measure])
    freqm1 = sp.zeros([n_measure])
    snr1 = sp.zeros([n_measure])

    wfun = sig.get_window("hann", window)

    subchan = 0
    for i_t, c_st in enumerate(start_vec):
        # pull out data
        z00 = drfObj.read_vector(c_st, window, chans[0], subchan)
        z01 = drfObj.read_vector(c_st + window, window, chans[0], subchan)
        z10 = drfObj.read_vector(c_st, window, chans[1], subchan)
        z11 = drfObj.read_vector(c_st + window, window, chans[1], subchan)

        F0 = scfft.fftshift(scfft.fft(z00 * wfun))
        F1 = scfft.fftshift(scfft.fft(z01 * wfun))

        res_temp = F0 * F1.conj()
        res_noise = sp.median(res_temp.real ** 2 + res_temp.imag ** 2) / sp.log(2.0)
        res0[i_t, :] = res_temp.real ** 2 + res_temp.imag ** 2
        freqm0[i_t] = fvec[fidx0[sp.argmax(res0[i_t, fidx0])]]
        snr0[i_t] = res0[i_t, fidx0].max() / res_noise
        # Cross correlation for second channel and get residual frequency
        F0 = scfft.fftshift(scfft.fft(z10 * wfun))
        F1 = scfft.fftshift(scfft.fft(z11 * wfun))

        res_temp = F0 * F1.conj()
        res_noise = sp.median(res_temp.real ** 2 + res_temp.imag ** 2) / sp.log(2.0)
        res1[i_t, :] = res_temp.real ** 2 + res_temp.imag ** 2
        # find max frequency over restricted sub-band
        freqm1[i_t] = fvec[fidx1[sp.argmax(res1[i_t, fidx1])]]
        snr1[i_t] = res1[i_t, fidx1].max() / res_noise

    thresh = 15
    # outlier removal
    keep0 = 10 * sp.log10(snr0) > thresh
    if keep0.sum() == 0:
        return False, 0
    dopfit0 = sp.interpolate.interp1d(
        tvec[keep0], freqm0[keep0], fill_value="extrapolate"
    )(tvec)
    dfmean0 = sp.mean(dopfit0)
    dfstd0 = sp.std(dopfit0)

    keep1 = 10 * sp.log10(snr1) > thresh
    if keep1.sum() == 0:
        return False, 0
    dopfit1 = sp.interpolate.interp1d(
        tvec[keep1], freqm1[keep1], fill_value="extrapolate"
    )(tvec)
    dfmean1 = sp.mean(dopfit1)
    dfstd1 = sp.std(dopfit1)

    toff = window / sps
    doppler0 = e["dop1"](tvec + toff)
    dmean0 = sp.mean(doppler0)
    dstd0 = sp.std(doppler0)

    doppler1 = e["dop2"](tvec + toff)
    dmean1 = sp.mean(doppler1)
    dstd1 = sp.std(doppler1)

    # cross correlation
    xcor0 = sig.correlate(dopfit0 - dfmean0, doppler0 - dmean0) / (
        dfstd0 * dstd0 * n_measure
    )
    xcor1 = sig.correlate(dopfit1 - dfmean1, doppler1 - dmean1) / (
        dfstd1 * dstd1 * n_measure
    )

    if xcor0.max() < 0.5 or xcor1.max() < 0.5:
        rfexist = False
        tshift = 0
    else:
        rfexist = True
        corboth = xcor0.max() + xcor1.max()
        lagmean = (
            xcor0.max() * sp.argmax(xcor0) + xcor1.max() * sp.argmax(xcor1)
        ) / corboth
        deltat = tvec[1] - tvec[0]
        tshift = deltat * (n_measure - lagmean)
    return rfexist, tshift, tvec, dopfit0, dopfit1, doppler0, doppler1


def calc_resid(maindir, e, window=2 ** 13, n_measure=500, timewin=[0, 0]):
    """
        Calculate the residual difference between the Doppler frequencies from the
        TLEs and measured data.

        Args:
            maindir (:obj:'str'): The directory where the data is located.
            e (:obj:'dict'): The dictionary with information on the output
            window (:obj:'int'): Window length in samples.
            n_window (:obj:'int'): Number of windows integrated.
            bandwidth (:obj:'int'): Number of bins in lowest sub-band to find max frequency.
        Returns:
             outdict (dict[str, obj]): Output data dictionary::

                    {
                            'cspec': Correlated phase residuals,
                            'max_bin': Frequency bin with max return,
                            'doppler_residual': Doppler residual,
                            'tvec': Time vector for each measurment,
                            'fvec': Frequency values array,
                            'res1': Phase residual channel 1,
                            'res0': Phase residual channel 0
                    }
    """
    # number of Hz to search over for max
    bw_search0 = 0.5e3
    bw_search1 = 1e3

    drfObj, chandict, start_indx, end_indx = open_file(maindir)

    chans = list(chandict.keys())
    sps = chandict[chans[0]]["sps"]
    start_indx = start_indx + timewin[0] * sps
    end_indx = end_indx - timewin[1] * sps
    start_vec = sp.linspace(start_indx, end_indx - window * 2, n_measure)
    tvec = start_vec / sps
    del_f = sps / window
    fvec = sp.arange(-window / 2.0, window / 2.0) * del_f

    bw_indx0 = int(bw_search0 / del_f)
    bw_indx1 = int(bw_search1 / del_f)
    fidx0 = sp.arange(-bw_indx0 // 2, bw_indx0 // 2, dtype=int) + window // 2
    fidx1 = sp.arange(-bw_indx1 // 2, bw_indx1 // 2, dtype=int) + window // 2

    res0 = sp.zeros([n_measure, window], dtype=float)
    res1 = sp.zeros([n_measure, window], dtype=float)
    snr0 = sp.zeros(n_measure, dtype=float)
    snr1 = sp.zeros(n_measure, dtype=float)

    freqm0 = sp.zeros([n_measure])
    freqm1 = sp.zeros([n_measure])
    wfun = sig.get_window("hann", window)
    idx = sp.arange(window)
    t_win = idx / sps
    win_s = float(window) / sps
    toff = window / sps
    subchan = 0
    for i_t, c_st in enumerate(start_vec):
        t_cur = tvec[i_t]
        # pull out data
        z00 = drfObj.read_vector(c_st, window, chans[0], subchan)
        z01 = drfObj.read_vector(c_st + window, window, chans[0], subchan)
        z10 = drfObj.read_vector(c_st, window, chans[1], subchan)
        z11 = drfObj.read_vector(c_st + window, window, chans[1], subchan)
        # determine the doppler shift from the sat motion
        # d["e"]["vel2"] is a function derived from the interp1d function
        tphase = sp.float64(t_cur + toff)
        doppler0 = -e["dop1"](tphase)
        doppler1 = -e["dop2"](tphase)
        # Cross correlation for first channel and get residual frequency
        osc00 = wfun * sp.exp(1.0j * 2.0 * sp.pi * doppler0 * t_win)
        osc01 = wfun * sp.exp(1.0j * 2.0 * sp.pi * doppler0 * (t_win + win_s))
        osc10 = wfun * sp.exp(1.0j * 2.0 * sp.pi * doppler1 * t_win)
        osc11 = wfun * sp.exp(1.0j * 2.0 * sp.pi * doppler1 * (t_win + win_s))
        F0 = scfft.fftshift(scfft.fft(z00 * osc00.astype(z00.dtype)))
        F1 = scfft.fftshift(scfft.fft(z01 * osc01.astype(z01.dtype)))
        res_temp = F0 * F1.conj()
        res0[i_t, :] = res_temp.real ** 2 + res_temp.imag ** 2
        freqm0[i_t] = fvec[fidx0[sp.argmax(res0[i_t, fidx0])]]
        nc0 = sp.median(res0[i_t, :]) / sp.log(2.0)
        snr0[i_t] = res0[i_t, fidx0].max() / nc0
        # Cross correlation for second channel and get residual frequency
        F0 = scfft.fftshift(scfft.fft(z10 * osc10.astype(z10.dtype)))
        F1 = scfft.fftshift(scfft.fft(z11 * osc11.astype(z11.dtype)))
        res_temp = F0 * F1.conj()
        res1[i_t, :] = res_temp.real ** 2 + res_temp.imag ** 2
        # find max frequency over restricted sub-band
        freqm1[i_t] = fvec[fidx1[sp.argmax(res1[i_t, fidx1])]]
        # normalize residuals
        nc1 = sp.median(res1[i_t, :]) / sp.log(2.0)
        snr1[i_t] = res1[i_t, fidx1].max() / nc1

        res0[i_t, :] = res0[i_t, :] / nc0
        res1[i_t, :] = res1[i_t, :] / nc1
    tvec[0] = tvec[0] - 100
    tvec[len(tvec) - 1] = tvec[len(tvec) - 1] + 100
    # outlier removal
    snrmean = 0.5 * snr0 + 0.5 * snr1
    dopfit = outlier_removed_fit(
        0.5 * (snr0 * freqm0 * 400.0 / 150 + snr1 * freqm1) / snrmean, snrmean
    )
    # interpolate residual
    doppler_residual = sp.interpolate.interp1d(tvec, dopfit)
    # correlate residuals together
    rescor = res0 * res1.conj()
    # pdb.set_trace()
    cspec = sp.mean(rescor, axis=0)
    return {
        "cspec": cspec.real,
        "max_bin": sp.argmax(cspec),
        "doppler_residual": doppler_residual,
        "dopfit": dopfit,
        "tvec": tvec,
        "fvec": fvec,
        "res1": res1,
        "res0": res0,
    }


def calc_TEC(
    maindir,
    window=4096,
    incoh_int=100,
    sfactor=4,
    offset=0.0,
    timewin=[0, 0],
    snrmin=0.0,
):
    """
    Estimation of phase curve using coherent and incoherent integration.

    Args:
        maindir (:obj:`str`): Path for data.
        window (:obj:'int'): Window length in samples.
        incoh_int (:obj:'int'): Number of incoherent integrations.
        sfactor (:obj:'int'): Overlap factor.
        offset (:obj:'int'): Overlap factor.
        timewin ((:obj:'list'): Overlap factor.)
    Returns:
         outdict (dict[str, obj]): Output data dictionary::

             {
                        "rTEC": Relative TEC in TECU,
                        "rTEC_sig":Relative TEC STD in TECU,
                        "S4": The S4 parameter,
                        "snr0":snr0,
                        "snr1":snr1,
                        "time": Time for each measurement in posix format,
             }
    """

    e = ephem_doponly(maindir, offset)
    resid = calc_resid(maindir, e)
    Nr = int((incoh_int + sfactor - 1) * (window / sfactor))

    drfObj, chandict, start_indx, end_indx = open_file(maindir)

    chans = list(chandict.keys())
    sps = chandict[chans[0]]["sps"]
    start_indx = start_indx + timewin[0] * sps
    end_indx = end_indx - timewin[1] * sps
    freq_ratio = chandict[chans[1]]["fo"] / chandict[chans[0]]["fo"]
    om0, om1 = (
        2.0
        * s_const.pi
        * sp.array([chandict[chans[0]]["fo"], chandict[chans[1]]["fo"]])
    )
    start_vec = sp.arange(start_indx, end_indx - Nr, Nr, dtype=float)
    tvec = start_vec / sps

    soff = window / sfactor
    toff = soff / sps
    idx = sp.arange(window)
    n_t1 = sp.arange(0, incoh_int) * soff
    IDX, N_t1 = sp.meshgrid(idx, n_t1)
    Msamp = IDX + N_t1
    ls_samp = float(Msamp.flatten()[-1])

    wfun = sig.get_window("hann", window)
    wmat = sp.tile(wfun[sp.newaxis, :], (incoh_int, 1))

    phase_00 = sp.exp(1.0j * 0.0)
    phase_10 = sp.exp(1.0j * 0.0)

    phase0 = sp.zeros(len(start_vec), dtype=sp.complex64)
    phase1 = sp.zeros(len(start_vec), dtype=sp.complex64)

    phase_cs0 = sp.zeros(len(start_vec), dtype=float)
    phase_cs1 = sp.zeros(len(start_vec), dtype=float)
    snr0 = sp.zeros(len(start_vec))
    snr1 = sp.zeros(len(start_vec))

    std0 = sp.zeros(len(start_vec))
    std1 = sp.zeros(len(start_vec))
    fi = window // 2
    subchan = 0
    outspec0 = sp.zeros((len(tvec), window))
    outspec1 = sp.zeros((len(tvec), window))
    print("Start Beacon Processing")
    for i_t, c_st in enumerate(start_vec):

        update_progress(float(i_t) / float(len(start_vec)))
        t_cur = tvec[i_t]

        z00 = drfObj.read_vector(c_st, Nr, chans[0], subchan)[Msamp]
        z01 = drfObj.read_vector(c_st + soff, Nr, chans[0], subchan)[Msamp]
        z10 = drfObj.read_vector(c_st, Nr, chans[1], subchan)[Msamp]
        z11 = drfObj.read_vector(c_st + soff, Nr, chans[1], subchan)[Msamp]

        tphase = sp.float64(t_cur + toff)
        doppler0 = -1.0 * (150.0 / 400.0) * resid["doppler_residual"](t_cur) - e[
            "dop1"
        ](tphase)
        doppler1 = -1.0 * resid["doppler_residual"](t_cur) - e["dop2"](tphase)

        osc00 = phase_00 * wmat * sp.exp(1.0j * 2.0 * sp.pi * doppler0 * (Msamp / sps))
        osc01 = (
            phase_00
            * wmat
            * sp.exp(1.0j * 2.0 * sp.pi * doppler0 * (Msamp / sps + float(soff) / sps))
        )
        osc10 = phase_10 * wmat * sp.exp(1.0j * 2.0 * sp.pi * doppler1 * (Msamp / sps))
        osc11 = (
            phase_10
            * wmat
            * sp.exp(1.0j * 2.0 * sp.pi * doppler1 * (Msamp / sps + float(soff) / sps))
        )

        f00 = scfft.fftshift(scfft.fft(z00 * osc00.astype(z00.dtype), axis=-1), axes=-1)
        f01 = scfft.fftshift(scfft.fft(z01 * osc01.astype(z01.dtype), axis=-1), axes=-1)
        f00spec = sp.power(f00.real, 2).sum(0) + sp.power(f00.imag, 2).sum(0)
        outspec0[i_t] = f00spec.real
        f00_cor = f00[:, fi] * sp.conj(f01[:, fi])
        # Use prod to average the phases together.
        phase0[i_t] = sp.cumprod(sp.power(f00_cor, 1.0 / float(incoh_int)))[-1]
        phase_cs0[i_t] = sp.cumsum(sp.diff(sp.unwrap(sp.angle(f00[:, fi]))))[-1]

        f10 = scfft.fftshift(scfft.fft(z10 * osc10.astype(z10.dtype), axis=-1), axes=-1)
        f11 = scfft.fftshift(scfft.fft(z11 * osc11.astype(z11.dtype), axis=-1), axes=-1)
        f10spec = sp.power(f10.real, 2).sum(0) + sp.power(f10.imag, 2).sum(0)

        f10_cor = f10[:, fi] * sp.conj(f11[:, fi])
        outspec1[i_t] = f10spec.real
        phase1[i_t] = sp.cumprod(sp.power(f10_cor, 1.0 / float(incoh_int)))[-1]
        phase_cs1[i_t] = sp.cumsum(sp.diff(sp.unwrap(sp.angle(f10[:, fi]))))[-1]

        std0[i_t] = sp.std(sp.angle(f00_cor))
        std1[i_t] = sp.std(sp.angle(f10_cor))
        snr0[i_t] = f00spec.real[fi] / sp.median(f00spec.real)
        snr1[i_t] = f10spec.real[fi] / sp.median(f10spec.real)

        # Phases for next time through the loop
        phase_00 = phase_00 * sp.exp(
            1.0j * 2.0 * sp.pi * doppler0 * ((ls_samp + 1.0) / sps)
        )

        phase_10 = phase_10 * sp.exp(
            1.0j * 2.0 * sp.pi * doppler1 * ((ls_samp + 1.0) / sps)
        )

    #
    phasecurve = sp.cumsum(sp.angle(phase0) * freq_ratio - sp.angle(phase1))
    phasecurve_amp = phase_cs0 * freq_ratio - phase_cs1
    stdcurve = sp.sqrt(
        sp.cumsum(float(sfactor) * incoh_int * (std0 ** 2.0 + std1 ** 2.0))
    )

    # SNR windowing, picking values with minimum snr
    snrwin = sp.logical_and(snr0 > snrmin, snr1 > snrmin)
    phasecurve = phasecurve[snrwin]
    phasecurve_amp = phasecurve_amp[snrwin]
    stdcurve = stdcurve[snrwin]
    snr0 = snr0[snrwin]
    snr1 = snr1[snrwin]
    tvec = tvec[snrwin]

    dt = sp.diff(tvec).mean()
    Nside = int(1.0 / dt / 2.0)
    lvec = sp.arange(-Nside, Nside)
    Lmat, Tmat = sp.meshgrid(lvec, sp.arange(len(tvec)))
    Sampmat = Lmat + Tmat
    Sampclip = sp.clip(Sampmat, 0, len(tvec) - 1)
    eps = s_const.e ** 2 / (8.0 * s_const.pi ** 2 * s_const.m_e * s_const.epsilon_0)
    aconst = s_const.e ** 2 / (2 * s_const.m_e * s_const.epsilon_0 * s_const.c)
    na = 9.0
    nb = 24.0
    f0 = 16.668e6

    # cTEC = f0*((na*nb**2)/(na**2-nb**2))*s_const.c/(2.*s_const.pi*eps)
    cTEC = 1e-16 * sp.power(om1 / om0 ** 2 - 1.0 / om1, -1) / aconst
    rTEC = cTEC * phasecurve
    rTEC = rTEC - rTEC.min()
    rTEC_amp = cTEC * phasecurve_amp
    rTEC_amp = rTEC_amp - rTEC_amp.min()
    rTEC_sig = cTEC * stdcurve
    S4 = sp.std(snr0[Sampclip], axis=-1) / sp.median(snr0, axis=-1)

    outdict = {
        "rTEC": rTEC,
        "rTEC_amp": rTEC_amp,
        "rTEC_sig": rTEC_sig,
        "S4": S4,
        "snr0": snr0,
        "snr1": snr1,
        "time": tvec,
        "resid": resid,
        "phase": phasecurve,
        "phase_amp": phasecurve_amp,
        "phasestd": stdcurve,
        "outspec0": outspec0,
        "outspec1": outspec1,
    }
    return outdict


#%% Plotting
def plotsti_vel(
    maindir,
    savename="chancomp.png",
    timewin=[0, 0],
    offset=0,
    window=512,
    sfactor=2,
    incoh_int=10,
    Nt=512,
):
    """
        Plot the velocity data over the sti data. This can be used to determie offsets so the data is properly aligned.

        Args:
            maindir (:obj:`str`): Path for data.
            window (:obj:'int'): Window length in samples.
            incoh_int (:obj:'int'): Number of incoherent integrations.
            sfactor (:obj:'int'): Overlap factor.
    """
    # Get the frequency information
    e = ephem_doponly(maindir, offset)

    # always -0th subchannel for beacons
    subchan = 0
    dec_vec = [8, 5]
    flim = [-10, 10]
    mindb = 15.0
    Nr = int(sp.prod(dec_vec) * (incoh_int + sfactor - 1) * (window / sfactor))

    drfObj, chandict, start_indx, end_indx = open_file(maindir)

    chans = list(chandict.keys())
    sps = chandict[chans[0]]["sps"]
    start_indx = start_indx + timewin[0] * sps
    end_indx = end_indx - timewin[1] * sps
    start_vec = sp.linspace(start_indx, end_indx - Nr, Nt, dtype=float)
    tvec = start_vec / sps
    f0 = e["dop1"](tvec[::4]) * 1e-3
    f1 = e["dop2"](tvec)[::4] * 1e-3

    fvec = (
        sp.arange(-window / 2, window / 2, dtype=float)
        * sps
        / sp.prod(dec_vec)
        / window
    )

    dates = [dt.datetime.fromtimestamp(ts) for ts in tvec]
    datenums = md.date2num(dates)
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")

    soff = window // sfactor
    idx = sp.arange(window)
    n_t1 = sp.arange(0, incoh_int) * soff
    IDX, N_t1 = sp.meshgrid(idx, n_t1)
    Msamp = IDX + N_t1

    wfun = sig.get_window("hann", window)
    wmat = sp.tile(wfun[sp.newaxis, :], (incoh_int, 1))

    sti0 = sp.zeros((Nt, window), float)
    sti1 = sp.zeros_like(sti0)
    for i_t, c_st in enumerate(start_vec):

        z0 = drfObj.read_vector(c_st, Nr, chans[0], subchan)
        z1 = drfObj.read_vector(c_st, Nr, chans[1], subchan)
        for idec in dec_vec:
            z0 = sig.decimate(z0, idec)
            z1 = sig.decimate(z1, idec)

        z0 = z0[Msamp]
        z1 = z1[Msamp]

        fft0 = scfft.fftshift(scfft.fft(z0 * wmat, axis=-1), axes=-1)
        fft1 = scfft.fftshift(scfft.fft(z1 * wmat, axis=-1), axes=-1)

        psd0 = sp.sum(fft0.real ** 2 + fft0.imag ** 2, axis=0).real
        psd1 = sp.sum(fft1.real ** 2 + fft1.imag ** 2, axis=0).real
        sti0[i_t] = psd0
        sti1[i_t] = psd1

    fig1 = plt.figure(figsize=(12, 12))

    plt.subplot(221)

    mesh = plt.pcolormesh(
        datenums, fvec * 1e-3, sp.transpose(10.0 * sp.log10(sti0)), vmin=mindb
    )

    scplot = plt.plot(datenums[::4], f0, "ko")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(flim)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (kHz)")
    plt.title("Power ch0 (dB) %1.2f MHz" % (150.012))
    plt.colorbar(mesh, ax=ax)

    plt.subplot(222)

    mesh = plt.pcolormesh(
        datenums, fvec * 1e-3, sp.transpose(10.0 * sp.log10(sti0)), vmin=mindb
    )

    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(flim)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (kHz)")
    plt.title("Power ch0 (dB) %1.2f MHz" % (150.012))
    plt.colorbar(mesh, ax=ax)

    plt.subplot(223)
    mesh = plt.pcolormesh(
        datenums, fvec * 1e-3, sp.transpose(10.0 * sp.log10(sti1)), vmin=mindb
    )
    scplot = plt.plot(datenums[::4], f1, "ko")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(flim)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (kHz)")
    plt.title("Power ch1 (dB) %1.2f MHz" % (400.032))
    plt.colorbar(mesh, ax=ax)

    plt.subplot(224)
    mesh = plt.pcolormesh(
        datenums, fvec * 1e-3, sp.transpose(10.0 * sp.log10(sti1)), vmin=mindb
    )
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(flim)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (kHz)")
    plt.title("Power ch1 (dB) %1.2f MHz" % (400.032))
    plt.colorbar(mesh, ax=ax)
    plt.tight_layout()
    print("Saving RF TLE comparison figure: " + savename)
    fig1.savefig(savename, dpi=300)
    plt.close(fig1)


def plot_resid(d, savename="resfig1.png"):
    """
        Plots the residual frequency after the first wipe using the TLE velocity.
    """
    flim = [-2.0e3, 2.0e3]
    t = d["tvec"]

    dates = [dt.datetime.fromtimestamp(ts) for ts in t]
    datenums = md.date2num(dates)
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")

    fig1 = plt.figure(figsize=(7, 9))
    doppler_residual = sp.interpolate.interp1d(d["tvec"], d["dopfit"])
    fvec = d["fvec"]
    res0 = d["res0"]
    res1 = d["res1"]
    plt.subplot(211)
    mesh = plt.pcolormesh(
        datenums, fvec, sp.transpose(10.0 * sp.log10(res0 + 1e-12)), vmin=-5, vmax=25
    )
    plt.plot(
        datenums, (150.0 / 400.0) * doppler_residual(t), "r--", label="doppler resid"
    )
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(flim)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (Hz)")
    plt.title("Power ch0 (dB) %1.2f MHz" % (150.012))
    plt.legend()
    plt.colorbar(mesh, ax=ax)

    # quicklook spectra of residuals spectra along with measured Doppler residual from second channel.
    plt.subplot(212)
    mesh = plt.pcolormesh(
        datenums, fvec, sp.transpose(10.0 * sp.log10(res1 + 1e-12)), vmin=-5, vmax=25
    )
    plt.plot(datenums, doppler_residual(t), "r--", label="doppler resid")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(flim)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (Hz)")
    plt.title("Power ch1 (dB), %1.2f MHz" % (400.032))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.legend()
    plt.colorbar(mesh, ax=ax)

    plt.tight_layout()
    print("Saving residual plots: " + savename)
    plt.savefig(savename, dpi=300)
    plt.close(fig1)


def plot_measurements(outdict, savename="measured.png"):
    """
        Plots the rTEC and S4 measurements.

        Args:
             outdict (dict[str, obj]): Output data dictionary::

                 {
                            "rTEC": Relative TEC in TECU,
                            "rTEC_sig":Relative TEC STD in TECU,
                            "S4": The S4 parameter,
                            "snr0":snr0,
                            "snr1":snr1,
                            "time": Time for each measurement in posix format,
                 }
            savename (obj:'str'): Name of the file that it will be saved to.
    """

    dates = [dt.datetime.fromtimestamp(ts) for ts in outdict["time"]]
    datenums = md.date2num(dates)
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")

    fig1 = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(datenums, outdict["rTEC"], color="black")
    # plt.plot(datenums, outdict['rTEC_amp'], color="black")
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlabel("UTC")
    plt.ylabel("rTEC (TECu)")
    plt.title("Relative TEC and S4 Parameters")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax2 = ax.twinx()
    plt.plot(datenums, outdict["S4"], color="red")
    ax2.grid(True)
    ax2.set_ylabel("S4", color="r")
    ax2.tick_params("y", colors="r")

    # SNR from beacon at those times
    plt.subplot(122)
    plt.plot(datenums, 10.0 * sp.log10(outdict["snr0"]), label="150 MHz")
    plt.plot(datenums, 10.0 * sp.log10(outdict["snr1"]), label="400 MHz")
    plt.legend()
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("SNR for Both Channels")
    plt.xlabel("UTC")
    plt.ylabel("SNR (dB)")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)

    plt.tight_layout()
    print("Saving measurement plots: " + savename)
    fig1.savefig(savename, dpi=300)
    plt.close(fig1)

    fig1 = plt.figure(figsize=(15, 5))
    plt.subplot(121)

    plt.plot(datenums, outdict["rTEC_amp"], color="black")
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlabel("UTC")
    plt.ylabel("rTEC (TECu)")
    plt.title("Relative TEC and S4 Parameters")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax2 = ax.twinx()
    plt.plot(datenums, outdict["S4"], color="red")
    ax2.grid(True)
    ax2.set_ylabel("S4", color="r")
    ax2.tick_params("y", colors="r")

    # SNR from beacon at those times
    plt.subplot(122)
    plt.plot(datenums, 10.0 * sp.log10(outdict["snr0"]), label="150 MHz")
    plt.plot(datenums, 10.0 * sp.log10(outdict["snr1"]), label="400 MHz")
    plt.legend()
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("SNR for Both Channels")
    plt.xlabel("UTC")
    plt.ylabel("SNR (dB)")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)

    plt.tight_layout()
    figpath, name = os.path.split(savename)
    savename = os.path.join(figpath, "amp" + name)
    print("Saving measurement plots: " + savename)
    fig1.savefig(savename, dpi=300)
    plt.close(fig1)


def plot_map(outdict, e, savename, fig1=None, m=None):
    """
        This function will plot the output data in a scatter plot over a map of the
        satellite path.

        Args:
            outdict (dict[str, obj]): Output dictionary from analyzebeacons.
            e (dict[str, obj]): Output dictionary from ephem_doponly.
            savename(:obj:`str`): Name of the file the image will be saved to.
            fig1(:obj:'matplotlib figure'): Figure.
            m (:obj:'basemap obj'): Basemap object
    """

    t = outdict["time"]
    slat = e["site_latitude"]
    slon = e["site_longitude"]
    if fig1 is None:
        fig1 = plt.figure()
    plt.figure(fig1.number)
    latlim = [math.floor(slat - 15.0), math.ceil(slat + 15.0)]
    lonlim = [math.floor(slon - 15.0), math.ceil(slon + 15.0)]
    if m is None:
        m = Basemap(
            lat_0=slat,
            lon_0=slon,
            llcrnrlon=lonlim[0],
            llcrnrlat=latlim[0],
            urcrnrlon=lonlim[1],
            urcrnrlat=latlim[1],
        )

    m.drawcoastlines(color="gray")

    m.plot(slon, slat, "rx")
    scat = m.scatter(
        e["sublon"](t),
        e["sublat"](t),
        c=outdict["rTEC"],
        cmap="viridis",
        vmin=0,
        vmax=math.ceil(sp.nanmax(outdict["rTEC"])),
    )
    plt.title("Map of TEC Over Satellite Path")
    cb = plt.colorbar(scat, label="rTEC in TECu")
    fig1.savefig(savename, dpi=300)

    # plt.draw()

    scat = m.scatter(
        e["sublon"](t),
        e["sublat"](t),
        c=outdict["rTEC_amp"],
        cmap="viridis",
        vmin=0,
        vmax=math.ceil(sp.nanmax(outdict["rTEC_amp"])),
    )
    plt.title("Map of TEC_Amp Over Satellite Path")
    cb.set_clim(vmin=0, vmax=math.ceil(sp.nanmax(outdict["rTEC_amp"])))
    cb.draw_all()

    # plt.tight_layout()
    figpath, name = os.path.split(savename)
    savename = os.path.join(figpath, "amp" + name)
    fig1.savefig(savename, dpi=300)
    plt.close(fig1)


#%% I/O for measurements
def save_output(maindirmeta, outdict, e):
    """
        This function saves the output of the relative TEC measurement processing.

        Args:
            maindir (:obj:`str`): Path for data.
            outdict (dict[str, obj]): Output data dictionary::

                {
                           "rTEC": Relative TEC in TECU,
                           "rTEC_sig":Relative TEC STD in TECU,
                           "S4": The S4 parameter,
                           "snr0":snr0,
                           "snr1":snr1,
                           "time": Time for each measurement in posix format,
                }
        """

    if not os.path.exists(maindirmeta):
        os.mkdir(maindirmeta)

    mdo = drf.DigitalMetadataWriter(
        metadata_dir=maindirmeta,
        subdir_cadence_secs=3600,
        file_cadence_secs=1,
        sample_rate_numerator=1,
        sample_rate_denominator=1,
        file_name="Outputparams",
    )
    # get rid of doppler residual
    if "doppler_residual" in outdict["resid"]:
        del outdict["resid"]["doppler_residual"]
    mdo.write(e["tsave"], outdict)


def readoutput(maindirmeta):
    """
        This function reads the saved output of the relative TEC measurement processing.
        If the processed data directory doesn't exist it returns None.

        Args:
            maindir (:obj:`str`): Path for RF data.

        Returns:
            outdict (dict[str, obj]): Output data dictionary. If the directory
            doesn't exists it returns None::

                {
                           "rTEC": Relative TEC in TECU,
                           "rTEC_sig":Relative TEC STD in TECU,
                           "S4": The S4 parameter,
                           "snr0":snr0,
                           "snr1":snr1,
                           "time": Time for each measurement in posix format,
                }
        """
    try:
        dmeta = drf.DigitalMetadataReader(maindirmeta)
    except IOError:
        return None
    metadict = dmeta.read_latest()
    outdict = list(metadict.values())[0]

    return outdict


#%% Run from commandline material
def analyzebeacons(input_args):
    """
        This function will run the analysis code and save the data. Plots will be
        made as well if desired.
    """
    # makes sure theres no trailing / for the path
    mainpath = os.path.expanduser(os.path.dirname(os.path.join(input_args.path, "")))
    maindirmeta = input_args.newdir
    if maindirmeta is None:
        maindirmeta = os.path.join(maindirmeta, "Processed")
    if not os.path.exists(maindirmeta):
        os.mkdir(maindirmeta)
    figspath = os.path.join(maindirmeta, "Figures")
    if not os.path.exists(figspath):
        os.mkdir(figspath)
    if input_args.savename is None:
        savename = os.path.join(figspath, "BeaconPlots.png")
    # rfexist, tleoff = corr_tle_rf(mainpath,timewin=[input_args.begoff, input_args.endoff])
    rfexist = True
    tleoff = input_args.tleoffset
    e = ephem_doponly(mainpath)
    if input_args.justplots or not rfexist:
        print("Analysis will not be run, only plots will be made.")
        plotsti_vel(
            mainpath,
            savename=os.path.join(figspath, "chancomp.png"),
            timewin=[input_args.begoff, input_args.endoff],
            offset=tleoff,
        )
        outdict = readoutput(maindirmeta)
        if outdict is None:
            print("No ouptut data exists")
        else:
            plot_resid(outdict["resid"], os.path.join(figspath, "resid.png"))
            plot_measurements(outdict, savename)
            plot_map(outdict, e, os.path.join(figspath, "mapdata.png"))
    else:

        outdict = calc_TEC(
            mainpath,
            window=input_args.window,
            incoh_int=input_args.incoh,
            sfactor=input_args.overlap,
            offset=tleoff,
            timewin=[input_args.begoff, input_args.endoff],
            snrmin=input_args.minsnr,
        )
        print("Saving everything to digital metadata.")
        outdict["window"] = input_args.window
        outdict["incoherent_integrations"] = input_args.incoh
        outdict["Overlap"] = input_args.overlap
        outdict["Time_Offset"] = tleoff
        outdict["Beginning_Offset"] = input_args.begoff
        outdict["Ending_Offset"] = input_args.endoff
        outdict["Min_SNR"] = input_args.minsnr
        if os.path.exists(maindirmeta):
            shutil.rmtree(maindirmeta)
        save_output(maindirmeta, outdict, e)

        if input_args.drawplots:
            print("Plotting data.")
            if not os.path.exists(figspath):
                os.mkdir(figspath)
            plotsti_vel(
                mainpath,
                savename=os.path.join(figspath, "chancomp.png"),
                timewin=[input_args.begoff, input_args.endoff],
                offset=tleoff,
            )
            plot_resid(outdict["resid"], os.path.join(figspath, "resid.png"))
            plot_measurements(outdict, savename)
            plot_map(outdict, e, os.path.join(figspath, "mapdata.png"))


def parse_command_line(str_input=None):
    """
        This will parse through the command line arguments
    """
    # if str_input is None:
    parser = argparse.ArgumentParser()
    # else:
    #     parser = argparse.ArgumentParser(str_input)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="prints debug output and additional detail.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="beacons.ini",
        help="Use configuration file <config>.",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        default=None,
        help="Path to the Digital RF files and meta data.",
    )
    parser.add_argument(
        "-d",
        "--drawplots",
        default=False,
        dest="drawplots",
        action="store_true",
        help="Bool to determine if plots will be made and saved.",
    )
    parser.add_argument(
        "-s", "--savename", dest="savename", default=None, help="Name of plot file."
    )
    parser.add_argument(
        "-w",
        "--window",
        dest="window",
        default=4096,
        type=int,
        help="Length of window in samples for FFT in calculations.",
    )
    parser.add_argument(
        "-i",
        "--incoh",
        dest="incoh",
        default=100,
        type=int,
        help="Number of incoherent integrations in calculations.",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        dest="overlap",
        default=4,
        type=int,
        help="Overlap for each of the FFTs.",
    )
    parser.add_argument(
        "-t",
        "--tleoffset",
        dest="tleoffset",
        default=0.0,
        type=float,
        help="Offset of the TLE time from the actual pass.",
    )
    parser.add_argument(
        "-b",
        "--begoff",
        dest="begoff",
        default=0.0,
        type=float,
        help="Number of seconds to jump ahead before measuring.",
    )
    parser.add_argument(
        "-e",
        "--endoff",
        dest="endoff",
        default=0.0,
        type=float,
        help="Number of seconds to jump ahead before measuring.",
    )
    parser.add_argument(
        "-m",
        "--minsnr",
        dest="minsnr",
        default=0.0,
        type=float,
        help="Minimum SNR for for phase curve measurement",
    )
    parser.add_argument(
        "-j",
        "--justplots",
        action="store_true",
        dest="justplots",
        default=False,
        help="Makes plots for input, residuals, and final measurements if avalible.",
    )
    parser.add_argument(
        "-n",
        "--newdir",
        dest="newdir",
        default=None,
        help="Directory that measured data will be saved.",
    )

    if str_input is None:
        return parser.parse_args()
    else:
        return parser.parse_args(str_input)


if __name__ == "__main__":
    """
        Main way run from command line
    """
    args_commd = parse_command_line()

    if args_commd.path is None:
        print("Please provide an input source with the -p option!")
        sys.exit(1)

    analyzebeacons(args_commd)
