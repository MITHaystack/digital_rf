#!/usr/bin/env python2
"""
:platform: Unix, Mac
:synopsis: This module processes the beacon receiver data in DigitalRF format to
get geophysical parameters and saves the output to digital metadata. Derived from
jitter code from Juha Vierinen.


"""


import os
import sys
import time
import datetime as dt
import argparse
import pdb
import scipy as sp
import scipy.fftpack as scfft
import scipy.signal as sig
import scipy.constants as s_const

import matplotlib.pyplot as plt
import matplotlib.dates as md
# Millstone imports
import digital_rf as drf

TLE_def = ('CASSIOPE','1 39265U 13055A   17015.93102647 +.00001768 +00000-0 +52149-4 0  9992',
                          '2 39265 080.9701 172.4450 0693099 333.2926 023.4045 14.21489964169876')
debug_plot = False

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
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
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def ephem_doponly(maindir,tleoff=10.):
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
    e2p = 3600.*24#ephem day to utc seconds
    passpath = os.path.expanduser(os.path.join(maindir, 'metadata/pass/'))
    passmeta = drf.DigitalMetadataReader(passpath)
    pdict = passmeta.read_latest()
    pdict1 = pdict[pdict.keys()[0]]
    rtime = (pdict1['rise_time']-ut0)*e2p

    Dop_bw = pdict1['doppler_bandwidth']
    t = sp.arange(0,(Dop_bw.shape[0]+1)*10, 10.)+rtime
    t = t.astype(float)

    # XXX Extend t vector because for the most part the velocities at the edge
    # are not changing much so to avoid having the interpolation extrapolate.
    # If extrapolation used then error messages that the time was off
    t[-1] = t[-1]+600
    t[-2] = t[-2]+500
    t[0] = t[0]-70

    tdop = (t[0:(len(t)-1)]+t[1:len(t)])/2.0
    tdop[0] = tdop[0]-35.0
    # XXX Used this to line up inital TLE times
    tdop = tdop-tleoff
    return({"t":t,
            "dop1":sp.interpolate.interp1d(tdop, Dop_bw[:,0], kind="cubic"),
            "dop2":sp.interpolate.interp1d(tdop, Dop_bw[:,1], kind="cubic")})

def outlier_removed_fit(m, n_iter=10, polyord=7):
    """
    Remove outliers using fited data.

    Args:
        m (:obj:`numpy array`): Phase curve.
        n_iter (:obj:'int'): Number of iteration outlier removal
        polyorder (:obj:'int'): Order of polynomial used.

    Returns:
        fit (:obj:'numpy array'): Curve with outliers removed
    """
    m2 = sp.copy(m)
    tv = sp.linspace(-1,1, num=len(m))
    A = sp.zeros([len(m), polyord])
    for j in range(polyord):
        A[:,j] = tv**(float(j))
    A2 = sp.copy(A)
    fit = None
    for i in range(n_iter):
        xhat = sp.linalg.lstsq(A2,m2)[0]
        fit = sp.dot(A,xhat)
        resid = (fit - m2)
        std = sp.std(resid)
        bidx = sp.where(sp.absolute(resid) > 2.0*std)[0]
        for bi in bidx:
            A2[bi,:]=0.0
            m2[bi]=0.0
    if debug_plot:
        plt.plot(m2,label="outlier removed")
        plt.plot(m,label="original")
        plt.plot(fit,label="fit")
        plt.legend()
        plt.ylim([sp.minimum(fit)-std*3.0,sp.maximum(fit)+std*3.0])
        plt.show()
    return(fit)

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

    chandict={}

    start_indx, end_indx=[0, sp.inf]
    # Get channel info
    for ichan in chans:
        curdict = {}
        chanmeta = drfObj.get_digital_rf_metadata(ichan)
        curdict['sps'] = chanmeta['samples_per_second']

        curdict['sind'], curdict['eind'] = drfObj.get_bounds(ichan)
        # determine the read boundrys assuming the sampling is the same.
        start_indx = sp.maximum(curdict['sind'], start_indx)
        end_indx = sp.minimum(curdict['eind'], end_indx)

        dmetaObj = drfObj.get_digital_metadata(ichan)
        fsamp, lsamp = dmetaObj.get_bounds()
        dmetadict = dmetaObj.read(fsamp, lsamp)
        dmetakeys = dmetadict.keys()

        curdict['fo'] = dmetadict[dmetakeys[0]]['center_frequencies'][0][0]
        chandict[ichan] = curdict

    return (drfObj, chandict, start_indx, end_indx)

def calc_resid(maindir,e,window=2**13,n_measure=500):
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
    bw_search = 6e3


    drfObj, chandict, start_indx, end_indx = open_file(maindir)

    chans = chandict.keys()
    sps = chandict[chans[0]]['sps']
    start_vec = sp.linspace(start_indx, end_indx-window*2, n_measure)
    tvec = start_vec/sps
    del_f = sps/window
    fvec = sp.arange(-window/2.0, window/2.0)*del_f

    bw_indx = int(bw_search/del_f)
    fidx = sp.arange(-bw_indx/2, bw_indx/2, dtype=int) + window/2


    res0 = sp.zeros([n_measure, window], dtype=sp.complex64)
    res1 = sp.zeros([n_measure, window], dtype=sp.complex64)

    freqm = sp.zeros([n_measure])

    wfun = sig.get_window('hann', window)
    idx = sp.arange(window)

    toff = window/sps
    subchan = 0
    for i_t, c_st in enumerate(start_vec):
        t_cur = tvec[i_t]
        #pull out data
        z00 = drfObj.read_vector(c_st, window, chans[0])[:, subchan]
        z01 = drfObj.read_vector(c_st+window, window, chans[0])[:, subchan]
        z10 = drfObj.read_vector(c_st, window, chans[1])[:, subchan]
        z11 = drfObj.read_vector(c_st+window, window, chans[1])[:, subchan]
        # determine the doppler shift from the sat motion
        # d["e"]["vel2"] is a function derived from the interp1d function
        doppler0 = -e["dop1"](t_cur+toff)
        doppler1 = -e["dop2"](t_cur+toff)
        # Cross correlation for first channel and get residual frequency
        osc00 = wfun*sp.exp(1.0j*2.0*sp.pi*doppler0*(idx/sps))
        osc01 = wfun*sp.exp(1.0j*2.0*sp.pi*doppler0*(idx/sps+ float(window)/sps))
        osc10 = wfun*sp.exp(1.0j*2.0*sp.pi*doppler1*(idx/sps))
        osc11 = wfun*sp.exp(1.0j*2.0*sp.pi*doppler1*(idx/sps+ float(window)/sps))
        F0 = scfft.fftshift(scfft.fft(z00*osc00.astype(z00.dtype)))
        F1 = scfft.fftshift(scfft.fft(z01*osc01.astype(z01.dtype)))
        res0[i_t, :] = F0*sp.conj(F1)
        # Cross correlation for second channel and get residual frequency
        F0 = scfft.fftshift(scfft.fft(z10*osc10.astype(z10.dtype)))
        F1 = scfft.fftshift(scfft.fft(z11*osc11.astype(z01.dtype)))
        res1[i_t, :] = F0*sp.conj(F1)
        # find max frequency over restricted sub-band
        freqm[i_t] = fvec[fidx[sp.argmax(sp.absolute(res1[i_t, fidx]))]]
        #normalize residuals
        res0[i_t, :] = res0[i_t, :]/sp.median(sp.absolute(res0[i_t, :]))
        res1[i_t, :] = res1[i_t, :]/sp.median(sp.absolute(res1[i_t, :]))
    tvec[0] = tvec[0] - 100
    tvec[len(tvec)-1] = tvec[len(tvec)-1] + 100
    #outlier removal
    dopfit = outlier_removed_fit(freqm)
    #interpolate residual
    doppler_residual = sp.interpolate.interp1d(tvec, dopfit)
    # correlate residuals together
    cspec = sp.mean(sp.absolute(res0*sp.conj(res1)), axis=0)
    return({"cspec":cspec, "max_bin":sp.argmax(cspec),
            "doppler_residual":doppler_residual,
            "tvec":tvec, "fvec":fvec, "res1":res1, "res0":res0})

def plot_resid(d,savename='resfig1.png'):
    """
        Plots the residual frequency after the first wipe using the TLE velocity.
    """
    flim = [-2.e3, 2.e3]
    t = d['e']['t']
    tlim = [t[1], t[-3]]

    datenums = md.date2num(dates)
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')

    fig1 = plt.figure(figsize=(15,5))
    tvec = d["resid"]["tvec"]
    dates = [dt.datetime.fromtimestamp(ts) for ts in tvec]
    fvec = d["resid"]["fvec"]
    res0 = d["resid"]["res0"]
    res1 = d["resid"]["res1"]
    plt.subplot(121)
    plt.pcolormesh(datenums, fvec, sp.transpose(10.*sp.log10(sp.absolute(res0))), vmin=-5, vmax=25)
    plt.plot(datenums,(150.0/400.0)*d["resid"]["doppler_residual"](tvec), "k--", label="doppler resid")
    plt.ylim(flim)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (Hz)")
    plt.title("Power ch0 (dB) %1.2f MHz"%(150.012))
    plt.legend()
    plt.colorbar(orientation="horizontal")

    plt.xlim(tlim)
     # quicklook spectra of residuals spectra along with measured Doppler residual from second channel.
    plt.subplot(122)
    plt.pcolormesh(datenums, fvec, sp.transpose(10.*sp.log10(sp.absolute(res1))), vmin=-5, vmax=25)
    plt.plot(datenums,d["resid"]["doppler_residual"](tvec),"k--",label="doppler resid")
    plt.ylim(flim)
    plt.xlabel("UTC")
    plt.ylabel("Frequency (Hz)")
    plt.title("Power ch1 (dB), %1.2f MHz"%(400.032))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    plt.colorbar(orientation="horizontal")
    plt.legend()
    plt.xlim(tlim)
    plt.savefig(savename)
    plt.close(fig1)

def plot_measurements(outdict,savename='measured.png'):
    """
        Plots the rTEC and S4 measurements.
    """

    dates = [dt.datetime.fromtimestamp(ts) for ts in outdict['time']]
    datenums = md.date2num(dates)
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')

    fig1 = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(datenums, outdict['rTEC'], color="black")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlabel("UTC")
    plt.ylabel("rTEC (TECu)")
    plt.legend()
    plt.title("Relative TEC and S4 Parameters")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax2 = ax.twinx()
    plt.plot(datenums, outdict['S4'], color='red')
    ax2.set_ylabel('S4', color='r')
    ax2.tick_params('y', colors='r')

    # SNR from beacon at those times
    plt.subplot(122)
    plt.plot(datenums, 10.0*sp.log10(outdict["snr0"]), label="150 MHz")
    plt.plot(datenums, 10.0*sp.log10(outdict["snr1"]), label="400 MHz")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("SNR for Both Channels")
    plt.xlabel("UTC")
    plt.ylabel("SNR (dB)")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)

    plt.tight_layout()

    fig1.savefig(savename)
    plt.close(fig1)

def calc_TEC(maindir, window=4096, incoh_int=100, sfactor=4, offset=0.,timewin=[0,0],snrmin=0.):
    """
    Estimation of phase curve using coherent and incoherent integration.

    Args:
        maindir (:obj:`str`): Path for data.
        window (:obj:'int'): Window length in samples.
        incoh_int (:obj:'int'): Number of incoherent integrations.
        sfactor (:obj:'int'): Overlap factor.
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
    Nr = int((incoh_int+sfactor-1)*(window/sfactor))

    drfObj, chandict, start_indx, end_indx = open_file(maindir)

    chans = chandict.keys()
    sps = chandict[chans[0]]['sps']
    start_indx = start_indx + timewin[0]*sps
    end_indx = end_indx - timewin[1]*sps
    freq_ratio = chandict[chans[0]]['fo']/chandict[chans[1]]['fo']
    start_vec = sp.arange(start_indx, end_indx-Nr, Nr, dtype=sp.float64)
    tvec = start_vec/sps

    soff = window/sfactor
    toff = soff/sps
    idx = sp.arange(window)
    n_t1 = sp.arange(0, incoh_int)* soff
    IDX, N_t1 = sp.meshgrid(idx, n_t1)
    Msamp = IDX + N_t1

    wfun = sig.get_window('hann', window)
    wmat = sp.tile(wfun[sp.newaxis, :], (incoh_int, 1))

    phase_00 = sp.exp(1.0j*0.0)
    phase_10 = sp.exp(1.0j*0.0)

    phase0 = sp.zeros(len(start_vec), dtype=sp.complex64)
    phase1 = sp.zeros(len(start_vec), dtype=sp.complex64)

    snr0 = sp.zeros(len(start_vec))
    snr1 = sp.zeros(len(start_vec))

    std0 = sp.zeros(len(start_vec))
    std1 = sp.zeros(len(start_vec))
    fi = window/2
    subchan = 0
    print("Start Beacon Processing")
    for i_t, c_st in enumerate(start_vec):

        update_progress(float(i_t)/float(len(start_vec)))
        t_cur = tvec[i_t]

        z00 = drfObj.read_vector(c_st, Nr, chans[0])[Msamp, subchan]
        z01 = drfObj.read_vector(c_st+soff, Nr, chans[0])[Msamp, subchan]
        z10 = drfObj.read_vector(c_st, Nr, chans[1])[Msamp, subchan]
        z11 = drfObj.read_vector(c_st+soff, Nr, chans[1])[Msamp, subchan]


        doppler0 = -1.0*(150.0/400.0)*resid["doppler_residual"](t_cur) - e["dop1"](t_cur+toff)
        doppler1 = -1.0*resid["doppler_residual"](t_cur) - e["dop2"](t_cur+toff)

        osc00 = phase_00*wmat*sp.exp(1.0j*2.0*sp.pi*doppler0*(Msamp/sps))
        osc01 = phase_00*wmat*sp.exp(1.0j*2.0*sp.pi*doppler0*(Msamp/sps+ float(window)/sps))
        osc10 = phase_10*wmat*sp.exp(1.0j*2.0*sp.pi*doppler1*(Msamp/sps))
        osc11 = phase_10*wmat*sp.exp(1.0j*2.0*sp.pi*doppler1*(Msamp/sps+ float(window)/sps))


        F0 = scfft.fftshift(scfft.fft(z00*osc00.astype(z00.dtype), axis=-1), axes=-1)
        F = scfft.fftshift(scfft.fft(z01*osc01.astype(z01.dtype), axis=-1), axes=-1)
        F0spec = sp.power(F0.real, 2).sum(0)+ sp.power(F0.imag, 2).sum(0)

        F0_cor = F0[:, fi]*sp.conj(F[:, fi])
        phase0[i_t] = F0_cor.sum()


        F1 = scfft.fftshift(scfft.fft(z10*osc10.astype(z10.dtype), axis=-1), axes=-1)
        F = scfft.fftshift(scfft.fft(z11*osc11.astype(z11.dtype), axis=-1), axes=-1)
        F1spec = sp.power(F1.real, 2).sum(0)+ sp.power(F1.imag, 2).sum(0)

        F1_cor = F1[:, fi]*sp.conj(F[:, fi])
        phase1[i_t] = F1_cor.sum()

        std0[i_t] = sp.std(sp.angle(F0_cor))
        std1[i_t] = sp.std(sp.angle(F1_cor))
        snr0[i_t] = F0spec.real[fi]/sp.median(F0spec.real)
        snr1[i_t] = F1spec.real[fi]/sp.median(F1spec.real)

        phase_00 = phase_00*sp.exp(1.0j*2.0*sp.pi*doppler0*(IDX.flatten()[-1]/sps))

        phase_10 = phase_10*sp.exp(1.0j*2.0*sp.pi*doppler1*(IDX.flatten()[-1]/sps))

    phasecurve = float(incoh_int)*sp.cumsum(sp.angle(phase1)*freq_ratio-sp.angle(phase0))
    stdcurve = sp.sqrt(sp.cumsum(float(sfactor)*incoh_int*(std0**2.0 + std1**2.0)))

    # SNR windowing, picking values with minimum snr
    snrwin = sp.logical_and(snr0 > snrmin, snr1 > snrmin)
    phasecurve = phasecurve[snrwin]
    stdcurve = stdcurve[snrwin]
    snr0 = snr0[snrwin]
    snr1 = snr1[snrwin]
    tvec = tvec[snrwin]

    dt=sp.diff(tvec).mean()
    Nside = int(1./dt/2.)
    lvec = sp.arange(-Nside,Nside)
    Lmat, Tmat =sp.meshgrid(lvec, sp.arange(len(tvec)))
    Sampmat = Lmat+Tmat
    Sampclip = sp.clip(Sampmat, 0, len(tvec)-1)
    eps = s_const.e**2/(8.*s_const.pi**2*s_const.m_e*s_const.epsilon_0)
    na = 9.
    nb = 24.
    f0 = 16.668e6

    cTEC = f0*((na*nb**2)/(na**2-nb**2))*s_const.c/(2.*s_const.pi*eps)
    rTEC = cTEC*(phasecurve-phasecurve.max())*1e-16
    rTEC_sig = cTEC*stdcurve*1e-16
    S4 = sp.std(snr0[Sampclip], axis=-1)/sp.median(snr0, axis=-1)

    outdict = {'rTEC':rTEC, 'rTEC_sig':rTEC_sig, 'S4':S4, 'snr0':snr0,
               'snr1':snr1, 'time':tvec}
    return outdict

def save_output(maindir, outdict):
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
    mainpath = os.path.expanduser(os.path.dirname(maindir))+'_Processed'
    if not os.path.exists(mainpath):
        os.mkdir(mainpath)

    mdo = drf.DigitalMetadataWriter(
        metadata_dir=mainpath,
        subdir_cadence_secs=3600,
        file_cadence_secs=1,
        sample_rate_numerator=1,
        sample_rate_denominator=1,
        file_name='Outputparams',
        )
    mdo.write(outdict['time'][0], outdict)

def analyzebeacons(input_args):
    """
        This function will run the analysis code and save the data. Plots will be
        made as well if desired.
    """
    mainpath = os.path.expanduser(os.path.dirname(input_args.path))
    if input_args.savename is None:
        savename = os.path.join(mainpath, 'BeaconPlots.png')
    outdict = calc_TEC(input_args.path, window=input_args.window,
                       incoh_int=input_args.incoh, sfactor=input_args.overlap,
                       offset=input_args.tleoffset, timewin=[input_args.begoff,input_args.endoff],
                       snrmin=input_args.minsnr)
    print("Saving everything to digital metadata.")
    outdict['window'] = input_args.window
    outdict['incoherent_integrations'] = input_args.incoh
    outdict['Overlap'] = input_args.overlap
    outdict['Time_Offset'] = input_args.tleoffset
    outdict['Beginning_Offset'] = input_args.begoff
    outdict['Ending_Offset'] = input_args.endoff
    outdict['Min_SNR'] = input_args.minsnr
    save_output(input_args.path, outdict)

    if input_args.drawplots:
        print("Plotting data.")
        savename = os.path.expanduser(input_args.savename)
        plot_measurements(outdict, savename)

def parse_command_line(str_input=None):
    """
        This will parse through the command line arguments
    """
    if str_input is None:
        parser = argparse.ArgumentParser()
    else:
        parser = argparse.ArgumentParser(str_input)

    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="prints debug output and additional detail.")
    parser.add_argument("-c", "--config", dest="config", default='beacons.ini',
                        help="Use configuration file <config>.")
    parser.add_argument('-p', "--path", dest='path',
                        default=None, help='Path to the Digital RF files and meta data.')
    parser.add_argument('-d', "--drawplots", default=False, dest='drawplots', action="store_true",
                        help="Bool to determine if plots will be made and saved.")
    parser.add_argument('-s', "--savename", dest='savename', default='measured.png',
                        help='Name of plot file.')
    parser.add_argument('-w', "--window", dest='window', default=4096, type=int,
                        help='Length of window in samples for FFT in calculations.')
    parser.add_argument('-i', "--incoh", dest='incoh', default=100, type=int,
                        help='Number of incoherent integrations in calculations.')
    parser.add_argument('-o', "--overlap", dest='overlap', default=4, type=int,
                        help='Overlap for each of the FFTs.')
    parser.add_argument('-t', "--tleoffset", dest='tleoffset', default=0., type=float,
                        help="Offset of the TLE time from the actual pass.")
    parser.add_argument('-b', "--begoff", dest='begoff', default=0., type=float,
                        help="Number of seconds to jump ahead before measuring.")
    parser.add_argument('-e', "--endoff", dest='endoff', default=0., type=float,
                        help="Number of seconds to jump ahead before measuring.")
    parser.add_argument('-m', "--minsnr", dest='minsnr', default=0., type=float,
                        help="Minimum SNR for for phase curve measurement")

    return parser.parse_args()



if __name__ == '__main__':
    """
        Main way run from command line
    """
    args_commd = parse_command_line()

    if args_commd.path is None:
        print "Please provide an input source with the -p option!"
        sys.exit(1)

    analyzebeacons(args_commd)
