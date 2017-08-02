#!/usr/bin/env python
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
import os
import sys
import argparse
import subprocess
import shutil
def parse_command_line(str_input=None):
    """
        This will parse through the command line arguments
    """
    # if str_input is None:
    parser = argparse.ArgumentParser()
    # else:
    #     parser = argparse.ArgumentParser(str_input)
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="prints debug output and additional detail.")
    parser.add_argument("-c", "--config", dest="config", default='',
                        help="Text based file that will hold the TLE offsets and directories")
    parser.add_argument('-p', "--path", dest='path',
                        default=None, help='Path, where all the data is kept.')
    parser.add_argument('-d', "--drawplots", default=False, dest='drawplots', action="store_true",
                        help="Bool to determine if plots will be made and saved.")
    parser.add_argument('-s', "--savename", dest='savename', default=None,
                        help='Name of plot file.')
    parser.add_argument('-w', "--window", dest='window', default=4096, type=int,
                        help='Length of window in samples for FFT in calculations.')
    parser.add_argument('-i', "--incoh", dest='incoh', default=100, type=int,
                        help='Number of incoherent integrations in calculations.')
    parser.add_argument('-o', "--overlap", dest='overlap', default=4, type=int,
                        help='Overlap for each of the FFTs.')
    parser.add_argument('-b', "--begoff", dest='begoff', default=0., type=float,
                        help="Number of seconds to jump ahead before measuring.")
    parser.add_argument('-e', "--endoff", dest='endoff', default=0., type=float,
                        help="Number of seconds to jump ahead before measuring.")
    parser.add_argument('-m', "--minsnr", dest='minsnr', default=0., type=float,
                        help="Minimum SNR for for phase curve measurement")
    parser.add_argument('-j', "--justplots", action="store_true",
                        dest="justplots", default=False,
                        help="Makes plots for input, residuals, and final measurements if avalible.")
    parser.add_argument('-n', "--newdir", dest="newdir", default=None,
                        help='Directory that measured data will be saved.')

    if str_input is None:
        return parser.parse_args()
    else:
        return parser.parse_args(str_input)


def beacon_list(input_args):
    """ """
    # split lines into lists of directory names and time shifts

    lines = [line.rstrip('\n').split(' ') for line in open(input_args.config)]

    if input_args.justplots:
        figsdir = os.path.join(input_args.newdir, 'Figures')
        if os.path.exist(figsdir):
            os.mkdir(figsdir)

    for iline in lines:
        curdir = iline[0]
        curtleoff = iline[1]
        rfdir = os.path.join(input_args.path, curdir)
        newdir = os.path.join(input_args.newdir, curdir)
        cmd = 'python beacon_analysis.py -p {0} -d -n {1} -t {2} -m 3 -b 120 -e 120'.format(rfdir, newdir, curtleoff)
        if input_args.justplots:
            cmd = cmd+' -j'
        subprocess.call(cmd, shell=True)
        if input_args.justplots:
            oldfig = os.path.join(newdir, 'Figures', 'chancomp.png')
            newfile = os.path.join(figsdir, curdir+'.png')
            shutil.copy(oldfig, newfile)
if __name__ == '__main__':
    """
        Main way run from command line
    """
    args_commd = parse_command_line()

    if args_commd.path is None:
        print "Please provide an input source with the -p option!"
        sys.exit(1)
    if args_commd.config == '':
        print "Please provide input for -c option!"
        sys.exit(1)


    beacon_list(args_commd)
