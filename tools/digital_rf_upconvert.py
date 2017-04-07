#!python

"""digital_rf_upconvert will convert a deprecated digital rf format to the newest one

Use verify_digital_rf_upconvert.py if you want to test the conversion.

$Id$
"""

# standard python imports
import os
import sys
import argparse
import traceback
import glob
import shutil

# third party imports
import numpy

# Millstone imports
import digital_rf
import digital_rf_deprecated_hdf5  # for reading old formatter

read_len = 1000000  # default read len


### main begins here ###
if __name__ == '__main__':

    # command line interface
    parser = argparse.ArgumentParser(
        description='digital_rf_upconvert will convert a deprecated digital rf format to the newest one.')
    parser.add_argument('--source', metavar='sourceDir',
                        help='Source top level directory containing Digital RF channels to be converted', required=True)
    parser.add_argument('--target', metavar='targetDir',
                        help='Target top level directory where Digital RF channels will be written after upconversion', required=True)
    parser.add_argument('--dir_secs', metavar='Subdir_cadence_seconds',
                        help='Number of seconds of data to store per subdirectory. Default is 3600 (one hour)',
                        default=3600, type=int)
    parser.add_argument('--file_millisecs', metavar='File_cadence_milliseconds',
                        help='Number of milliseconds of data to store per file. Default is 1000 (one second)',
                        default=1000, type=int)
    parser.add_argument('--gzip', metavar='GZIP compression 0-9.',
                        help='Level of GZIP compression 1-9.  Default=0, for no compression',
                        type=int, default=0)
    parser.add_argument('--checksum', action='store_true',
                        help='Set this flag to turn on Hdf5 checksums')
    args = parser.parse_args()

    try:
        reader = digital_rf_deprecated_hdf5.read_hdf5(args.source)
    except:
        traceback.print_exc()
        sys.exit(-1)

    # convert each channel separately
    for channel in reader.get_channels():
        print('working on channel %s' % (channel))
        metaDict = reader.get_rf_file_metadata(channel)
        bounds = reader.get_bounds(channel)

        # this code only works if the sample rate is an integer
        sample_rate = metaDict['sample_rate'][0]
        if math.fmod(sample_rate, 1.0) != 0.0:
            raise ValueError, 'Cannot guess the numerator and denominator with a fractional sample rate %f' % (
                sample_rate)
        sample_rate_numerator = long(sample_rate)
        sample_rate_denominator = long(1)

        # read critical metadata
        is_complex = bool(metaDict['is_complex'][0])
        num_subchannels = metaDict['num_subchannels'][0]

        # get first sample to find dtype
        if is_complex:
            data = reader.read_vector(bounds[0], 1, channel)
        else:
            data = reader.read_vector_raw(bounds[0], 1, channel)
        cont_blocks = reader.get_continuous_blocks(
            bounds[0], bounds[1], channel)
        subdir = os.path.join(args.target, channel)
        if not os.access(subdir, os.R_OK):
            os.makedirs(subdir)

        # copy in any metadata* files
        metadataFiles = glob.glob(os.path.join(
            args.source, channel, 'metadata*.h5'))
        for f in metadataFiles:
            shutil.copy(f, subdir)

        # get dtype to convert to if complex
        if is_complex:
            if data.dtype == numpy.complex64:
                this_dtype = numpy.float32
            elif data.dtype == numpy.complex128:
                this_dtype = numpy.float64
            else:
                raise ValueError, 'Unexpected complex type %s' % (
                    str(data.dtype))
        else:
            this_dtype = data.dtype

        # create a drf 2 writer
        writer = digital_rf.DigitalRFWriter(subdir, this_dtype, args.dir_secs, args.file_millisecs, bounds[0],
                                            sample_rate_numerator, sample_rate_denominator, str(metaDict[
                                                                                                'uuid_str']),
                                            is_complex=is_complex,
                                            num_subchannels=num_subchannels,
                                            compression_level=args.gzip, checksum=args.checksum,
                                            marching_periods=False)

        if not os.access(subdir, os.R_OK):
            os.mkdir(subdir)

        # write all the data
        for startSample, sampleLen in cont_blocks:
            thisSample = startSample
            endSample = (startSample + sampleLen) - 1
            while thisSample < endSample:
                this_len = min((endSample - thisSample) + 1, read_len)

                # convert if complex
                if is_complex:
                    data = reader.read_vector(thisSample, this_len, channel)
                    new_data = numpy.zeros(
                        (len(data), 2 * num_subchannels), dtype=this_dtype)
                    new_data[:, 0:2 * num_subchannels:2] = data.real
                    new_data[:, 1:2 * num_subchannels:2] = data.imag
                    writer.rf_write(new_data, thisSample - bounds[0])
                else:
                    data = reader.read_vector_raw(
                        thisSample, this_len, channel)
                    writer.rf_write(data, thisSample - bounds[0])
                thisSample += this_len

        writer.close()
