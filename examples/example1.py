# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""example1.py used used to test bug relating to need for setting c array order

$Id$
"""
# standard python imports
import datetime
import time

# third party imports
import numpy

# Millstone imports
import digital_rf


data = numpy.array([[0.00074581,  0.00997215],
                    [-0.00222403, -0.00974955],
                    [0.00365252,  0.00930909],
                    [-0.00499937, -0.00866062],
                    [0.00623448,  0.00781865],
                    [-0.00733026, -0.006802],
                    [0.00826225,  0.0056334],
                    [-0.00900963, -0.00433896],
                    [0.00955571,  0.00294761]])

print(data)
print(data.shape)

voltage = numpy.array([0.00074581 + 0.00997215j,
                       -0.00222403 + -0.00974955j,
                       0.00365252 + 0.00930909j,
                       -0.00499937 + -0.00866062j,
                       0.00623448 + 0.00781865j,
                       -0.00733026 + -0.006802j,
                       0.00826225 + 0.0056334j,
                       -0.00900963 + -0.00433896j,
                       0.00955571 + 0.00294761j], dtype=numpy.complex)


data_ri = []
data_ri.append(numpy.real(voltage))
data_ri.append(numpy.imag(voltage))
data_ri = numpy.transpose(numpy.array(data_ri))

print(data_ri)
print(data_ri.shape)

# constants
save_dir_data = '/tmp/test/chb'
sample_rate_numerator = long(1.0E6)
sample_rate_denominator = 1
sample_rate = numpy.longdouble(sample_rate_numerator) / sample_rate_denominator
start_time = datetime.datetime(2017, 1, 1)
t = long(numpy.uint64(time.mktime(start_time.timetuple()) * sample_rate))


drf_out2 = digital_rf.DigitalRFWriter(save_dir_data, 'f', 3600, 1000, t, sample_rate_numerator, sample_rate_denominator,
                                      'drf_dummy_data_v0.1.py', 0, False, True, 1,
                                      False, False)
drf_out2.rf_write(data_ri.astype('f4'))
drf_out2.close()

testReadObj = digital_rf.DigitalRFReader(['/tmp/test'])
start_index, end_index = testReadObj.get_bounds('chb')
blocks = testReadObj.get_continuous_blocks(start_index, end_index, 'chb')
for measurement_index in blocks:
    data = testReadObj.read_vector(
        int(measurement_index), int(blocks[measurement_index]), 'chb')
    print(data)
    print(data.dtype)
