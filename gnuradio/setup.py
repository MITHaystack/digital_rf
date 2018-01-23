# ----------------------------------------------------------------------------
# Copyright (c) 2018 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Setup file for the gr_digital_rf package."""
from setuptools import setup
import os
# to use a consistent encoding
from codecs import open

# read __version__ variable by exec-ing python/_version.py
version = {}
try:
    with open(os.path.join('python', '_version.py')) as fp:
        exec(fp.read(), version)
except IOError:
    version['__version__'] = None

setup(
    name='gr_digital_rf',
    version=version['__version__'],
    description=(
        'Read and write files in Digital RF format of HDF5 using GNU Radio.'
    ),
    long_description=(
        'Digital RF is a disk storage and archival format for radio signals.'
        ' It uses HDF5 files with embedded metadata and a predictable naming'
        ' scheme to produce a self-describing data format suitable to a'
        ' variety of use cases.'
    ),

    url='https://github.com/MITHaystack/digital_rf',

    author='MIT Haystack Observatory',
    author_email='openradar-developers@openradar.org',

    license='BSD-3-Clause',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Other Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],

    keywords='hdf5 radio rf',

    install_requires=['digital_rf'],

    package_dir={
        'gr_digital_rf': 'python',
    },
    packages=['gr_digital_rf'],
    data_files=[
        ('share/gnuradio/grc/blocks', [
            'grc/gr_digital_rf_digital_rf_channel_sink.xml',
            'grc/gr_digital_rf_digital_rf_channel_source.xml',
            'grc/gr_digital_rf_digital_rf_sink.xml',
            'grc/gr_digital_rf_digital_rf_source.xml',
        ]),
    ],
    scripts=[
        'apps/thor.py',
    ],
)
