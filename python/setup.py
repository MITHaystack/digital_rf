# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Setup file for the digital_rf package."""
import os
import sys
import warnings
# to use a consistent encoding
from codecs import open

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Get the long description from the README file
with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

# read __version__ variable by exec-ing python/_version.py
version = {}
with open(os.path.join('digital_rf', '_version.py')) as fp:
    exec(fp.read(), version)


# subclass build_ext so we only add build settings for dependencies
# at build time
class build_ext(_build_ext):
    def _add_build_settings(self):
        # get numpy settings for extension (importing numpy)
        try:
            import numpy
            np_includes = numpy.get_include()
        except (ImportError, AttributeError):
            # if numpy is not installed get the headers from the .egg directory
            import numpy.core
            np_includes = os.path.join(
                os.path.dirname(numpy.core.__file__), 'include',
            )
        np_config = dict(
            include_dirs=[np_includes]
        )

        # get hdf5 settings for extension (importing pkgconfig)
        hdf5_config = dict(
            include_dirs=[os.path.join(sys.prefix, 'include')],
            library_dirs=[os.path.join(sys.prefix, 'lib')],
            libraries=['hdf5']
        )
        if not sys.platform.startswith('win'):
            hdf5_config['include_dirs'].extend([
                '/opt/local/include',
                '/usr/local/include',
            ])
            hdf5_config['library_dirs'].extend([
                '/opt/local/lib',
                '/usr/local/lib',
            ])
        HDF5_ROOT = os.getenv('HDF5_ROOT', None)
        if HDF5_ROOT is not None:
            # use specified HDF5_ROOT if available (as environment variable)
            hdf5_config['include_dirs'] = [os.path.join(HDF5_ROOT, 'include')]
            hdf5_config['library_dirs'] = [os.path.join(HDF5_ROOT, 'lib')]
        else:
            # try pkg-config
            try:
                import pkgconfig
            except ImportError:
                warnings.warn(
                    'python-pkgconfig not installed and HDF5_ROOT not'
                    ' specified, using default include and library path for'
                    ' HDF5'
                )
            else:
                if pkgconfig.exists('hdf5'):
                    hdf5_pkgconfig = pkgconfig.parse('hdf5')
                    for k in ('include_dirs', 'library_dirs', 'libraries'):
                        hdf5_config[k] = list(hdf5_pkgconfig[k])
                else:
                    warnings.warn(
                        'pkgconfig cannot find HDF5 and HDF5_ROOT not'
                        ' specified, using default include and library path'
                        ' for HDF5'
                    )

        # update extension settings
        for c in (np_config, hdf5_config):
            for k, v in c.items():
                getattr(self, k).extend(v)

    def finalize_options(self):
        _build_ext.finalize_options(self)
        self._add_build_settings()


setup(
    name='digital_rf',
    version=version['__version__'],
    description='Python tools to read/write Digital RF data in HDF5 format',
    long_description=long_description,

    url='https://github.com/MITHaystack/digital_rf',

    author='MIT Haystack Observatory',
    author_email='openradar-developers@openradar.org',

    license='BSD-3-Clause',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ],

    keywords='hdf5 radio rf',

    install_requires=[
        'h5py', 'numpy', 'packaging', 'python-dateutil', 'pytz', 'six',
        'sounddevice', 'watchdog',
    ],
    setup_requires=['numpy', 'pkgconfig'],

    packages=[
        'digital_rf',
        'gr_digital_rf',
    ],
    data_files=[
        ('share/gnuradio/grc/blocks', [
            'grc/gr_digital_rf_digital_rf_channel_sink.xml',
            'grc/gr_digital_rf_digital_rf_channel_source.xml',
            'grc/gr_digital_rf_digital_rf_sink.xml',
            'grc/gr_digital_rf_digital_rf_source.xml',
        ]),
    ],
    ext_modules=[
        # extension settings without external dependencies
        Extension(
            name='digital_rf._py_rf_write_hdf5',
            sources=['lib/py_rf_write_hdf5.c', 'lib/rf_write_hdf5.c'],
            include_dirs=['include'],
            library_dirs=[],
            libraries=[]
        ),
    ],
    entry_points={
        'console_scripts': ['drf=digital_rf.drf_command:main'],
    },
    scripts=[
        'tools/digital_metadata_archive.py',
        'tools/digital_rf_archive.py',
        'tools/digital_rf_upconvert.py',
        'tools/drf_cross_sti.py',
        'tools/drf_plot.py',
        'tools/drf_sti.py',
        'tools/drf_sound.py',
        'tools/thor.py',
        'tools/verify_digital_rf_upconvert.py',
    ],

    cmdclass={
        'build_ext': build_ext,
    },
)
