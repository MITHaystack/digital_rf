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
import re
import sys

# to use a consistent encoding
from codecs import open

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

import versioneer


def localpath(*args):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *args))


# Get the long description from the README file
with open(localpath("README.rst"), encoding="utf-8") as f:
    long_description = f.read()

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

# library files to include in a binary distribution
# (rely on specifying this with environment variables when it's needed)
external_libs = []
external_libs_env = os.getenv("DRF_PACKAGE_EXTERNAL_LIBS", None)
if sys.platform.startswith("win") and external_libs_env is not None:
    external_lib_list = list(filter(None, external_libs_env.split(";")))
    external_libs.extend([("Lib/site-packages/digital_rf", external_lib_list)])
    if external_lib_list:
        istr = (
            "INFO: external libraries included by DRF_PACKAGE_EXTERNAL_LIBS:" " {0}"
        ).format(external_lib_list)
        print(istr)

# h5py spec to require (helpful for Windows where we rely on the hdf5.dll
# provided by h5py since we require h5py anyway, but in order to make sure
# versions match we need to require a specific h5py version)
h5py_spec_env = os.getenv("DRF_H5PY_SPEC", None)
if h5py_spec_env is None:
    h5py_spec = "h5py"
else:
    h5py_spec = h5py_spec_env
    print("INFO: h5py specified by DRF_H5PY_SPEC: {0}".format(h5py_spec))


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

            np_includes = os.path.join(os.path.dirname(numpy.core.__file__), "include")
        np_config = dict(include_dirs=[np_includes])

        # get hdf5 settings for extension (importing pkgconfig)
        HDF5_ROOT = os.getenv("HDF5_ROOT", sys.prefix)
        hdf5_config = dict(
            include_dirs=[os.path.join(HDF5_ROOT, "include")],
            library_dirs=[os.path.join(HDF5_ROOT, "lib")],
            libraries=["hdf5"],
            define=[],
        )
        if sys.platform.startswith("win"):
            hdf5_config["define"].extend(
                [("_HDF5USEDLL_", None), ("H5_BUILT_AS_DYNAMIC_LIB", None)]
            )
        else:
            hdf5_config["include_dirs"].extend(
                ["/opt/local/include", "/usr/local/include", "/usr/include"]
            )
            hdf5_config["library_dirs"].extend(
                ["/opt/local/lib", "/usr/local/lib", "/usr/lib"]
            )
        # try pkg-config to override default settings
        try:
            import pkgconfig
        except ImportError:
            infostr = (
                "INFO: python-pkgconfig not installed. Defaulting to" ' HDF5_ROOT="{0}"'
            )
            print(infostr.format(HDF5_ROOT))
        else:
            try:
                hdf5_exists = pkgconfig.exists("hdf5")
            except EnvironmentError:
                infostr = (
                    "INFO: pkg-config not installed. Defaulting to" ' HDF5_ROOT="{0}"'
                )
                print(infostr.format(HDF5_ROOT))
            else:
                if hdf5_exists:
                    hdf5_pkgconfig = pkgconfig.parse("hdf5")
                    for k in ("include_dirs", "library_dirs", "libraries"):
                        hdf5_config[k] = list(hdf5_pkgconfig[k])
                else:
                    infostr = (
                        "INFO: pkgconfig cannot find HDF5. Defaulting to"
                        ' HDF5_ROOT="{0}"'
                    )
                    print(infostr.format(HDF5_ROOT))
        # use environment variables to override discovered settings
        hdf5_env_config = dict(
            include_dirs="HDF5_INCLUDE_DIRS",
            library_dirs="HDF5_LIBRARY_DIRS",
            libraries="HDF5_LIBRARIES",
            define="HDF5_DEFINE",
        )
        for k, e in hdf5_env_config.items():
            env_val = os.getenv(e, None)
            if env_val is not None:
                val_list = list(filter(None, env_val.split(";")))
                used = set()
                vals = []
                for v in val_list:
                    if v not in used:
                        used.add(v)
                        if k == "define":
                            items = v.split("=", 2)
                            if items[0].startswith("-D"):
                                items[0] = items[0][2:]
                            if len(items) == 1:
                                vals.append((items[0], None))
                            else:
                                vals.append(tuple(items))
                        else:
                            vals.append(v)
                # update hdf5_config
                hdf5_config[k] = vals
                print("INFO: {0}={1}".format(e, vals))

        # update extension settings
        for c in (np_config, hdf5_config):
            for k, v in c.items():
                cur = getattr(self, k)
                if cur is not None:
                    cur.extend(i for i in v if i not in cur)
                else:
                    setattr(self, k, v)

    def _convert_abspath_libraries(self):
        if sys.platform.startswith("win"):
            libname_re = re.compile("(?P<libname>.*)")
        else:
            libname_re = re.compile("^lib(?P<libname>.*)")
        for k, lib in enumerate(self.libraries):
            if os.path.isabs(lib):
                libdir, libfile = os.path.split(lib)
                libfilename, _ = os.path.splitext(libfile)
                libname = libname_re.sub("\g<libname>", libfilename)
                # replace library entry with its name and add dir to path
                self.libraries[k] = libname
                if libdir not in self.library_dirs:
                    self.library_dirs.append(libdir)

    def run(self):
        self._add_build_settings()
        self._convert_abspath_libraries()
        _build_ext.run(self)


cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=build_ext)

setup(
    name="digital_rf",
    version=versioneer.get_version(),
    description="Python tools to read/write Digital RF data in HDF5 format",
    long_description=long_description,
    url="https://github.com/MITHaystack/digital_rf",
    author="MIT Haystack Observatory",
    author_email="openradar-developers@openradar.org",
    license="BSD-3-Clause",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: C",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="hdf5 radio rf",
    install_requires=[
        h5py_spec,
        "numpy",
        "packaging",
        "python-dateutil",
        "pytz",
        "six",
        "watchdog",
    ],
    setup_requires=pytest_runner,
    tests_require=["pytest>=3"],
    extras_require={
        "all": ["matplotlib", "pandas", "sounddevice", "scipy"],
        "dataframe": ["pandas"],
        "plot": ["matplotlib", "scipy"],
        "sound": ["sounddevice"],
    },
    packages=["digital_rf", "gr_digital_rf"],
    data_files=[
        (
            "share/gnuradio/grc/blocks",
            [
                "grc/gr_digital_rf_digital_rf_channel_sink.xml",
                "grc/gr_digital_rf_digital_rf_channel_source.xml",
                "grc/gr_digital_rf_digital_rf_sink.xml",
                "grc/gr_digital_rf_digital_rf_source.xml",
                "grc/gr_digital_rf_raster_aggregate.xml",
                "grc/gr_digital_rf_raster_chunk.xml",
                "grc/gr_digital_rf_raster_select.xml",
                "grc/gr_digital_rf_raster_tag.xml",
                "grc/gr_digital_rf_vector_aggregate.xml",
                "grc/gr_digital_rf_digital_rf_channel_sink.block.yml",
                "grc/gr_digital_rf_digital_rf_channel_source.block.yml",
                "grc/gr_digital_rf_digital_rf_sink.block.yml",
                "grc/gr_digital_rf_digital_rf_source.block.yml",
                "grc/gr_digital_rf_raster_aggregate.block.yml",
                "grc/gr_digital_rf_raster_chunk.block.yml",
                "grc/gr_digital_rf_raster_select.block.yml",
                "grc/gr_digital_rf_raster_tag.block.yml",
                "grc/gr_digital_rf_vector_aggregate.block.yml",
                "grc/gr_digital_rf.tree.yml",
            ],
        )
    ]
    + external_libs,
    ext_modules=[
        # extension settings without external dependencies
        Extension(
            name="digital_rf._py_rf_write_hdf5",
            sources=["lib/py_rf_write_hdf5.c", "lib/rf_write_hdf5.c"],
            include_dirs=list(
                filter(
                    None,
                    [
                        localpath("include"),
                        (
                            localpath("include/windows")
                            if sys.platform.startswith("win")
                            else None
                        ),
                    ],
                )
            ),
            library_dirs=[],
            libraries=list(
                filter(None, ["m" if not sys.platform.startswith("win") else None])
            ),
            define_macros=list(
                filter(
                    None,
                    [
                        (
                            ("digital_rf_EXPORTS", None)
                            if sys.platform.startswith("win")
                            else None
                        )
                    ],
                )
            ),
        )
    ],
    entry_points={"console_scripts": ["drf=digital_rf.drf_command:main"]},
    scripts=[
        "tools/digital_metadata_archive.py",
        "tools/digital_rf_archive.py",
        "tools/digital_rf_upconvert.py",
        "tools/digital_rf_update_properties.py",
        "tools/drf_cross_sti.py",
        "tools/drf_plot.py",
        "tools/drf_sti.py",
        "tools/drf_sound.py",
        "tools/thor.py",
        "tools/thorosmo.py",
        "tools/thorpluto.py",
        "tools/uhdtodrf.py",
        "tools/verify_digital_rf_upconvert.py",
    ],
    cmdclass=cmdclass,
)
