.. -*- mode: rst -*-

|GitHub|_ |PyPI|_ |Conda|_ |Travis|_

.. |GitHub| image:: https://badge.fury.io/gh/MITHaystack%2Fdigital_rf.svg
.. _GitHub: https://badge.fury.io/gh/MITHaystack%2Fdigital_rf

.. |PyPI| image:: https://badge.fury.io/py/digital-rf.svg
.. _PyPI: https://pypi.python.org/pypi/digital-rf/

.. |Conda| image:: https://anaconda.org/ryanvolz/digital_rf/badges/version.svg
.. _Conda: https://anaconda.org/ryanvolz/digital_rf

.. |Travis| image:: https://travis-ci.org/MITHaystack/digital_rf.svg?branch=master
.. _Travis: https://travis-ci.org/MITHaystack/digital_rf

|DigitalRF|
===========

.. |DigitalRF| image:: docs/digital_rf_logo.png
    :alt: Digital RF
    :width: 100%

The Digital RF project encompasses a standardized HDF5 format for reading and writing of radio frequency data and the software for doing so. The format is designed to be self-documenting for data archive and to allow rapid random access for data processing. For details on the format, refer to the 'documents' directory in the source tree.

This suite of software includes libraries for reading and writing data in the Digital RF HDF5 format in C (``libdigital_rf``), Python (``digital_rf``) with blocks for GNU Radio (``gr_digital_rf``), and MATLAB. It also contains the `thor` UHD radio recorder script, Python tools for managing and processing Digital RF data, example scripts that demonstrate basic usage, and example applications that encompass a complete data recording and processing chain for various use cases.


Important Links
===============

Official source code repo: https://github.com/MITHaystack/digital_rf

Issue tracker: https://github.com/MITHaystack/digital_rf/issues

User mailing list for help/questions: openradar-users@openradar.org

Developer mailing list: openradar-developers@openradar.org


Dependencies
============

The main package components are divided into subdirectories by language (C, Python, and MATLAB) and can be built and installed separately or all together. Their individual dependencies are listed below by component.

Build
-----

all
  * cmake >= 3.0 (``cmake``)

c
  * hdf5 >= 1.8 (``libhdf5-dev``)

python
  * Cheetah (``python-cheetah``)
  * hdf5 >= 1.8 (``libhdf5-dev``)
  * numpy (``python-numpy``)
  * pkgconfig (``python-pkgconfig``)
  * python == 2.7 (``python-dev``)
  * setuptools (``python-setuptools``)

matlab
  * cmake >= 3.0 (``cmake``)
  * MATLAB >= R2016a

Runtime
-------

c
  * hdf5 >= 1.8 (``libhdf5``)

python
  * h5py (``python-h5py``)
  * hdf5 >= 1.8 (``libhdf5``)
  * numpy (``python-numpy``)
  * packaging (``python-packaging``)
  * python == 2.7 (``python``)
  * python-dateutil (``python-dateutil``)
  * pytz (``python-tz``)
  * six (``python-six``)

matlab
  * MATLAB >= R2014b

Runtime [optional feature]
--------------------------

python
  * gnuradio [gr_digital_rf] (``gnuradio``)
  * gr-uhd [thor] (``libgnuradio-uhd``)
  * matplotlib [tools] (``python-matplotlib``)
  * pandas [digital_metadata] (``python-pandas``)
  * pytest >= 3 [tests] (``python-pytest``)
  * python-sounddevice [tools] (``pip install sounddevice``)
  * scipy [tools] (``python-scipy``)
  * watchdog [mirror, ringbuffer, watchdog] (``python-watchdog``)


Installation
============

If you're just getting started with Digital RF, we recommend using the Python package. The easiest way to install it is through PyPI_ with `pip`::

    pip install digital_rf

If you're interested in the C library or development, see below for ways to install the full project package.


Using source code package
-------------------------

First, ensure that you have the above-listed dependencies installed.

Clone the repository and enter the source directory::

    git clone https://github.com/MITHaystack/digital_rf.git
    cd digital_rf

Create a build directory to keep the source tree clean::

    mkdir build
    cd build

Build and install::

    cmake ..
    make
    sudo make install

Finally, you may need to update the library cache so the newly-installed ``libdigital_rf`` is found::

    sudo ldconfig

Note that it is also possible to build the different language libraries separately by following the CMake build procedure from within the `c`, `matlab`, and `python` directories.


The MATLAB toolbox is not created by default. If you have MATLAB R2016a or higher and want to create an installable toolbox package, run the following from the build directory::

    make matlab

The toolbox package will then be found at `build/matlab/digital_rf.mltbx`.


Using Conda package
-------------------

Alternatively, you can install digital_rf using our Conda_ binary package. Our package is compatible with the `conda-forge <https://conda-forge.github.io/>`_ distribution of community-maintained packages.

In an existing Conda environment, run the following to install digital_rf and its dependencies::

    conda config --add channels ryanvolz
    conda config --add channels conda-forge
    conda install digital_rf


Example Usage
=============

Python and C examples can be found in the examples directory in the source tree. The C examples can be compiled from the build directory by running::

    make examples


The following Python commands will load and read data located in a directory '/data/test'.

Load the module and create a reader object::

    import digital_rf as drf
    do = drf.DigitalRFReader('/data/test')

List channels::

    do.get_channels()

Get data bounds for channel 'cha'::

    s, e = do.get_bounds('cha')

Read first 10 samples from channel 'cha'::

    data = do.read_vector(s, 10, 'cha')


Testing
=======

To execute the C test suite, run the following from the build directory::

    make test

The python tests found in the tests directory in the source tree can be run directly after ``digital_rf`` has been installed.

Both the C and python tests create test files in '/tmp/hdf5*'. To cleanup afterward, run::

    rm -r /tmp/hdf5*


Acknowledgments
===============

This work was supported by the National Science Foundation under the Geospace Facilities and MRI programs, and by National Instruments / Ettus corporation through the donation of software radio hardware. We are grateful for the support that made this development possible.
