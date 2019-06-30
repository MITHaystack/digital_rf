.. -*- mode: rst -*-

|GitHub|_ |PyPI|_ |Conda|_ |Travis|_ |AppVeyor|_

.. |GitHub| image:: https://badge.fury.io/gh/MITHaystack%2Fdigital_rf.svg
.. _GitHub: https://badge.fury.io/gh/MITHaystack%2Fdigital_rf

.. |PyPI| image:: https://badge.fury.io/py/digital-rf.svg
.. _PyPI: https://pypi.python.org/pypi/digital-rf/

.. |Conda| image:: https://anaconda.org/ryanvolz/digital_rf/badges/version.svg
.. _Conda: https://anaconda.org/ryanvolz/digital_rf

.. |Travis| image:: https://travis-ci.org/MITHaystack/digital_rf.svg?branch=master
.. _Travis: https://travis-ci.org/MITHaystack/digital_rf

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/vt741g5mjsh841ai?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/ryanvolz/digital-rf

|DigitalRF|
===========

.. |DigitalRF| image:: docs/digital_rf_logo.png
    :alt: Digital RF
    :width: 100%

The Digital RF project encompasses a standardized HDF5 format for reading and writing of radio frequency data and the software for doing so. The format is designed to be self-documenting for data archive and to allow rapid random access for data processing. For details on the format, refer to the 'documents' directory in the source tree.

This suite of software includes libraries for reading and writing data in the Digital RF HDF5 format in C (``libdigital_rf``), Python (``digital_rf``) with blocks for GNU Radio (``gr_digital_rf``), and MATLAB. It also contains the ``thor.py`` UHD radio recorder script, Python tools for managing and processing Digital RF data, example scripts that demonstrate basic usage, and example applications that encompass a complete data recording and processing chain for various use cases.


Important Links
===============

:Official source code repo: https://github.com/MITHaystack/digital_rf
:Issue tracker: https://github.com/MITHaystack/digital_rf/issues
:User mailing list for help/questions: openradar-users@openradar.org
:Developer mailing list: openradar-developers@openradar.org


Citation
========

If you use digital_rf in a scientific publication, we would appreciate a citation such as the following:

    Volz, R., Rideout, W. C., Swoboda, J., Vierinen, J. P., & Lind, F. D. (2018). Digital RF (Version 2.6.1). MIT Haystack Observatory. Retrieved from https://github.com/MITHaystack/digital_rf

.. code-block:: bibtex

    @software{DigitalRF,
    author = {Volz, Ryan and Rideout, William C. and Swoboda, John and Vierinen, Juha P. and Lind, Frank D.}
    title = {Digital {{RF}}},
    url = {https://github.com/MITHaystack/digital_rf},
    version = {2.6.1},
    publisher = {{MIT Haystack Observatory}},
    date = {2018-08-07},
    }


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
  * hdf5 >= 1.8 (``libhdf5-dev``)
  * mako (``python-mako``)
  * numpy (``python-numpy``)
  * pkgconfig (``python-pkgconfig``)
  * python 2.7 or 3.5+ (``python-dev``)
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
  * python 2.7 or 3.5+ (``python``)
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

This will install the ``digital_rf`` and ``gr_digital_rf`` Python packages and GNU Radio Companion (GRC) blocks. If you're interested in the C library or development, see below for ways to install the full project package.

If you plan on using Digital RF with GNU Radio, make sure to run the `pip` command in the same Python environment that your GNU Radio installation uses so that GNU Radio can find the packages. Depending on your GNU Radio installation, it may be necessary to add the Digital RF blocks to your GRC blocks path by creating or editing the GRC configuration file

:Unix (local): $HOME/.gnuradio/config.conf
:Windows (local): %APPDATA%/.gnuradio/config.conf
:Unix (global): /etc/gnuradio/conf.d/grc.conf
:Custom (global): {INSTALL_PREFIX}/etc/gnuradio/conf.d/grc.conf

to contain::

    [grc]
    local_blocks_path = {PIP_PREFIX}/share/gnuradio/grc/blocks

(replacing ``{PIP_PREFIX}`` with the pip installation prefix, "/usr/local" for example).


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

The toolbox package will then be found at "build/matlab/digital_rf.mltbx".


Using Conda package
-------------------

Alternatively, you can install digital_rf using our Conda_ binary package. Our package is compatible with the `conda-forge <https://conda-forge.github.io/>`_ distribution of community-maintained packages.

In an existing Conda environment, run the following to install ``digital_rf`` and its dependencies::

    conda config --add channels ryanvolz
    conda config --add channels conda-forge
    conda install digital_rf

Using MacPorts
--------------

Digital RF can be installed though MacPorts, using the port install command::

    sudo ports install digital_rf

This will install and build all of the needed dependencies using MacPorts.

Example Usage
=============

Python and C examples can be found in the examples directory in the source tree. The C examples can be compiled from the build directory by running::

    make examples


The following Python commands will load and read data located in a directory "/data/test".

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

The C and tests create test files in "/tmp/hdf5*". To cleanup afterward, run::

    rm -r /tmp/hdf5*

The python tests require ``pytest`` to run. From the source directory, you can simply run::

    pytest


Acknowledgments
===============

This work was supported by the National Science Foundation under the Geospace Facilities and MRI programs, and by National Instruments / Ettus corporation through the donation of software radio hardware. We are grateful for the support that made this development possible.
