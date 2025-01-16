.. -*- mode: rst -*-

|GitHub|_ |PyPI|_ |Conda|_ |CI|_

.. |GitHub| image:: https://badge.fury.io/gh/MITHaystack%2Fdigital_rf.svg
.. _GitHub: https://badge.fury.io/gh/MITHaystack%2Fdigital_rf

.. |PyPI| image:: https://badge.fury.io/py/digital-rf.svg
.. _PyPI: https://pypi.python.org/pypi/digital-rf/

.. |Conda| image:: https://anaconda.org/conda-forge/digital_rf/badges/version.svg
.. _Conda: https://anaconda.org/conda-forge/digital_rf

.. |CI| image:: https://github.com/MITHaystack/digital_rf/workflows/conda-matrix/badge.svg
.. _CI: https://github.com/MITHaystack/digital_rf/actions?query=workflow%3Aconda-matrix+branch%3Amaster

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

If you use Digital RF in a scientific publication, we would appreciate a citation such as the following (BibTeX_):

    Volz, R., Rideout, W. C., Swoboda, J., Vierinen, J. P., & Lind, F. D. (2025). Digital RF (Version 2.6.10). MIT Haystack Observatory. Retrieved from https://github.com/MITHaystack/digital_rf

.. _BibTeX: bibtex.bib


Dependencies
============

The main package components are divided into subdirectories by language (C, Python, and MATLAB) and can be built and installed separately or all together. Their individual dependencies are listed below by component.

Build
-----

all
  * cmake >= 3.20 (``cmake``)

c
  * hdf5 >= 1.8 (``libhdf5-dev``)

python
  * build (``python3-build`` or ``pip install build``)
  * hdf5 >= 1.8 (``libhdf5-dev``)
  * mako (``python-mako``)
  * numpy (``python-numpy``)
  * python 3.8+ (``python-dev``)
  * scikit-build-core (``python3-scikit-build-core`` or ``pip install scikit-build-core``)
  * setuptools-scm (``python3-setuptools-scm`` or ``pip install setuptools-scm``)

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
  * python 3.8+ (``python``)
  * python-dateutil (``python-dateutil``)
  * six (``python-six``)

matlab
  * MATLAB >= R2014b

Runtime [optional feature]
--------------------------

python
  * gnuradio [gr_digital_rf] (``gnuradio``)
  * gr-osmosdr [thorosmo] (``gr-osmosdr``)
  * gr-iio [thorpluto] (``gr-iio``)
  * gr-uhd [thor] (``libgnuradio-uhd``)
  * matplotlib [tools] (``python-matplotlib``)
  * pandas [digital_metadata] (``python-pandas``)
  * pytest >= 3 [tests] (``python-pytest``)
  * python-sounddevice [tools] (``pip install sounddevice``)
  * scipy [tools] (``python-scipy``)
  * uhd [uhdtodrf] (``python3-uhd``)
  * watchdog [mirror, ringbuffer, watchdog] (``python-watchdog``)


Installation
============

If you're just getting started with Digital RF, we recommend using the Conda_ binary package. It is available in the `conda-forge <https://conda-forge.github.io/>`_ distribution of community-maintained packages.

In an existing Conda environment, run the following to install ``digital_rf`` and its dependencies::

    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install digital_rf

You may also want to install the ``gnuradio-core`` package in order to make use of ``gr_digital_rf``::

    conda install gnuradio-core

Using PyPI package (wheel)
--------------------------

Alternatively, you can most likely install Digital RF through PyPI_ with `pip` using a pre-built wheel::

    pip install digital_rf

This will install the ``digital_rf`` and ``gr_digital_rf`` Python packages and GNU Radio Companion (GRC) blocks. (If you're interested in the C library or development, see other installation methods for ways to install the full project package.)

If you plan on using Digital RF with GNU Radio, make sure to run the `pip` command in the same Python environment that your GNU Radio installation uses so that GNU Radio can find the packages. Depending on your GNU Radio installation, it may be necessary to add the Digital RF blocks to your GRC blocks path by creating or editing the GRC configuration file

:Unix (local): $HOME/.gnuradio/config.conf
:Windows (local): %APPDATA%/.gnuradio/config.conf
:Unix (global): /etc/gnuradio/conf.d/grc.conf
:Custom (global): {INSTALL_PREFIX}/etc/gnuradio/conf.d/grc.conf

to contain::

    [grc]
    local_blocks_path = {PIP_PREFIX}/share/gnuradio/grc/blocks

(replacing ``{PIP_PREFIX}`` with the pip installation prefix, "/usr/local" for example).

Using MacPorts
--------------

Digital RF can be installed though MacPorts, using the port install command::

    sudo ports install digital_rf

This will install and build all of the needed dependencies using MacPorts.

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

CMake will attempt to find your Python installation in the usual places. If this fails, you can specify a particular Python interpreter by adding ``-DPython_EXECUTABLE={PATH}`` (replacing ``{PATH}`` with the interpreter path) to the cmake command.

Finally, you may need to update the library cache so the newly-installed ``libdigital_rf`` is found::

    sudo ldconfig

Note that it is also possible to build the different language libraries separately by following the CMake build procedure from within the `c` and `matlab` directories. The `python` package can be built and installed on its own using any Python build frontend compatible with `pyproject.toml`, e.g.::

    python -m build

The MATLAB toolbox is not created by default. If you have MATLAB R2016a or higher and want to create an installable toolbox package, run the following from the build directory::

    make matlab

The toolbox package will then be found at "build/matlab/digital_rf.mltbx".


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
