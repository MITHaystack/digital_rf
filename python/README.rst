The Digital RF project encompasses a standardized HDF5 format for reading and writing of radio frequency data and the software for doing so. The format is designed to be self-documenting for data archive and to allow rapid random access for data processing.

This package includes:

* ``digital_rf`` Python package
* Tools for managing and processing Digital RF data
* ``gr_digital_rf`` Python package for interfacing with GNU Radio
* GNU Radio Companion blocks
* ``thor.py`` UHD radio recorder script
* Example scripts and applications

Digital RF C and MATLAB libraries can be found at the `official source code repository <https://github.com/MITHaystack/digital_rf>`_. To build from source, you must have the HDF5 library and headers installed.

For help and/or questions, contact the `user mailing list <openradar-users@openradar.org>`_.


GNU Radio Configuration
=======================

If you plan on using Digital RF with GNU Radio, make sure to run the `pip` command in the same Python environment that your GNU Radio installation uses so that GNU Radio can find the packages. Depending on your GNU Radio installation, it may be necessary to add the Digital RF blocks to your GRC blocks path by creating or editing the GRC configuration file

:Unix (local): $HOME/.gnuradio/config.conf
:Windows (local): %APPDATA%/.gnuradio/config.conf
:Unix (global): /etc/gnuradio/conf.d/grc.conf
:Custom (global): {INSTALL_PREFIX}/etc/gnuradio/conf.d/grc.conf

to contain::

    [grc]
    local_blocks_path = {PIP_PREFIX}/share/gnuradio/grc/blocks

(replacing ``{PIP_PREFIX}`` with the pip installation prefix, "/usr/local" for example).


Example Usage
=============

The following code will load and read data located in a directory "/data/test".

Load the module and create a reader object::

    import digital_rf as drf
    do = drf.DigitalRFReader('/data/test')

List channels::

    do.get_channels()

Get data bounds for channel 'cha'::

    s, e = do.get_bounds('cha')

Read first 10 samples from channel 'cha'::

    data = do.read_vector(s, 10, 'cha')
