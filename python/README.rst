The Digital RF project encompasses a standardized HDF5 format for reading and writing of radio frequency data and the software for doing so. The format is designed to be self-documenting for data archive and to allow rapid random access for data processing.

This package includes:

* ``digital_rf`` Python package
* Tools for managing and processing Digital RF data
* ``gr_digital_rf`` Python package for interfacing with GNU Radio
* GNU Radio Companion blocks
* `thor` UHD radio recorder script
* Example scripts and applications

Digital RF C and MATLAB libraries can be found at the `official source code repository <https://github.com/MITHaystack/digital_rf>`_. To build from source, you must have the HDF5 library and headers installed.

For help and/or questions, contact the `user mailing list <openradar-users@openradar.org>`_.


Example Usage
=============

The following code will load and read data located in a directory '/data/test'.

Load the module and create a reader object::

    import digital_rf as drf
    do = drf.DigitalRFReader('/data/test')

List channels::

    do.get_channels()

Get data bounds for channel 'cha'::

    s, e = do.get_bounds('cha')

Read first 10 samples from channel 'cha'::

    data = do.read_vector(s, 10, 'cha')
