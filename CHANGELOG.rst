=====================
digital_rf Change Log
=====================

.. current developments

v2.6.4.1
====================

**Fixed:**

* Fix drf_plot.py, drf_sti.py, drf_cross_sti.py, and drf_sound.py to be compatible with Python 3 by removing use of string module functions and listing dict keys objects.
* Fix the plotting tools to be compatible with Matplotlib 3 by removing use of hold functions on axes instances.
* The thorpluto.py script has been fixed to work with both the released gr-iio version (compatible with GNU Radio 3.7) and the unreleased gr-iio version that is compatible with GNU Radio 3.8.
* Fix an error with thorpluto.py when no mainboard is specified (it should have used the first available.)

**Authors:**

* Ryan Volz



v2.6.4
====================

**Added:**

* Add the "thorpluto" tool for writing data from the ADALM PLUTO using GNU Radio. This script requires gr-iio to run. Note that this script duplicates some of the functionality of the existing "thor" recorder script, and may be subsumed or arguments may change in a future consolidation.
* Option to use the CMake variable ``Python_EXECUTABLE`` to manually set the path to the Python interpreter (useful if autodetection fails or Python 2 is preferred).
* Add ``stop_on_time_tag`` parameter to the Digital RF Sink blocks, useful when time tags only happen for USRP dropped samples but the 'rx_time' tag value falsely indicates no drop.
* Add the "thorosmo" tool for writing data from osmosdr-supported receivers using GNU Radio, and add the "uhdtodrf" tool for writing data from UHD devices using the UHD Python API *without* using GNU Radio. Note that these scripts duplicate some of the functionality of the existing "thor" recorder script, and they may be subsumed or their arguments may change in a future consolidation.
* Add option to force polling for watchdog functions (ringbuffer, mirror, etc.), which is useful as a fallback when the default watchdog observer fails silently.

**Changed:**

* The ``thor.py`` script's ``stop_on_dropped`` parameter now includes the new ``stop_on_time_tag`` behavior.

**Fixed:**

* Fix an IndexError when using `stop_on_skipped` or `stop_on_time_tag` with `gr_digital_rf.digital_rf_channel_sink`. If the skip/tag happened with only one data block to be written, the IndexError would trigger upon trying to index to a second data block.




v2.6.3
====================

- Format Python code and enforce a standard style using Black.
- Include a small amount of example data to demonstrate the format and enable example scripts that don't depend on writing data first.
- Add yaml-based GRC files and fix a gr_digital_rf Python 3 bug for GNU Radio 3.8+ compatibility.
- Fix the MATLAB reader when dealing with very high sample rates.
- Fix resampling and channelizer filters in thor.py to correctly account for filter delays and keep the timing consistent.
- Clean up and fix various minor issues throughout the Python codebase.



v2.6.2
====================

This is entirely a bugfix release. Notable fixes include:

- Fix error using Digital RF Source with GNU Radio 3.7.12, which changed the type of its input and output signature objects.
- Digital RF Source now outputs zeros for missing values, to minimize impact on downstream processing.
- Make tests compatible with pytest >=4.
- The watchdog_drf module now works for non-inotify observers (i.e. non-Linux).
- Proper cleanup of tmp.rf@*.h5 files when thor and mirror/ringbuffer tools are killed.



v2.6.1
====================

This is primarily a bugfix release. Changes include:

- Add raster and vector tools to gr_digital_rf for working with periodic data.
- Disable file locking with HDF5 >= 1.10 for digital_metadata, which restores behavior so it matches that of HDF5 1.8.
- Fix error using digital_rf_sink with GNU Radio 3.7.12, which changed the type of its input and output signature objects.
- Fix the digital_rf_sink block in GRC to correctly pass an empty center frequency.



v2.6.0
====================

The main impetus for this release is a complete reorganization of the build system to enable Digital RF to run with Python 3 and on Windows. Major changes include:

- Python 3.5+ compatibility (excluding 'gr_digital_rf' since GNU Radio does not support Python 3 yet).
- Windows compatibility, including conda packages.
- Automated CI testing with revamped Python tests using 'pytest'.
- Python package available on PyPI (for 'pip' install), including binary wheels for Linux, OSX, and Windows.
- Python 'gr_digital_rf' packaged with 'digital_rf' since we no longer require GNU Radio to build (dropping Digital RF C Sink support).
- New 'thor' options:
  - Output channel settings including rational resampling, frequency shifting, and channelizing
  - Clock source (10 MHz ref) and time source (PPS) split out from sync_source
  - Clock lock check with nolock option to skip
  - LO source/export
  - Tuning arguments
  - DC offset and IQ balance
- Updated sounder example transmit script with some 'thor' features.
- Matlab reader fixes for recent data format and packaging as a Matlab Toolbox.
- Various bug fixes and improvements (see commit log for full list of changes).



v2.5.4
====================

This release incorporates many robustness improvements and fixes based on testing with the Millstone Hill radar. Major changes include:

- Ringbuffer/mirror/watchdog code are now more efficient and robust to errors.
- New 'drf cp' and 'drf mv' commands for copying and moving data.
- The watchdog and cp/mv commands now support specifying a start and end time to watch/copy/move only a particular window of time.
- Many fixes and updates to the beacon example.
- Added ability to specify input/output chunksize in the GNU Radio Digital RF Sources/Sinks in order to tweak performance for a particular application.



v2.5.3
====================

Improvements to watchdog_drf, list_drf, mirror, and ringbuffer. Can now monitor directories that don't yet exist or get deleted and ringbuffer by file count and duration.



v2.5.2
====================

Fix build on OSX for C version of gr_drf Digital RF Sink.



v2.5.1
====================

The main new feature is a GNU Radio Digital RF Sink written entirely in Python that writes receiver and recorder metadata previously handled only in the thor recording script.



v2.5
====================

First release intended for public use. The 'metadata.h5' files that previously indicated a Digital RF/Metadata channel directory and that stored properties inherent to the channel have been renamed to 'drf_properties.h5' and 'dmd_properties.h5', respectively, to avoid confusion with accompanying Digital Metadata.



v2.4
====================

First release with a revamped CMake build system and including the gr_drf GNU Radio module and many examples.



v2.0 - Dec 30, 2015
====================

Major update to Digital RF, in that file and subdirectory names were made predictable. To do this, each file and subdirectory now contains a set range of samples, and files and subdirectories will no longer have set number of samples when data is gappy. This greatly simplified the read api, since globs were no longer needed to find the data files that need to be opened; instead all needed file names can be derived.



v1.1.1 - Aug 4, 2014
====================

The python read methods have changed. The method read_vector now returns all data in format numpy.complex8, no matter how the data was stored in the underlying Hdf5 file. A new method, read_vector_raw duplicates the old read_vector method, returning data in the format stored in the Hdf5 raw files. The method read_vector_c81d that returns data as a single subchannel in numpy.complex8 format still exists, but issues a UserWarning recommending use of the other methods.



v1.1 - July 7, 2014
====================

The directory naming convention has changed from HH:MM:SS since certain file systems disallowed colons in directory names.  This affected both the read and write API's.



v1.0 - May 29, 2014
====================

The first major release of the C and Python API's supporting the Digital RF HDF5 raw data format.
