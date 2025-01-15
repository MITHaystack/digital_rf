=====================
digital_rf Change Log
=====================

.. current developments

v2.6.10
====================

**Added:**

* Building of Python package wheels using `cibuildwheel` has been added as a GitHub Actions workflow, making use of the change in build system (see below).

**Changed:**

* The build system has been updated to use pyproject.toml + scikit-build-core for the Python package, replacing the legacy setup.py + setuptools system of yore. As a result, the CMake build procedure has changed a bit, notably in that building a Python wheel using CMake is no longer supported. Use `python -m build` or an equivalent frontend from the source root to build Python wheels instead. The Python package can still be built and installed as before using CMake, just not as a wheel.
* When reading Digital Metadata, convert numpy object arrays to a list for better interoperability (particularly with GNU Radio's PMT type). Also ensure that arrays of strings have the str type.

**Removed:**

* Python 2 and Python < 3.8 will no longer be supported, and the README and build system has been updated accordingly.
* The dependency on `pytz` has been removed in favor of equivalent functionality now in the Python standard library.

**Fixed:**

* Replace deprecated use of Python datetime's `utcnow()` and `utcfromtimestamp()`.
* Fix CMake so that digital_rf.dll actually gets installed on Windows! This makes the C library usable independent of the Python bindings.
* Fixed `TypeError: unsupported operand type(s) for /: 'int' and 'numpy.longdouble'` issue with gr_digital_rf.digital_rf_channel_source by converting the `int` to a `numpy.uint64` before division.
* Fixed aliased numpy function names for numpy 2.0 compatibility.
* Updated to numpy 2.0's new definition for the copy keyword for np.array() by switching to np.asarray() [as suggested](https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword).

**Authors:**

* Ryan Volz



v2.6.9
====================

**Added:**

* Add "link" method to mirror module and "ln" to `drf mirror`, to hard link files instead of copying. When hard linking is not possible (e.g. across different partitions), it will fall back to copying.
* Add `drf ln` command to link files (hard or symbolic).
* Added option `rdcc_nbytes` to `DigitalRFReader` to allow specification of HDF5 chunk cache size (see HDF5 documentation for details). This also increases the default chunk cache size from 1 MB to 4 MB to speed up reading of compressed or checksummed data in a typical use case.
* The sounder/tx.py example script has been updated to accept waveform files in complex int16 format through use of the new `--type int16` argument.

**Fixed:**

* Fixed thorpluto.py, changed the iio.pluto_source to iio.fmcomms2_source_fc32 as the pluto_source was removed in newer versions of iio.

**Authors:**

* Ryan Volz
* John Swoboda
* Juha Vierinen



v2.6.8
====================

**Changed:**

* drf_sti: Updated to have better arguments (consistent with more recent tools), handle data gaps, and add simple channel sum beamforming.
* thor: Swap order of setting USRP clock and time sources, time first. This should reduce the number of re-syncs necessary with modern USRPs (N3xx, X4xx) in the absence of being able to do a set_sync_source call.
* thor: Put USRP clock, time, and lo arguments into device string, and do not set those arguments after device initialization if they do not change. This means that thor will do less re-initialization of the device settings during startup.

**Fixed:**

* Fixed Python DigitalRFReader and DigitalMetadataReader for compatibility with numpy 1.23 on Windows (and possibly other platforms with np.longdouble==np.double).

**Authors:**

* Ryan Volz
* Frank Lind



v2.6.7
====================

**Added:**

* Added the `DigitalRFReader.read_vector_1d` method for reading data and always returning a 1-D array of the smallest safe floating point type, replacing `DigitalRFReader.read_vector_c81d`.
* Basic logging support has been added, with the case of failing to import the `watchdog_drf` module being the only instance of logged information so far. The logging level can be set using either the `DRF_LOGLEVEL` or `LOGLEVEL` environment variables. The default level is `WARNING`, and the `watchdog_drf` import error is logged at the `INFO` level.

**Changed:**

* Renamed the GNU Radio companion block tree title from "gr_digital_rf" to "Digital RF" to better match the style of other out-of-tree modules.
* `DigitalRFReader.read_vector` no longer always returns an array with a `np.complex64` dtype. Instead, the array will always have be of the smallest floating point type (either complex or real) that will safely fit hold the underlying data without loss of precision. We recommend manually changing to a smaller type if a loss of precision is acceptable. The benefit over this function over `DigitalRFReader.read_vector_raw` is that you don't have to worry about handling complex integer data with a compound dtype.
* The Python package now depends on `oldest-supported-numpy` instead of just `numpy`, so that source builds can maintain maximum compatibility with different `numpy` versions.

**Deprecated:**

* The `DigitalRFReader.read_vector_c81d` method is deprecated and will be removed in digital_rf version 3. Use read_vector_1d instead and append `.astype('c8', casting='unsafe', copy=False)` if a strict return dtype of complex64 is desired.

**Fixed:**

* Fixed #25 (digital_rf_sink: version check on GNU Radio causes TypeError) by removing the GNU Radio version check since it wasn't actually doing anything helpful anymore.
* Fix thor.py failures when recording multiple channels (e.g. `AttributeError: 'list_iterator' object has no attribute 'start'`). Some flowgraph blocks were being garbage-collected before/during execution because no references were stored to the Python objects with GNU Radio 3.9+. Now thor.py keeps these references itself.
* Fix thor.py error when setting a stop time with GNU Radio 3.9+.
* Improve thor.py start time tagging with at least the B2xx radios.
* Improve thor.py reliability with stop times by not attempting to stop at an exact time, but instead just stop when we are sure we are past the stopping time.
* Fix stream tag handling in Digital RF Sink and Raster blocks. The `get_tags_in_window` function is broken in GNU Radio 3.9.2.0, so use `get_tags_in_range` instead.
* The `watchdog_drf` module is now compatible with recent versions of the `watchdog` package, from version 1 up through at least version 2.1.2.

**Authors:**

* Ryan Volz



v2.6.6
====================

**Deprecated:**

* The `digital_rf_get_unix_time` function is now deprecated, as it relies on a `long double` sample rate. Use `digital_rf_get_unix_time_rational` instead.

**Fixed:**

* Fix incorrect file bound calculation in `digital_rf_get_subdir_file` on platforms that have a `long double` that is different from amd64, notably at least the aarch64 ARM platform. This fixes a bug where writes failed with error messages "Failed to write data" and "Request index M before first expected index N".
* Regularized use of 64 bit integer types and their conversion to Python values, perhaps correcting behavior when compiled on 32-bit architectures.
* Cleaned up compiler warnings about comparing signed and unsigned values.
* Cleaned up testing warnings about invalid values in equals comparison.

**Authors:**

* Ryan Volz



v2.6.5
====================

**Added:**

* Added start sample to debug printing of 'digital_rf_channel_sink' to complement the debug printing of rx_time tags.

**Changed:**

* The Digital RF (Channel) Source/Sink blocks for gnuradio-companion have been modified to accept 'raw' input for the start and end identifiers instead of strings, allowing variables to be used. Existing flowgraphs may require quotes to be placed around existing string input.

**Fixed:**

* The drf_watchdog module is now compatible with watchdog 0.10+. There may be a slight change of behavior (duplicate or out of order events) but the mirror and ringbuffer utilities can handle it gracefully.
* Better error message when no samples are specified with drf_plot.py.
* Fix the Digital RF sink blocks and GRC yaml to prevent an empty array for center_frequencies being written as Digital Metadata (currently happens with default GRC block with GNU Radio 3.8).
* Clarified docstrings (and updated to actual modern behavior) for 'start' and 'end' in Digital RF source/sinks.

**Authors:**

* Ryan Volz



v2.6.4.4
====================

**Fixed:**

* Fix to drf_plot.py to ignore negative infinity values when autoscaling.
* Fix thorpluto.py for better compatibility when both the libiio python bindings (iio.py module) and the gr-iio package (either gnuradio.iio or just iio) are installed.

**Authors:**

* Ryan Volz



v2.6.4.3
====================

**Fixed:**

* Fix matched filtering in drf_plot.py to run with Python 3 and use the correct code (not reversed). Also shift the filtered result so that ranges are the same before and after filtering.
* Fix RTI and STI plots in drf_plot.py tool for Python 3. Once again the assumption of an integer result from division rears its ugly head.



v2.6.4.2
====================

**Fixed:**

* Fix an error seen when deleting the Digital RF Reader object (such as on interpreter shutdown) caused by trying to close the cached HDF5 file handle.
* Fix another Python 3 issue with the plotting tools caused by getting a float from division when an integer is required.
* Fix automatic plot scaling in the plotting tools to handle data with NaNs.

**Authors:**

* Ryan Volz



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
