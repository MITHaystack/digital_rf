**Added:**

* The Digital RF and Digital Metadata reader objects now provide a ``sample_rate`` property that represents the rational sample rate as a ``fractions.Fraction`` object. The ``samples_per_second`` property of ```np.longdouble``` dtype still exists for backwards compatibility, but new code should use ``sample_rate`` instead.
* Add new ``digital_rf_get_timestamp_floor`` and ``digital_rf_get_sample_ceil`` C functions that can be used to convert between a timestamp and sample index, given the rational sample rate. These have been made available so that users can perform these calculations in a way that is consistent with what is done internally.
* Add new ``datetime_to_timedelta_tuple``, ``get_samplerate_frac``, ``sample_to_time_floor``, ``time_to_sample_ceil`` Python utility functions for datetime, timestamp, and sample index math. These match the new C functions.

**Changed:**

* All internal code has been updated so that sample rate calculations use a rational representation instead of a ``np.longdouble`` floating point.

**Deprecated:**

* Deprecate ``samples_to_timedelta`` utility function. Use ``sample_to_time_floor`` instead and create a timedelta object if necessary: ``datetime.timedelta(seconds=seconds, microseconds=picoseconds // 1000000)``.
* Deprecate ``time_to_sample`` utility function. Use ``time_to_sample_ceil`` instead in combination with ``datetime_to_timedelta_tuple`` if necessary.

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
