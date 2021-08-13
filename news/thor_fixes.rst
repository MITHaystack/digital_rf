**Added:**

* <news item>

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Fix thor.py failures when recording multiple channels (e.g. `AttributeError: 'list_iterator' object has no attribute 'start'`). Some flowgraph blocks were being garbage-collected before/during execution because no references were stored to the Python objects with GNU Radio 3.9+. Now thor.py keeps these references itself.
* Fix thor.py error when setting a stop time with GNU Radio 3.9+.
* Improve thor.py start time tagging with at least the B2xx radios.
* Improve thor.py reliability with stop times by not attempting to stop at an exact time, but instead just stop when we are sure we are past the stopping time.

**Security:**

* <news item>
