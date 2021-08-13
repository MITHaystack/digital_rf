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

**Security:**

* <news item>
