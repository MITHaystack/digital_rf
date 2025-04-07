**Added:**

* <news item>

**Changed:**

* Enforce keyword-only arguments for Python code adjacent to Digital RF's use of the watchdog library to have consistency with watchdog and try to prevent future issues.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Fix unhandled exception with watch/mirror/ringbuffer observing a non-existent directory with watchdog >= 5.0.0. Starting with watchdog 5.0.0, the ObservedWatch object requires that the `recursive` init argument be provided as a keyword argument.

**Security:**

* <news item>
