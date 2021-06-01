**Added:**

* Basic logging support has been added, with the case of failing to import the `watchdog_drf` module being the only instance of logged information so far. The logging level can be set using either the `DRF_LOGLEVEL` or `LOGLEVEL` environment variables. The default level is `WARNING`, and the `watchdog_drf` import error is logged at the `INFO` level.

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* The `watchdog_drf` module is now compatible with recent versions of the `watchdog` package, from version 1 up through at least version 2.1.2.

**Security:**

* <news item>
