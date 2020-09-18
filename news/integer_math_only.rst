**Added:**

* <news item>

**Changed:**

* <news item>

**Deprecated:**

* The `digital_rf_get_unix_time` function is now deprecated, as it relies on a `long double` sample rate. Use `digital_rf_get_unix_time_rational` instead.

**Removed:**

* <news item>

**Fixed:**

* Fix incorrect file bound calculation in `digital_rf_get_subdir_file` on platforms that have a `long double` that is different from amd64, notably at least the aarch64 ARM platform. This fixes a bug where writes failed with error messages "Failed to write data" and "Request index M before first expected index N".

**Security:**

* <news item>
