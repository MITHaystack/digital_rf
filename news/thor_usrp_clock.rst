**Added:**

* <news item>

**Changed:**

* thor: Swap order of setting USRP clock and time sources, time first. This should reduce the number of re-syncs necessary with modern USRPs (N3xx, X4xx) in the absence of being able to do a set_sync_source call.
* thor: Put USRP clock, time, and lo arguments into device string, and do not set those arguments after device initialization if they do not change. This means that thor will do less re-initialization of the device settings during startup.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
