**Added:**

* Building of Python package wheels using `cibuildwheel` has been added as a GitHub Actions workflow, making use of the change in build system (see below).

**Changed:**

* The build system has been updated to use pyproject.toml + scikit-build-core for the Python package, replacing the legacy setup.py + setuptools system of yore. As a result, the CMake build procedure has changed a bit, notably in that building a Python wheel using CMake is no longer supported. Use `python -m build` or an equivalent frontend from the source root to build Python wheels instead. The Python package can still be built and installed as before using CMake, just not as a wheel.

**Deprecated:**

* <news item>

**Removed:**

* Python 2 and Python < 3.8 will no longer be supported, and the README and build system has been updated accordingly.
* The dependency on `pytz` has been removed in favor of equivalent functionality now in the Python standard library.

**Fixed:**

* Replace deprecated use of Python datetime's `utcnow()` and `utcfromtimestamp()`.

**Security:**

* <news item>
