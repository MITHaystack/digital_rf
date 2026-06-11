**Added:**

* <news item>

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* DigitalRFReader.read_vector_raw would fail for a single sample because np.squeeze would make the result a scalar and not an array. This fixes that by using np.atleast_1d to ensure an array is returned.

**Security:**

* <news item>
