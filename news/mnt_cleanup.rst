**Added:**

* Added the `DigitalRFReader.read_vector_1d` method for reading data and always returning a 1-D array of the smallest safe floating point type, replacing `DigitalRFReader.read_vector_c81d`.

**Changed:**

* `DigitalRFReader.read_vector` no longer always returns an array with a `np.complex64` dtype. Instead, the array will always have be of the smallest floating point type (either complex or real) that will safely fit hold the underlying data without loss of precision. We recommend manually changing to a smaller type if a loss of precision is acceptable. The benefit over this function over `DigitalRFReader.read_vector_raw` is that you don't have to worry about handling complex integer data with a compound dtype.
* The Python package now depends on `oldest-supported-numpy` instead of just `numpy`, so that source builds can maintain maximum compatibility with different `numpy` versions.

**Deprecated:**

* The `DigitalRFReader.read_vector_c81d` method is deprecated and will be removed in digital_rf version 3. Use read_vector_1d instead and append `.astype('c8', casting='unsafe', copy=False)` if a strict return dtype of complex64 is desired.

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
