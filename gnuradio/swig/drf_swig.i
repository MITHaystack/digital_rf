/* -*- c++ -*- */

#define GRDRF_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "drf_swig_doc.i"

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "gr_drf/digital_rf_sink.h"
%}

// needed to initialize numpy C api and not have segfaults
%init %{
    import_array();
%}

%typemap(in) long double {
    PyObject* obj;
    PyArrayObject* arr;
    npy_longdouble val;
    PyArray_Descr* longdoubleDescr = PyArray_DescrFromType(NPY_LONGDOUBLE);

    if (PyArray_CheckScalar($input)) {
        if (PyArray_IsZeroDim($input)) {
            obj = PyArray_Return((PyArrayObject*)$input);
        } else {
            obj = $input;
        }
        // only exact conversion from longdouble or integer type are allowed
        if (PyArray_EquivTypes(PyArray_DescrFromScalar(obj), longdoubleDescr) ||
                PyDataType_ISINTEGER(PyArray_DescrFromScalar(obj))) {
            PyArray_CastScalarToCtype(obj, &val, longdoubleDescr);
            $1 = (long double) val;
        }
        else {
            SWIG_exception(SWIG_TypeError, "expected numpy.longdouble, string,"
                " or integer for conversion to long double");
        }
    } else if (PyString_Check($input) || PyUnicode_Check($input)) {
        // create numpy longdouble scalar from string and cast to long double
        arr = (PyArrayObject *)PyArray_FromAny($input, longdoubleDescr, 0, 0,
                                               NPY_ARRAY_FORCECAST, NULL);
        if (!arr) {
            PyObject *exc, *val, *tb;
            PyErr_Fetch(&exc, &val, &tb);
            SWIG_exception(SWIG_ValueError, PyString_AsString(val));
        }
        obj = PyArray_ToScalar(PyArray_DATA(arr), arr);
        PyArray_CastScalarToCtype(obj, &val, longdoubleDescr);
        $1 = (long double) val;
    } else if (PyInt_Check($input)) {
        $1 = (long double) PyInt_AsLong($input);
    } else if (PyLong_Check($input)) {
        $1 = (long double) PyLong_AsLong($input);
    } else {
        SWIG_exception(SWIG_TypeError, "expected numpy.longdouble, string,"
            " or integer for conversion to long double");
    }
}

%apply unsigned long long *INPUT { uint64_t *start_sample_index }


%include "gr_drf/digital_rf_sink.h"
GR_SWIG_BLOCK_MAGIC2(drf, digital_rf_sink);
