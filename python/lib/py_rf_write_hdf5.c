/*
 * Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
 * All rights reserved.
 *
 * Distributed under the terms of the BSD 3-clause license.
 *
 * The full license is in the LICENSE file, distributed with this software.
*/
/* The Python C extension for the rf_write_hdf5 C library
 *
 * $Id$
 *
 * This file exports the following methods to python
 * init
 * rf_write
 * free
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "digital_rf.h"
#include "hdf5.h"


// declarations
void init_py_rf_write_hdf5(void);
hid_t get_hdf5_data_type(char byteorder, char dtype_char, int bytecount);


static PyObject * _py_rf_write_hdf5_get_version(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_version returns the library version as a string */
{
	PyObject *retObj = Py_BuildValue("s", digital_rf_get_version());
	return(retObj);
}


void free_py_rf_write_hdf5(PyObject *capsule)
/* free_py_rf_write_hdf5 frees all C references
 *
 * Input: PyObject pointer to _py_rf_write_hdf5 PyCapsule object
 */
{
	Digital_rf_write_object * hdf5_write_data_object;

	/* get C pointer to Digital_rf_write_object */
	hdf5_write_data_object = (Digital_rf_write_object *)PyCapsule_GetPointer(capsule, NULL);

	digital_rf_close_write_hdf5(hdf5_write_data_object);

}


static PyObject * _py_rf_write_hdf5_init(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_init returns a pointer as a PyCObject to the Digital_rf_write_object struct created
 *
 * Inputs: python list with
 * 	1. directory - python string where Hdf5 files will be written
 * 	2. byteorder - python string of 'big' or 'little' describing the data format
 * 	3. dtype_char - python string with one character representing data type (i,u,b,B,f,d)
 * 	4. bytecount - python int giving the number of bytes in the data
 * 	5. subdir_cadence_secs - python int giving the number of seconds of data per subdirectory
 * 	6. file_cadence_millisecs - python int giving milliseconds of data per file
 * 	7. start_global_index - python int giving start time in samples since 1970 (unix_timestamp * sample rate)
 * 	8. sample_rate_numerator - python long giving the sample rate numerator
 * 	9. sample_rate_denominator - python long giving the sample rate denominator
 * 	10. uuid_str - python string representing the uuid
 * 	11. compression_level - python int representing the compression level (0-9)
 * 	12. checksum - 1 if checksums set, 0 if no checksum
 * 	13. is_complex - 1 if complex (I and Q) samples, 0 if single valued
 * 	14. num_subchannels - number of subchannels in data.  Must be at least 1 (int)
 * 	15. is_continuous - 1 is continuous data (allows faster write/read if no compression or checksum), 0 is with gaps
 * 	16. marching_periods - 1 for marching periods, 0 for none
 *
 *  Returns PyObject representing pointer to malloced struct if success, NULL pointer if not
 */
{
	// input arguments
	char * directory = NULL;
	char * byteorder = NULL;
	char * dtype_char = NULL;
	int bytecount = 0;
	uint64_t subdir_cadence_secs = 0;
	uint64_t file_cadence_millisecs = 0;
	uint64_t start_global_index = 0;
	uint64_t sample_rate_numerator = 0;
	uint64_t sample_rate_denominator = 0;
	char * uuid_str = NULL;
	int compression_level = 0;
	int checksum = 0;
	int is_complex = 0;
	int num_subchannels = 0;
	int is_continuous = 0;
	int marching_periods = 0;

	// local variables
	PyObject *retObj;
	hid_t hdf5_dtype;
	Digital_rf_write_object * hdf5_write_data_object;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "sssiKKKKKsiiiiii",
			  &directory,
			  &byteorder,
			  &dtype_char,
			  &bytecount,
			  &subdir_cadence_secs,
			  &file_cadence_millisecs,
			  &start_global_index,
			  &sample_rate_numerator,
			  &sample_rate_denominator,
			  &uuid_str,
			  &compression_level,
			  &checksum,
			  &is_complex,
			  &num_subchannels,
			  &is_continuous,
			  &marching_periods))
	{
		return NULL;
	}

	// find out what Hdf5 data type to use
	hdf5_dtype = get_hdf5_data_type(byteorder[0], dtype_char[0], bytecount);
	if (hdf5_dtype == -1)
	{
		fprintf(stderr, "Failed to find datatype for %c, %c, %i\n", byteorder[0], dtype_char[0], bytecount);
		PyErr_SetString(PyExc_RuntimeError, "Failed to find datatype for %c, %c, %i\n");
		return(NULL);
	}

	// create needed object (for now, we always get matching periods)
	hdf5_write_data_object = digital_rf_create_write_hdf5(directory, hdf5_dtype, subdir_cadence_secs, file_cadence_millisecs,
			                                              start_global_index, sample_rate_numerator, sample_rate_denominator,
														  uuid_str, compression_level, checksum, is_complex, num_subchannels,
			                                              is_continuous, marching_periods);

	if (!hdf5_write_data_object)
	{
		PyErr_SetString(PyExc_RuntimeError, "Failed to create Digital_rf_write_object\n");
		return(NULL);
	}

	// create python wrapper around a pointer to return
	retObj = PyCapsule_New((void *)hdf5_write_data_object, NULL, free_py_rf_write_hdf5);

    //return pointer;
    return(retObj);

}


static PyObject * _py_rf_write_hdf5_rf_write(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_rf_write writes a block of continous data to an Hdf5 channel
 *
 * Inputs: python list with
 * 	1. PyCObject containing pointer to data structure
 * 	2. numpy array of data to write
 * 	3. next_sample - long long containing where sample id to be written (globally)
 *
 * 	Returns next available global sample if success, 0 if not
 *
 */
{
	// input arguments
	PyObject * pyCObject;
	PyArrayObject * pyNumArr;
	uint64_t next_sample;

	// local variables
	Digital_rf_write_object * hdf5_write_data_object;
	void * data; /* will point to numpy array's data block */
	uint64_t vector_length; /* will be set to length of data */
	int result;
	PyObject *retObj;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOK",
			  &pyCObject,
			  &pyNumArr,
			  &next_sample))
	{
		return(NULL);
	}

	/* get C pointer to Digital_rf_write_object */
	hdf5_write_data_object = (Digital_rf_write_object *)PyCapsule_GetPointer(pyCObject, NULL);

	/* get C pointer to numpy array data */
	data = PyArray_DATA(pyNumArr);
	vector_length = (uint64_t)(PyArray_DIMS(pyNumArr)[0]);

	result = digital_rf_write_hdf5(hdf5_write_data_object, next_sample, data, vector_length);
	if (result)
	{
		PyErr_SetString(PyExc_RuntimeError, "Failed to write data\n");
		return(NULL);
	}

	/* success */
	retObj = Py_BuildValue("K", hdf5_write_data_object->global_index);
	return(retObj);

}


static PyObject * _py_rf_write_hdf5_rf_block_write(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_rf_block_write writes a block of data with gaps to an Hdf5 channel
 *
 * Inputs: python list with
 * 	1. PyCObject containing pointer to data structure
 * 	2. numpy array of data to write - must be 2-D with shape
 *  	(length, num_subchannels) and a dtype giving the size of the complete
 *  	sample, whether complex or real
 * 	3. numpy array of global sample count - must by array of type np.uint64
 * 	4. numpy array of block sample count - gives the position in each data arr of the
 * 		global sample given in the global sample count array above.  Len of this
 * 		array must be the same as the one before, and it must also be an array
 *  	of type np.uint64
 *
 * 	 Returns next available global sample if success, 0 if not
 */
{
	// input arguments
	PyObject * pyCObject;
	PyArrayObject * pyNumArr;
	PyArrayObject * pyGlobalArr;
	PyArrayObject * pyBlockArr;

	// local variables
	Digital_rf_write_object * hdf5_write_data_object;
	void * data; /* will point to numpy array's data block */
	uint64_t * global_arr; /* will point to numpy pyGlobalArr's data block */
	uint64_t * block_arr; /* will point to numpy pyGlobalArr's data block */
	uint64_t vector_length;
	uint64_t index_length;
	uint64_t i;
	uint64_t block_length;
	uint64_t block_index;
	uint64_t next_block_index;
	uint64_t next_sample;
	int result;
	PyObject *retObj;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOOO",
			  &pyCObject,
			  &pyNumArr,
			  &pyGlobalArr,
			  &pyBlockArr))
	{
		return(NULL);
	}

	/* get C pointer to Digital_rf_write_object */
	hdf5_write_data_object = (Digital_rf_write_object *)PyCapsule_GetPointer(pyCObject, NULL);

	/* get lengths */
	vector_length = (uint64_t)(PyArray_DIMS(pyNumArr)[0]);
	index_length = (uint64_t)(PyArray_DIMS(pyGlobalArr)[0]);

	if (hdf5_write_data_object->is_continuous && index_length > 1)
	{
		/* write each block in separate calls since digital_rf_write_blocks_hdf5
		 * requires a single continuous block per call with is_continuous */
		for (i=0; i<index_length; i++)
		{
			block_index = *((uint64_t *)PyArray_GETPTR1(pyBlockArr, i));
			if (i + 1 == index_length)
				next_block_index = vector_length;
			else
				next_block_index = *((uint64_t *)PyArray_GETPTR1(pyBlockArr, i+1));
			block_length = next_block_index - block_index;

			data = PyArray_GETPTR2(pyNumArr, block_index, 0);
			next_sample = *((uint64_t *)PyArray_GETPTR1(pyGlobalArr, i));

			result = digital_rf_write_hdf5(hdf5_write_data_object, next_sample, data, block_length);
			if (result)
			{
				PyErr_SetString(PyExc_RuntimeError, "Failed to write data\n");
				return(NULL);
			}
		}
	}
	else
	{
		/* get C pointers to numpy arrays */
		data = PyArray_DATA(pyNumArr);
		global_arr = PyArray_DATA(pyGlobalArr);
		block_arr = PyArray_DATA(pyBlockArr);

		result = digital_rf_write_blocks_hdf5(hdf5_write_data_object, global_arr, block_arr, index_length, data, vector_length);
		if (result)
		{
			PyErr_SetString(PyExc_RuntimeError, "Failed to write data\n");
			return(NULL);
		}
	}

	/* success */
	retObj = Py_BuildValue("K", hdf5_write_data_object->global_index);
	return(retObj);

}


static PyObject * _py_rf_write_hdf5_get_last_file_written(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_last_file_written returns last file written as string
 *
 * Inputs: python list with
 * 	1. PyCObject containing pointer to data structure
 *
 *  Returns python string of full path to last file written
 */
{
	// input arguments
	PyObject * pyCObject;

	// local variables
	Digital_rf_write_object * hdf5_write_data_object;
	PyObject *retObj;
	char * last_file_written;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "O",
			  &pyCObject))
	{
		return(NULL);
	}

	/* get C pointer to Digital_rf_write_object */
	hdf5_write_data_object = (Digital_rf_write_object *)PyCapsule_GetPointer(pyCObject, NULL);

	last_file_written = digital_rf_get_last_file_written(hdf5_write_data_object);

	/* success */
	retObj = Py_BuildValue("s", last_file_written);
	free(last_file_written);
	return(retObj);

}


static PyObject * _py_rf_write_hdf5_get_last_dir_written(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_last_dir_written returns last directory (full path to dir) written as string
 *
 * Inputs: python list with
 * 	1. PyCObject containing pointer to data structure
 *
 *  Returns python string of full path to last directory written
 */
{
	// input arguments
	PyObject * pyCObject;

	// local variables
	Digital_rf_write_object * hdf5_write_data_object;
	PyObject *retObj;
	char * last_dir_written;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "O",
			  &pyCObject))
	{
		return(NULL);
	}

	/* get C pointer to Digital_rf_write_object */
	hdf5_write_data_object = (Digital_rf_write_object *)PyCapsule_GetPointer(pyCObject, NULL);

	last_dir_written = digital_rf_get_last_dir_written(hdf5_write_data_object);

	/* success */
	retObj = Py_BuildValue("s", last_dir_written);
	free(last_dir_written);
	return(retObj);

}


static PyObject * _py_rf_write_hdf5_get_last_utc_timestamp(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_last_utc_timestamp returns utc timestamp at time of last write as uint_64
 *
 * Inputs: python list with
 * 	1. PyCObject containing pointer to data structure
 *
 *  Returns python int of utc timestamp at time of last write
 */
{
	// input arguments
	PyObject * pyCObject;

	// local variables
	Digital_rf_write_object * hdf5_write_data_object;
	PyObject *retObj;
	uint64_t last_timestamp;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "O",
			  &pyCObject))
	{
		return(NULL);
	}

	/* get C pointer to Digital_rf_write_object */
	hdf5_write_data_object = (Digital_rf_write_object *)PyCapsule_GetPointer(pyCObject, NULL);

	last_timestamp = digital_rf_get_last_write_time(hdf5_write_data_object);

	/* success */
	retObj = Py_BuildValue("K", last_timestamp);
	return(retObj);

}




/********** helper methods ******************************/
hid_t get_hdf5_data_type(char byteorder, char dtype_char, int bytecount)
/* get_hdf5_data_type returns an Hdf5 datatype that corresponds to the arguments
 *
 * Inputs:
 * 	char byteorder - char representing byteorder according to np.dtype
 * 		(< little-endian, > big-endian, | not applicable)
 * 	char dtype_char - i int, u - unsigned int, f - float
 *      (also accepts d for double for legacy, ignoring bytecount, assuming 8)
 * 	int bytecount - bytecount
 *
 * Returns hid_t HDF% datatype if recognized, -1 if not
 *
 */
{
	if (byteorder == '<')
	{
		if (dtype_char == 'f' && bytecount == 4)
			return(H5T_IEEE_F32LE);
		else if (dtype_char == 'f' && bytecount == 8)
			return(H5T_IEEE_F64LE);
		else if (dtype_char == 'd')
			return(H5T_IEEE_F64LE);
		else if (dtype_char == 'i' && bytecount == 1)
			return(H5T_STD_I8LE);
		else if (dtype_char == 'i' && bytecount == 2)
			return(H5T_STD_I16LE);
		else if (dtype_char == 'i' && bytecount == 4)
			return(H5T_STD_I32LE);
		else if (dtype_char == 'i' && bytecount == 8)
			return(H5T_STD_I64LE);
		else if (dtype_char == 'u' && bytecount == 1)
			return(H5T_STD_U8LE);
		else if (dtype_char == 'u' && bytecount == 2)
			return(H5T_STD_U16LE);
		else if (dtype_char == 'u' && bytecount == 4)
			return(H5T_STD_U32LE);
		else if (dtype_char == 'u' && bytecount == 8)
			return(H5T_STD_U64LE);
	}
	else if (byteorder == '>')
	{
		if (dtype_char == 'f' && bytecount == 4)
			return(H5T_IEEE_F32BE);
		else if (dtype_char == 'f' && bytecount == 8)
			return(H5T_IEEE_F64BE);
		else if (dtype_char == 'd')
			return(H5T_IEEE_F64BE);
		else if (dtype_char == 'i' && bytecount == 1)
			return(H5T_STD_I8BE);
		else if (dtype_char == 'i' && bytecount == 2)
			return(H5T_STD_I16BE);
		else if (dtype_char == 'i' && bytecount == 4)
			return(H5T_STD_I32BE);
		else if (dtype_char == 'i' && bytecount == 8)
			return(H5T_STD_I64BE);
		else if (dtype_char == 'u' && bytecount == 1)
			return(H5T_STD_U8BE);
		else if (dtype_char == 'u' && bytecount == 2)
			return(H5T_STD_U16BE);
		else if (dtype_char == 'u' && bytecount == 4)
			return(H5T_STD_U32BE);
		else if (dtype_char == 'u' && bytecount == 8)
			return(H5T_STD_U64BE);
	}
	else if (dtype_char == 'i')
		return(H5T_STD_I8LE);
	else if (dtype_char == 'u')
		return(H5T_STD_U8LE);
	// error if we got here
	return(-1);
}


static PyObject * _py_rf_write_hdf5_get_unix_time(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_unix_time returns a tuple of (year,month,day,hour,minute,second,picosecond)
 * given an input unix_sample_index and sample_rate
 *
 * Inputs: python list with
 * 	1. unix_sample_index - python int representing number of samples at given sample rate since UT midnight 1970-01-01
 * 	2. sample_rate_numerator - python int sample rate numerator in Hz
 * 	3. sample_rate_denominator - python int sample rate denominator in Hz
 *
 *  Returns tuple with (year,month,day,hour,minute,second,picosecond) if success, NULL pointer if not
 */
{
	// input arguments
	uint64_t unix_sample_index = 0;
	uint64_t sample_rate_numerator = 0;
	uint64_t sample_rate_denominator = 0;

	// local variables
	PyObject *retObj;
	int year, month, day;
	int hour, minute, second;
	uint64_t picosecond;
	int result;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "KKK",
			  &unix_sample_index,
			  &sample_rate_numerator,
			  &sample_rate_denominator))
	{
		return NULL;
	}

	// call underlying method
	result = digital_rf_get_unix_time_rational(
		unix_sample_index, sample_rate_numerator, sample_rate_denominator,
		&year, &month, &day, &hour, &minute, &second, &picosecond);
	if (result != 0)
		return(NULL);

	// create needed object
	retObj = Py_BuildValue("iiiiiiK",  year, month, day,
                           hour, minute, second, picosecond);

    //return tuple;
    return(retObj);

}


static PyObject * _py_rf_write_hdf5_get_timestamp_floor(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_timestamp_floor converts a sample index into the nearest
 *  earlier timestamp (flooring) divided into second and picosecond parts, using
 *  the sample rate expressed as a rational fraction.
 *
 *  Flooring is used so that sample falls in the window of time represented by
 *  the returned timestamp, which includes that time up until the next possible
 *  timestamp: second + [picosecond, picosecond + 1).
 *
 * Inputs: python list with
 * 	1. unix_sample_index - python int representing number of samples at given sample rate since UT midnight 1970-01-01
 * 	2. sample_rate_numerator - python int sample rate numerator in Hz
 * 	3. sample_rate_denominator - python int sample rate denominator in Hz
 *
 *  Returns tuple with (second,picosecond) if success, NULL pointer if not
 */
{
	// input arguments
	uint64_t unix_sample_index = 0;
	uint64_t sample_rate_numerator = 0;
	uint64_t sample_rate_denominator = 0;

	// local variables
	PyObject *retObj;
	uint64_t second;
	uint64_t picosecond;
	int result;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "KKK",
			  &unix_sample_index,
			  &sample_rate_numerator,
			  &sample_rate_denominator))
	{
		return NULL;
	}

	// call underlying method
	result = digital_rf_get_timestamp_floor(
		unix_sample_index, sample_rate_numerator, sample_rate_denominator,
		&second, &picosecond);
	if (result != 0)
		return(NULL);

	// create needed object
	retObj = Py_BuildValue("KK", second, picosecond);

    //return tuple;
    return(retObj);

}


static PyObject * _py_rf_write_hdf5_get_sample_ceil(PyObject * self, PyObject * args)
/* _py_rf_write_hdf5_get_sample_ceil converts a timestamp (divided into second
 *  and picosecond parts) into the next nearest sample (ceil), using the sample
 *  rate expressed as a rational fraction.
 *
 *  Ceiling is used to complement the flooring in get_timestamp_floor, so that
 *  get_sample_ceil(get_timestamp_floor(sample_index)) == sample_index.
 *
 * Inputs: python list with
 * 	1. second - python int giving the whole seconds part of the timestamp
 *  2. picosecond - python int giving the picoseconds part of the timestamp
 * 	2. sample_rate_numerator - python int sample rate numerator in Hz
 * 	3. sample_rate_denominator - python int sample rate denominator in Hz
 *
 *  Returns an integer sample index if success, NULL pointer if not
 */
{
	// input arguments
	uint64_t second = 0;
	uint64_t picosecond = 0;
	uint64_t sample_rate_numerator = 0;
	uint64_t sample_rate_denominator = 0;

	// local variables
	PyObject *retObj;
	uint64_t sample_index;
	int result;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "KKKK",
			  &second,
			  &picosecond,
			  &sample_rate_numerator,
			  &sample_rate_denominator))
	{
		return NULL;
	}

	// call underlying method
	result = digital_rf_get_sample_ceil(
		second, picosecond, sample_rate_numerator, sample_rate_denominator,
		&sample_index);
	if (result != 0)
		return(NULL);

	// create needed object
	retObj = Py_BuildValue("K", sample_index);
    return(retObj);

}



/********** Initialization code for module ******************************/

static PyMethodDef _py_rf_write_hdf5Methods[] =
{
	  {"init",           	           _py_rf_write_hdf5_init,          		METH_VARARGS},
	  {"rf_write",           	       _py_rf_write_hdf5_rf_write,          	METH_VARARGS},
	  {"rf_block_write",           	   _py_rf_write_hdf5_rf_block_write,    	METH_VARARGS},
	  {"get_last_file_written",        _py_rf_write_hdf5_get_last_file_written, METH_VARARGS},
	  {"get_last_dir_written",         _py_rf_write_hdf5_get_last_dir_written,  METH_VARARGS},
	  {"get_last_utc_timestamp",       _py_rf_write_hdf5_get_last_utc_timestamp,METH_VARARGS},
	  {"get_unix_time",           	   _py_rf_write_hdf5_get_unix_time,     	METH_VARARGS},
	  {"get_timestamp_floor",          _py_rf_write_hdf5_get_timestamp_floor,     	METH_VARARGS},
	  {"get_sample_ceil",         	   _py_rf_write_hdf5_get_sample_ceil,     	METH_VARARGS},
	  {"get_version",                  _py_rf_write_hdf5_get_version,           METH_NOARGS},
      {NULL,      NULL}        /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
	#define MOD_ERROR_VAL NULL
	#define MOD_SUCCESS_VAL(val) val
	#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
	#define MOD_DEF(ob, name, doc, methods) \
		static struct PyModuleDef moduledef = { \
			PyModuleDef_HEAD_INIT, \
			name,     /* m_name */ \
			doc,      /* m_doc */ \
			-1,       /* m_size */ \
			methods,  /* m_methods */ \
			NULL,     /* m_reload */ \
			NULL,     /* m_traverse */ \
			NULL,     /* m_clear */ \
			NULL,     /* m_free */ \
		}; \
		ob = PyModule_Create(&moduledef);
#else
	#define MOD_ERROR_VAL
	#define MOD_SUCCESS_VAL(val)
	#define MOD_INIT(name) void init##name(void)
	#define MOD_DEF(ob, name, doc, methods) \
		ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_py_rf_write_hdf5)
{
	PyObject *m;

	MOD_DEF(
		m,  /* module object */
		"_py_rf_write_hdf5",  /* module name */
		"Python extension for the Digital RF rf_write_hdf5 C library",  /* module doc */
		_py_rf_write_hdf5Methods  /* module methods */
	)

	if (m == NULL)
		return MOD_ERROR_VAL;

	// needed to initialize numpy C api and not have segfaults
	import_array();

	return MOD_SUCCESS_VAL(m);
}
