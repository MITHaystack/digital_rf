/*
 * Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
 * All rights reserved.
 *
 * Distributed under the terms of the BSD 3-clause license.
 *
 * The full license is in the LICENSE file, distributed with this software.
*/
/*******************************************************
 * digital_rf.h is the header file for the rf_write_hdf5 C library
 *
 * Version 2.0 - modified so that all subdirectory and file names are predictable
 *
 * The rf_write_hdf5 library supports the writing of rf data
 * into the Hdf5 file as specified in the accompanying documentation.  This
 * format is design to support RF archiving with excellent random access capability.
 *
 * Note that there are four different indices used in this API, and the names of these indices
 * are named with the prefixes:
 * 	global_ - this is the overall index, where 0 represents the first sample recorded, and
 * 		the index always refers to the number of sample periods since that first sample. Note now that
 * 		what is stored in the Hdf5 file is global_index + index of first sample, which is the number of samples
 * 		between midnight UT 1970-01-01 and the start of the experiment at the given sample_rate.  This mean this
 * 		index is an absolute UTC time (leap seconds are ignored, unless they occur during data taking).
 * 	data_ - this is the index into the block of data as passed in by the user, and zero is always
 * 		the first sample in that data array passed in.
 * 	dataset_ - this index always refers to a position in the Hdf5 rf_data dataset in a particular Hdf5 file
 * 	block_ - this index always refers to a position in the Hdf5 rf_data_index dataset that stores indices
 * 		into /rf_data
 *
 * 	Written by Bill Rideout (brideout@haystack.mit.edu), in collaboration with Juha Vierinan (x@haystack.mit.edu)
 * 	and Frank Lind (flind@haystack.mit.edu)
 *
 * $Id$
 */

#ifndef _RF_WRITE_HDF5_
#define _RF_WRITE_HDF5_

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <inttypes.h>

#include "hdf5.h"
#include "H5Tpublic.h"

#include "digital_rf_version.h"

#ifdef _WIN32
#  ifdef digital_rf_STATIC_DEFINE
#    define EXPORT
#  else
#    ifdef digital_rf_EXPORTS
#      define EXPORT __declspec(dllexport)
#    else
#      define EXPORT __declspec(dllimport)
#    endif
#  endif
#else
#  define EXPORT
#endif

/* string sizes */
#define SMALL_HDF5_STR 265
#define MED_HDF5_STR 512
#define BIG_HDF5_STR 1024

/* chunk size for rf_data_index */
#define CHUNK_SIZE_RF_DATA_INDEX 100

#define DIGITAL_RF_EPOCH "1970-01-01T00:00:00Z"
#define DIGITAL_RF_TIME_DESCRIPTION "All times in this format are in number of samples since the epoch in the epoch attribute.  The first sample time will be sample_rate * UTC time at first sample.  Attribute init_utc_timestamp records this init UTC time so that a conversion to any other time is possible given the number of leapseconds difference at init_utc_timestamp.  Leapseconds that occur during data recording are included in the data."


typedef struct digital_rf_write_object {

    /* this structure encapsulates all information needed to write to a series of Hdf5 files in a directory */
	char *     directory;       		/* Channel directory name where all data is stored - will always end with a "/" */
	char *     sub_directory;           /* Present sub-directory in form YYYY-MM-DDTHH:MM:SS - will always end with a "/" */
	char       basename[SMALL_HDF5_STR];/* Basename of last hdf5 file written to */
	int        is_complex;              /* 1 if complex (IQ) data, 0 if single valued */
	int        num_subchannels;         /* number of subchannels in the data stream.  Must be at least 1. */
	int        rank;            		/* 2 if complex (IQ) data or num_subchannels > 1, 1 otherwise */
	char *     uuid_str;        		/* UUID in str form */
	uint64_t   subdir_cadence_secs;		/* Number of seconds of data found in one subdir. */
	uint64_t   file_cadence_millisecs; 	/* number of milliseconds of data per file. Rule:  subdir_cadence_secs*1000 % file_cadence_millisecs == 0 */
	uint64_t   global_start_sample;     /* time of first sample in number of samples since UT midnight 1970-01-01 */
	uint64_t   sample_rate_numerator;   /* sample rate numerator. Final sample rate is sample_rate_numerator/sample_rate_denominator in Hz */
	uint64_t   sample_rate_denominator; /* sample rate denominator. Final sample rate is sample_rate_numerator/sample_rate_denominator in Hz */
	long double sample_rate;            /* calculated sample_rate set to sample_rate_numerator/sample_rate_denominator */
	uint64_t   max_chunk_size;          /* smallest possible value for maximum number of samples in a file = floor((file_cadence_millisecs/1000)*sample_rate) */
	int        is_continuous;           /* 1 if continuous data being written, 0 if there might be gaps */
	int        needs_chunking;  		/* 1 if /rf_data needs chunking (either not is_continuous or compression or checksums used) */
	hsize_t    chunk_size;      		/* With Digital RF 2.0 hard coded to CHUNK_SIZE_RF_DATA */
	hid_t      dtype_id;        		/* individual field data type as defined by hdf5.h */
	hid_t      complex_dtype_id;        /* complex compound data type if is_complex, with fields r and i */
	uint64_t   global_index;    		/* index into the next sample that could be written (global) */
	int        present_seq;     		/* The present Hdf5 file sequence. Init value is -1 */
	uint64_t   dataset_index;   		/* the next available index in open Hdf5 to write to */
	uint64_t   dataset_avail;   		/* the number of samples in the dataset available for writing to */
	uint64_t   block_index;     		/* the next available row in the open Hdf5 file/rf_data_index dataset to write to */
	hid_t      dataset;         		/* Dataset presently opened            */
	hid_t      dataspace;       		/* Dataspace used (rf_data)            */
	hid_t      filespace;       		/* filespace object used               */
	hid_t      memspace;        		/* memspace object used                */
	hid_t      hdf5_file;       		/* Hdf5 file presently opened          */
	hid_t      dataset_prop;    		/* Hdf5 dataset property               */
	hid_t      index_dataset;   		/* Hdf5 rf_data_index dataset          */
	hid_t      index_prop;      		/* Hdf5 rf_data_index property         */
	int        next_index_avail;		/* the next available row in /rf_data_index */
	int        marching_dots;           /* non-zero if marching dots desired when writing, 0 if not */
	uint64_t   init_utc_timestamp;      /* unix time when channel init called - stored as attribute in each file */
	uint64_t   last_utc_timestamp;      /* unix time when last write called - supports digital_rf_get_last_write_time method */
	int        has_failure;				/* bool flag to detect a io error has occured, disallows all following writes */

} Digital_rf_write_object;

/* Public method declarations */


EXPORT int digital_rf_write_blocks_hdf5(
	Digital_rf_write_object *hdf5_data_object, uint64_t * global_index_arr, uint64_t * data_index_arr,
	uint64_t index_len, void * vector, uint64_t vector_length
);


#ifdef __cplusplus
	extern "C" EXPORT const char * digital_rf_get_version(void);
	extern "C" EXPORT int digital_rf_get_unix_time(
		uint64_t, long double, int*, int*, int*, int*, int*, int*, uint64_t*);
	extern "C" EXPORT int digital_rf_get_unix_time_rational(
		uint64_t, uint64_t, uint64_t, int*, int*, int*, int*, int*, int*, uint64_t*);
	extern "C" EXPORT Digital_rf_write_object * digital_rf_create_write_hdf5(
		char*, hid_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, char *, int, int, int, int, int, int);
	extern "C" EXPORT int digital_rf_write_hdf5(Digital_rf_write_object*, uint64_t, void*,uint64_t);
	extern "C" EXPORT char * digital_rf_get_last_file_written(Digital_rf_write_object *);
	extern "C" EXPORT char * digital_rf_get_last_dir_written(Digital_rf_write_object *);
	extern "C" EXPORT uint64_t digital_rf_get_last_write_time(Digital_rf_write_object *);
	extern "C" EXPORT int digital_rf_close_write_hdf5(Digital_rf_write_object*);

#else
	EXPORT const char * digital_rf_get_version(void);
	EXPORT int digital_rf_get_unix_time(uint64_t global_sample,
		long double sample_rate, int * year, int * month, int *day, int * hour,
		int * minute, int * second, uint64_t * picosecond);
	EXPORT int digital_rf_get_unix_time_rational(uint64_t global_sample,
		uint64_t sample_rate_numerator, uint64_t sample_rate_denominator,
		int * year, int * month, int *day, int * hour, int * minute,
		int * second, uint64_t * picosecond);
	EXPORT Digital_rf_write_object * digital_rf_create_write_hdf5(
		char * directory, hid_t dtype_id, uint64_t subdir_cadence_secs,
		uint64_t file_cadence_millisecs, uint64_t global_start_sample,
		uint64_t sample_rate_numerator,
		uint64_t sample_rate_denominator, char * uuid_str,
		int compression_level, int checksum, int is_complex,
		int num_subchannels, int is_continuous, int marching_dots);
	EXPORT int digital_rf_write_hdf5(Digital_rf_write_object *hdf5_data_object,
		uint64_t global_leading_edge_index, void * vector,
		uint64_t vector_length);
	EXPORT char * digital_rf_get_last_file_written(Digital_rf_write_object *hdf5_data_object);
	EXPORT char * digital_rf_get_last_dir_written(Digital_rf_write_object *hdf5_data_object);
	EXPORT uint64_t digital_rf_get_last_write_time(Digital_rf_write_object *hdf5_data_object);
	EXPORT int digital_rf_close_write_hdf5(Digital_rf_write_object *hdf5_data_object);
#endif

/* Private method declarations */
int digital_rf_get_timestamp_floor(uint64_t sample_index, uint64_t sample_rate_numerator,
								   uint64_t sample_rate_denominator, uint64_t * second, uint64_t * picosecond);
int digital_rf_get_sample_ceil(uint64_t second, uint64_t picosecond,
							   uint64_t sample_rate_numerator, uint64_t sample_rate_denominator, uint64_t * sample_index);
int digital_rf_get_time_parts(time_t unix_second, int * year, int * month, int *day,
		                     int * hour, int * minute, int * second);
int digital_rf_get_subdir_file(Digital_rf_write_object *hdf5_data_object, uint64_t global_sample,
							   char * subdir, char * basename, uint64_t * samples_left, uint64_t * max_samples_this_file);
int digital_rf_free_hdf5_data_object(Digital_rf_write_object *hdf5_data_object);
int digital_rf_check_hdf5_directory(char * directory);
uint64_t digital_rf_write_samples_to_file(Digital_rf_write_object *hdf5_data_object, uint64_t samples_written, uint64_t * global_index_arr,
		uint64_t * data_index_arr, uint64_t index_len, void * vector, uint64_t vector_length);
int digital_rf_create_hdf5_file(Digital_rf_write_object *hdf5_data_object, char * subdir, char * basename,
								uint64_t samples_to_write, uint64_t samples_left, uint64_t max_samples_this_file);
int digital_rf_close_hdf5_file(Digital_rf_write_object *hdf5_data_object);
int digital_rf_create_new_directory(Digital_rf_write_object *hdf5_data_object, char * subdir);
int digital_rf_set_fill_value(Digital_rf_write_object *hdf5_data_object);
void digital_rf_write_metadata(Digital_rf_write_object *hdf5_data_object);
uint64_t * digital_rf_create_rf_data_index(Digital_rf_write_object *hdf5_data_object, uint64_t samples_written, uint64_t samples_left,
		uint64_t max_samples_this_file, uint64_t * global_index_arr, uint64_t * data_index_arr, uint64_t index_len, uint64_t vector_len,
		uint64_t next_global_sample, int * rows_to_write, uint64_t * samples_to_write, int file_exists);
int digital_rf_write_rf_data_index(Digital_rf_write_object * hdf5_data_object, uint64_t * rf_data_index_arr, int block_index_len);
uint64_t digital_rf_get_global_sample(uint64_t samples_written, uint64_t * global_index_arr, uint64_t * data_index_arr,
		                              uint64_t index_len);
int digital_rf_extend_dataset(Digital_rf_write_object * hdf5_data_object, uint64_t samples_to_write);
int digital_rf_handle_metadata(Digital_rf_write_object * hdf5_data_object);
int digital_rf_is_little_endian(void);


#endif
