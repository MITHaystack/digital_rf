/*
 * Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
 * All rights reserved.
 *
 * Distributed under the terms of the BSD 3-clause license.
 *
 * The full license is in the LICENSE file, distributed with this software.
*/
/* Implementation of rf_hdf5 library
 *
  See digital_rf.h for overview of this module.

  Written 2/2014 by B. Rideout

  Major modification to Digital RF 2.0 in Nov 2015

  $Id$
*/

#ifdef _WIN32
#  include "wincompat.h"
#else
#  include <unistd.h>
#endif

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#include "digital_rf.h"


/* Public method implementations */
Digital_rf_write_object * digital_rf_create_write_hdf5(char * directory, hid_t dtype_id, uint64_t subdir_cadence_secs,
													  uint64_t file_cadence_millisecs, uint64_t global_start_sample,
													  uint64_t sample_rate_numerator,
													  uint64_t sample_rate_denominator, char * uuid_str,
		                                              int compression_level, int checksum, int is_complex,
		                                              int num_subchannels, int is_continuous, int marching_dots)
/*  digital_rf_create_write_hdf5 returns an Digital_rf_write_object used to write a single channel of RF data to
 * a directory, or NULL with error to standard error if failure.
 *
 * Inputs:
 * 		char * directory - a directory under which to write the resultant Hdf5 files.  Must already exist.
 * 			Hdf5 files will be stored as YYYY-MM-DDTHH-MM-SS/rf@<unix_second>.<3 digit millsecond>.h5
 * 		hid_t dtype_id - data type id as defined by hdf5.h
 * 		uint64_t subdir_cadence_secs - Number of seconds of data found in one subdir. For example, 3600
 * 			subdir_cadence_secs will be saved in each subdirectory
 * 		uint64_t file_cadence_millisecs - number of milliseconds of data per file. Rule:
 * 			subdir_cadence_secs*1000 % file_cadence_millisecs must equal 0
 * 		uint64_t global_start_sample - The start time of the first sample in units of samples since UT midnight 1970-01-01.
 * 		uint64_t sample_rate_numerator - sample rate numerator. Final sample rate is sample_rate_numerator/sample_rate_denominator in Hz
 * 		uint64_t sample_rate_denominator - sample rate denominator. Final sample rate is sample_rate_numerator/sample_rate_denominator in Hz
 * 		        These two arguments enforce the rule that sample rates must be a rational fraction.
 * 		char * uuid_str - a string containing a UUID generated for that channel.  uuid_str saved in
 * 			each resultant Hdf5 file's metadata.
 * 		int compression_level - if 0, no compression used.  If 1-9, level of gzip compression.  Higher compression
 * 			means smaller file size and more time used.
 * 		int checksum - if non-zero, HDF5 checksum used.  If 0, no checksum used.
 * 		int is_complex - 1 if complex (IQ) data, 0 if single-valued
 * 		int num_subchannels - the number of subchannels of complex or single valued data recorded at once.
 * 			Must be 1 or greater. Note: A single stream of complex values is one subchannel, not two.
 * 		int is_continuous - 1 if data will be stored in a continuous format (rf_data_index has length 1 for each file, files contain all
 *			possible samples, gaps are filled with a fill value except when entire file can be omitted)
 *			0 if data will be stored in a gapped format (rf_data_index identifies continuous chunks in solid rf_data dataset, no fill values
 *			are used, files can be smaller than maximum size)
 * 		int marching_dots - non-zero if marching dots desired when writing; 0 if not
 *
 * 	Hdf5 format
 *
 * 	/rf_data - dataset of size (*,) or (*, num_subchannels), datatype = dtype_id
 * 	/rf_data_index - dataset of size (num of separate block of data, 2), datatype - uint_64, length at least 1
 *  /rf_data has 14 attributes: sequence_num (int), subdir_cadence_secs (int), uuid_str (string), sample_rate_numerator (uint_64),
 *      sample_rate_denominator (uint_64),
 *  	is_complex (0 or 1 - int), num_subchannels (int), computer_time (unix timestamp from computer when
 *  	file created) (uint64_t), init_utc_timestamp (unix timestamp from computer when first sample written - used to
 *  	determine leapsecond offset), epoch ('1970-01-01 00:00:00 UT'), digital_rf_time_description - string describing
 *  	sample time used, file_cadence_millisecs, digital_rf_version (now '2.0', was '1.0'), and is_continuous (0 or 1)
 */
{
	/* local variables */
	Digital_rf_write_object * hdf5_data_object;
	hsize_t  chunk_dims[2];

	/* allocate overall object */
	if ((hdf5_data_object = (Digital_rf_write_object *)malloc(sizeof(Digital_rf_write_object)))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}
	/* init everything to NULL that will be malloced */
	hdf5_data_object->directory = NULL;
	hdf5_data_object->sub_directory = NULL;
	hdf5_data_object->uuid_str = NULL;
	hdf5_data_object->dataset = 0;
	hdf5_data_object->dataset_prop = 0;
	hdf5_data_object->dataspace = 0;
	hdf5_data_object->filespace = 0;
	hdf5_data_object->memspace = 0;
	hdf5_data_object->hdf5_file = 0; /* indicates no Hdf5 file presently opened */
	hdf5_data_object->index_dataset = 0;
	hdf5_data_object->index_prop = 0;
	hdf5_data_object->next_index_avail = 0;

	/* set directory name */
	if ((hdf5_data_object->directory = (char *)malloc(sizeof(char) * (strlen(directory)+2)))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}
	strcpy(hdf5_data_object->directory, directory);
	if (hdf5_data_object->directory[strlen(hdf5_data_object->directory)] != '/')
		strcat(hdf5_data_object->directory, "/");

	if (digital_rf_check_hdf5_directory(hdf5_data_object->directory))
	{
		fprintf(stderr, "%s does not exist or is not a directory\n", hdf5_data_object->directory);
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}

	/* set UUID */
	if ((hdf5_data_object->uuid_str = (char *)malloc(sizeof(char) * (strlen(uuid_str)+1)))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}
	strcpy(hdf5_data_object->uuid_str, uuid_str);

	if (compression_level < 0 || compression_level > 9)
	{
		fprintf(stderr, "Illegal compression level, must be 0-9\n");
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}

	if (num_subchannels < 1)
	{
		fprintf(stderr, "Illegal num_subchannels %i, must be greater than 0\n", num_subchannels);
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}
	hdf5_data_object->num_subchannels = num_subchannels;

	/* check cadence rules */
	if (subdir_cadence_secs < 1)
	{
		fprintf(stderr, "Illegal subdir_cadence_secs %" PRIu64 ", must be greater than 0\n", subdir_cadence_secs);
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}
	if (file_cadence_millisecs < 1)
	{
		fprintf(stderr, "Illegal file_cadence_millisecs %" PRIu64 ", must be greater than 0\n", file_cadence_millisecs);
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}
	if (subdir_cadence_secs*1000 % file_cadence_millisecs != 0)
	{
		fprintf(stderr, "Illegal subdir_cadence_secs %" PRIu64 ", file_cadence_millisecs %" PRIu64 " combination, subdir_cadence_secs*1000 %% file_cadence_millisecs must equal 0\n",
				subdir_cadence_secs, file_cadence_millisecs);
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}

	/* init other values in hdf5_data_object */
	hdf5_data_object->subdir_cadence_secs = subdir_cadence_secs;
	hdf5_data_object->file_cadence_millisecs = file_cadence_millisecs;
	hdf5_data_object->global_start_sample = global_start_sample;
	hdf5_data_object->sample_rate_numerator = sample_rate_numerator;
	hdf5_data_object->sample_rate_denominator = sample_rate_denominator;
	hdf5_data_object->sample_rate =  (long double)sample_rate_numerator / (long double)sample_rate_denominator;
	hdf5_data_object->dtype_id = dtype_id;
	hdf5_data_object->global_index = 0; /* index of next sample to write */
	hdf5_data_object->present_seq = -1; /* present file seq number, -1 because no file open */
	hdf5_data_object->dataset_index = 0; /* where in the dataset the next write should occur */
	hdf5_data_object->dataset_avail = 0; /* how many samples are free to write to it the open Hdf5 file */
	hdf5_data_object->block_index = 0;   /* the next available row in the open Hdf5 file/rf_data_index dataset to write to */
	hdf5_data_object->is_continuous = is_continuous;
	hdf5_data_object->marching_dots = marching_dots;
	hdf5_data_object->has_failure = 0;  /* this will be set to 1 if there is an IO error, disabling further writes */

	/* init_utc_timestamp - stored as attribute to allow conversion to astronomical times */
	hdf5_data_object->init_utc_timestamp = (uint64_t)(global_start_sample/hdf5_data_object->sample_rate);
	hdf5_data_object->last_utc_timestamp = 0; /* no last write time yet */

	if (hdf5_data_object->num_subchannels == 1)
		hdf5_data_object->rank = 1;
	else
		hdf5_data_object->rank = 2;

	if (is_complex)
	{
		hdf5_data_object->is_complex = 1;
		/* create complex compound datatype */
		hdf5_data_object->complex_dtype_id = H5Tcreate(H5T_COMPOUND, 2*H5Tget_size(hdf5_data_object->dtype_id));
		/* create r column */
		H5Tinsert (hdf5_data_object->complex_dtype_id, "r", 0, hdf5_data_object->dtype_id);
		/* create i column */
		H5Tinsert (hdf5_data_object->complex_dtype_id, "i", H5Tget_size(hdf5_data_object->dtype_id),
				   hdf5_data_object->dtype_id);
	}
	else
	{
		hdf5_data_object->is_complex = 0;
		hdf5_data_object->complex_dtype_id = (hid_t)0; /* make sure its not used by accident */
	}

	/* check for illegal values */
	if (hdf5_data_object->sample_rate <= 0.0)
	{
		fprintf(stderr, "Illegal sample_rate, must be positive\n");
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}
	/* max_chunk_size is smallest possible value of maximum number of samples in a file */
	/* (with non-integer sampling rates, files can have different max number of samples) */
	hdf5_data_object->max_chunk_size = (uint64_t)((file_cadence_millisecs/1000.0)*hdf5_data_object->sample_rate);


	/* dataset_prop is constant, so we can start to set this up in init */
	hdf5_data_object->dataset_prop = H5Pcreate (H5P_DATASET_CREATE);
	if (compression_level != 0)
		H5Pset_deflate (hdf5_data_object->dataset_prop, compression_level);
	if (checksum)
		H5Pset_filter (hdf5_data_object->dataset_prop, H5Z_FILTER_FLETCHER32, 0, 0, NULL);
	hdf5_data_object->chunk_size = 0;
	if (checksum || compression_level != 0 || is_continuous != 1)
		hdf5_data_object->needs_chunking = 1;
	else
		hdf5_data_object->needs_chunking = 0;

	/* set fill value for data gaps according to input dtype_id */
	if (digital_rf_set_fill_value(hdf5_data_object))
	{
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}

	/* index_prop is constant so we can start to set this up in init */
	hdf5_data_object->index_prop = H5Pcreate (H5P_DATASET_CREATE);
	chunk_dims[0] = CHUNK_SIZE_RF_DATA_INDEX;
	chunk_dims[1] = 2;
	H5Pset_chunk (hdf5_data_object->index_prop, 2, chunk_dims);

	if (digital_rf_handle_metadata(hdf5_data_object))
	{
		digital_rf_close_write_hdf5(hdf5_data_object);
		return(NULL);
	}

	/* done - return object */
	return(hdf5_data_object);
}



int digital_rf_write_hdf5(Digital_rf_write_object *hdf5_data_object, uint64_t global_leading_edge_index, void * vector,
						  uint64_t vector_length)
/*
 * digital_rf_write_hdf5 writes a continuous block of data from vector into one or more Hdf5 files
 *
 * Inputs:
 * 		Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 * 		uint64_t global_leading_edge_index - index to write data to.  This is a global index with zero
 * 			representing the sample taken at the time global_start_sample specified in the init method.
 * 			Note that all values stored in Hdf5 file will have global_start_sample added, and this offset
 * 			should NOT be added by the user. Error raised and -1 returned if before end of last write.
 * 		void * vector - pointer into data vector to write
 * 		uint64_t vector_length - number of samples to write to Hdf5
 *
 * 	Affects - Writes data to existing open Hdf5 file.  May close that file and write some or all of remaining data to
 * 		new Hdf5 file.
 *
 * 	Returns 0 if success, non-zero and error written if failure.
 *
 */
{
	uint64_t data_index_arr[1] = {0};
	uint64_t index_len = 1;
	int result;

	if (hdf5_data_object->has_failure)
	{
		fprintf(stderr, "A previous fatal io error precludes any further calls to digital_rf_write_hdf5.\n");
		return(-1);
	}

	/* just call full method digital_rf_write_blocks_hdf5 under the covers */
	result = digital_rf_write_blocks_hdf5(hdf5_data_object, &global_leading_edge_index, data_index_arr, index_len, vector, vector_length);
	return(result);

}


int digital_rf_write_blocks_hdf5(Digital_rf_write_object *hdf5_data_object, uint64_t * global_index_arr, uint64_t * data_index_arr,
		                         uint64_t index_len, void * vector, uint64_t vector_length)
/*
 * digital_rf_write_blocks_hdf5 writes blocks of data from vector into one or more Hdf5 files
 *
 * Inputs:
 * 		Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 * 		uint64_t * global_index_arr - an array of global indices into the samples being written.  The global
 * 			index is the total number of sample periods since data taking began, including gaps.  Note that
 * 			all values stored in Hdf5 file will have global_start_sample added, and this offset
 * 			should NOT be added by the user.  Error is raised if any value is before its expected value (meaning repeated data).
 * 		uint64_t * data_index_arr - an array of len = len(global_index_arr), where the indices are related to which
 * 			sample in the vector being passed in is being referred to in global_index_arr.  First value must be 0
 * 			or error raised.  Values must be increasing, and cannot be equal or greater than vector_length or error raised.
 * 		uint_64 index_len - the len of both global_index_arr and data_index_arr.  Must be greater than 0.
 * 		void * vector - pointer into data vector to write
 * 		uint64_t vector_length - number of samples to write to Hdf5
 *
 * 	Affects - Writes data to existing open Hdf5 file.  May close that file and write some or all of remaining data to
 * 		new Hdf5 file.
 *
 * 	Returns 0 if success, non-zero and error written if failure.
 *
 */
{
	char error_str[SMALL_HDF5_STR] = "";
	uint64_t samples_written = 0; /* total samples written so far to all Hdf5 files during this write call */
	uint64_t dataset_samples_written = 0; /* number of samples written to the present file */
	hsize_t chunk_dims[2] = {0, hdf5_data_object->num_subchannels};
	hsize_t chunk_size = 0;

	if (hdf5_data_object->has_failure)
	{
		fprintf(stderr, "A previous fatal io error precludes any further calls to digital_rf_write_blocks_hdf5.\n");
		return(-1);
	}

	/* verify data exists */
	if (!vector)
	{
		snprintf(error_str, SMALL_HDF5_STR, "Null data passed in\n");
		fprintf(stderr, "%s", error_str);
		return(-2);
	}

	/* verify not writing in the past */
	if (global_index_arr[0] < hdf5_data_object->global_index)
	{
		snprintf(error_str, SMALL_HDF5_STR, "Request index %" PRIu64 " before first expected index %" PRIu64 " in digital_rf_write_hdf5\n",
				global_index_arr[0], hdf5_data_object->global_index);
		fprintf(stderr, "%s", error_str);
		return(-3);
	}

	/* set chunking if needed */
	if (hdf5_data_object->needs_chunking && !hdf5_data_object->chunk_size)
	{
		if (vector_length*10 < hdf5_data_object->max_chunk_size)
			chunk_size = vector_length*10;
		else if (vector_length < hdf5_data_object->max_chunk_size)
			chunk_size = vector_length;
		else
			chunk_size = hdf5_data_object->max_chunk_size;
		hdf5_data_object->chunk_size = chunk_size;
		chunk_dims[0] = chunk_size;
		H5Pset_chunk (hdf5_data_object->dataset_prop, hdf5_data_object->rank, chunk_dims);
	}

	/* verify continuous if is_continuous */
	if (hdf5_data_object->is_continuous && index_len > 1)
	{
		snprintf(error_str, SMALL_HDF5_STR, "Gapped data passed in, but is_continuous set\n");
		fprintf(stderr, "%s", error_str);
		return(-4);
	}

	/* loop until all data written - this loop breaks multiple file writes into a series single file writes*/
	while (samples_written < vector_length)
	{
		dataset_samples_written = digital_rf_write_samples_to_file(hdf5_data_object, samples_written,
				global_index_arr, data_index_arr, index_len, vector, vector_length);
		if (dataset_samples_written == 0)
		{
			fprintf(stderr, "Problem detected, dataset_samples_written = 0 after  %" PRIu64 " samples_written\n", samples_written);
			return(-6);
		}
		samples_written += dataset_samples_written;
	}


	return(0);
}

char * digital_rf_get_last_file_written(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_get_last_file_written returns a malloced string containing the full path to the last hdf5 file written to
 *
 * Inputs:
 * 		Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 *
 * 	Returns:
 * 		char * containing the full path to the last hdf5 file written to. User is responsible for freeing string when done.
 * 		Returns empty string if no data written.
 */
{
	/* local variables */
	char fullpath[BIG_HDF5_STR] = ""; 	/* to be set to the full path */
	char * ret_str;  /* string to be malloced */

	if (hdf5_data_object->sub_directory == NULL)
	{
		/* no data written yet */
		if ((ret_str = (char *)malloc(sizeof(char) * (2)))==0)
		{
			fprintf(stderr, "malloc failure - unrecoverable\n");
			exit(-1);
		}
		strcpy(ret_str, "");
		return(ret_str);
	}

	strcpy(fullpath, hdf5_data_object->directory);
	strcat(fullpath, hdf5_data_object->sub_directory);
	strcat(fullpath, strstr(hdf5_data_object->basename, "rf"));
	if ((ret_str = (char *)malloc(sizeof(char) * (strlen(fullpath)+2)))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}
	strcpy(ret_str, fullpath);
	return(ret_str);
}


char * digital_rf_get_last_dir_written(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_get_last_dir_written returns a malloced string containing the full path to the last dir written to
 *
 * Inputs:
 * 		Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 *
 * 	Returns:
 * 		char * containing the full path to the last directory written to. User is responsible for freeing string when done.
 * 		Returns empty string if no data written.
 */
{
	/* local variables */
	char fullpath[BIG_HDF5_STR] = ""; 	/* to be set to the full path */
	char * ret_str;  /* string to be malloced */

	if (hdf5_data_object->sub_directory == NULL)
	{
		/* no data written yet */
		if ((ret_str = (char *)malloc(sizeof(char) * (2)))==0)
		{
			fprintf(stderr, "malloc failure - unrecoverable\n");
			exit(-1);
		}
		strcpy(ret_str, "");
		return(ret_str);
	}

	strcpy(fullpath, hdf5_data_object->directory);
	strcat(fullpath, hdf5_data_object->sub_directory);
	if ((ret_str = (char *)malloc(sizeof(char) * (strlen(fullpath)+2)))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}
	strcpy(ret_str, fullpath);
	return(ret_str);
}



uint64_t digital_rf_get_last_write_time(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_get_last_write_time returns the unix timestamp of the last write
 *
 * Inputs:
 * 		Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 *
 * 	Returns: uint64_t representing the unix timestamp of the last write.  If no writes occurred yet, returns 0.
 */
{
	return(hdf5_data_object->last_utc_timestamp);
}


int digital_rf_close_write_hdf5(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_close_write_hdf5 closes open Hdf5 file if needed and releases all memory associated with hdf5_data_object
 *
 * Inputs:
 * 		Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 */
{
	if (hdf5_data_object != NULL)
	{
		/* close file */
		H5Dclose (hdf5_data_object->dataset);
		hdf5_data_object->dataset = 0;
		H5Dclose (hdf5_data_object->index_dataset);
		hdf5_data_object->index_dataset = 0;
		H5Sclose (hdf5_data_object->dataspace);
		hdf5_data_object->dataspace = 0;
		if (hdf5_data_object->filespace)
		{
			H5Sclose (hdf5_data_object->filespace);
			hdf5_data_object->filespace = 0;
		}
		if (hdf5_data_object->memspace)
		{
			H5Sclose (hdf5_data_object->memspace);
			hdf5_data_object->memspace = 0;
		}
		H5Fclose (hdf5_data_object->hdf5_file);
		hdf5_data_object->hdf5_file = 0;
		hdf5_data_object->dataset_index = 0;

		/* rename closed file to finalized name (or delete if errored) */
		digital_rf_close_hdf5_file(hdf5_data_object);

		/* finally free all resources in hdf5_data_object */
		digital_rf_free_hdf5_data_object(hdf5_data_object);
	}
	return(0);
}


int digital_rf_get_unix_time(uint64_t global_sample, long double sample_rate, int * year, int * month, int *day,
		                     int * hour, int * minute, int * second, uint64_t * picosecond)
/* get_unix_time converts a global_sample and a sample rate into year, month, day
 * 	hour, minute, second, picosecond
 *
 * 	Returns 0 if success, -1 if failure.
 *
 */
{
	struct tm *gm;
	time_t unix_second;
	long double unix_remainder;

	/* set values down to second using gmtime */
	unix_second = (time_t)(global_sample / sample_rate);
	gm = gmtime(&unix_second);
	if (gm == NULL)
		return(-1);
	*year = gm->tm_year + 1900;
	*month = gm->tm_mon + 1;
	*day = gm->tm_mday;
	*hour = gm->tm_hour;
	*minute = gm->tm_min;
	*second = gm->tm_sec;

	/* set picoseconds */
	if (fmod(sample_rate, 1.0) == 0.0) /* use integer logic when sample rate can be converted to an integer */
		unix_remainder = (long double)(global_sample - (unix_second * (uint64_t)sample_rate));
	else
		unix_remainder = fmodl((long double)global_sample, sample_rate);
	*picosecond = (uint64_t)floorl((unix_remainder/sample_rate)*1.0E12L + 0.5L);
	return(0);
}


int digital_rf_get_unix_time_rational(uint64_t global_sample,
	uint64_t sample_rate_numerator, uint64_t sample_rate_denominator,
	int * year, int * month, int *day, int * hour, int * minute, int * second,
	uint64_t * picosecond)
/* get_unix_time_rational converts a global_sample and a sample rate into year,
 *  month, day, hour, minute, second, picosecond
 *
 * 	Returns 0 if success, -1 if failure.
 *
 */
{
	struct tm *gm;
	time_t unix_second;
	uint64_t unix_remainder;

	/* set values down to second using gmtime */
	unix_second = (time_t)(global_sample * sample_rate_denominator / sample_rate_numerator);
	gm = gmtime(&unix_second);
	if (gm == NULL)
		return(-1);
	*year = gm->tm_year + 1900;
	*month = gm->tm_mon + 1;
	*day = gm->tm_mday;
	*hour = gm->tm_hour;
	*minute = gm->tm_min;
	*second = gm->tm_sec;

	/* set picoseconds */
	unix_remainder = global_sample - (unix_second * sample_rate_numerator / sample_rate_denominator);
	*picosecond = unix_remainder * 1000000000000 * sample_rate_denominator / sample_rate_numerator;
	return(0);
}



/* Private Method implementations */


int digital_rf_get_subdir_file(Digital_rf_write_object *hdf5_data_object, uint64_t global_sample,
							   char * subdir, char * basename, uint64_t * samples_left, uint64_t * max_samples_this_file)
/* digital_rf_get_subdir_file sets the name of the derived subdir in form YYYY-MM-DDTHH-MM-SS and basename in form
 * rf@%llu.3f.h5
 *
 * Inputs:
 * 	Digital_rf_write_object *hdf5_data_object - C struct created by digital_rf_create_write_hdf5
 * 	global_sample - global sample of first sample to write to file (first sample written is 0)
 * 	subdir - char array allocated by caller to be filled in with YYYY-MM-DDTHH-MM-SS according to global_sample
 * 		and subdir_cadence_secs
 * 	basename - char array allocated by caller to be filled in with rf@%llu.3f.tmp.h5 according to global_sample and
 * 		subdir_cadence_secs and file_cadence_millisecs.  Note that when file is no longer actively written to, or
 * 		when digital_rf_close_write_hdf5 called, will be moved to remove tmp. part of name.
 * 	samples_left - pointer to uint64_t to be set to the number of samples left to be written in file.  Will be greater than
 * 		zero and less than or equal to max_samples_this_file
 *  max_samples_this_file - pointer to uint64_t to be set to the maximum total number of samples that this file can hold
 *
 * 	Returns 0 if success, -1 if failure.
 *
 */
{
	int year, month, day, hour, minute, second; /* for time conversion */
	uint64_t picosecond;			/* for time conversion */
	uint64_t sample_sec;      		/* the unix second associated with global_sample */
	uint64_t sample_millisec; 		/* the unix millisecond associated with global_sample */
	uint64_t dir_sec;  				/* the unix second the directory starts */
	uint64_t file_millisec;  		/* the unix millisecond the file starts */
	uint64_t file_sec_part; 		/* second part of file name */
	uint64_t file_millisec_part; 	/* millisecond part of file name */
	uint64_t file_start_sample;		/* first sample in file */
	uint64_t next_file_sec_part;	/* second part of *next* file name */
	uint64_t next_file_millisec_part;		/* millisecond part of *next* file name */
	uint64_t next_file_start_sample;		/* first sample in *next* file */

	/* convert global_sample to absolute samples since 1970 */
	global_sample += hdf5_data_object->global_start_sample;

	/* first derive the subdirectory name */
	sample_sec = (uint64_t)(global_sample/hdf5_data_object->sample_rate);
	dir_sec = (sample_sec / hdf5_data_object->subdir_cadence_secs) * hdf5_data_object->subdir_cadence_secs;
	if (digital_rf_get_unix_time(dir_sec, 1.0, &year, &month, &day, &hour, &minute, &second, &picosecond))
		return(-1);
	snprintf(subdir, BIG_HDF5_STR, "%04i-%02i-%02iT%02i-%02i-%02i", year, month, day, hour, minute, second);

	/* next derive the file name */
	sample_millisec = (uint64_t)((global_sample/hdf5_data_object->sample_rate)*1000);
	file_millisec = (sample_millisec / hdf5_data_object->file_cadence_millisecs) * hdf5_data_object->file_cadence_millisecs;
	file_sec_part = file_millisec / 1000; /* second part of file name */
	file_millisec_part = file_millisec % 1000; /* millisecond part of file name */
	snprintf(basename, SMALL_HDF5_STR, "tmp.rf@%" PRIu64 ".%03" PRIu64 ".h5", file_sec_part, file_millisec_part);
	file_millisec += hdf5_data_object->file_cadence_millisecs; /* now file_millisec is the unix start millisecond of the *next* file */
	next_file_sec_part = file_millisec / 1000;
	next_file_millisec_part = file_millisec % 1000;

	/* finally derive the number of samples available to write in this file (independent of whether file exists) */
	/* ceil needed because implicit flooring puts sample at time just before file starts */
	file_start_sample = (uint64_t)ceill(file_sec_part*hdf5_data_object->sample_rate + (file_millisec_part*hdf5_data_object->sample_rate)/1000);
	next_file_start_sample = (uint64_t)ceill(next_file_sec_part*hdf5_data_object->sample_rate + (next_file_millisec_part*hdf5_data_object->sample_rate)/1000);
	*samples_left = next_file_start_sample - global_sample;
	*max_samples_this_file = next_file_start_sample - file_start_sample;

	if (*samples_left < 1 || *samples_left > *max_samples_this_file)
	{
		fprintf(stderr, "got illegal samples_left %" PRIu64 "\n", *samples_left);
		return(-1);
	}

	return(0);
}


int digital_rf_free_hdf5_data_object(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_free_hdf5_data_object frees all resources in hdf5_data_object */
{
	if (hdf5_data_object->directory != NULL)
		free(hdf5_data_object->directory);
	if (hdf5_data_object->sub_directory != NULL)
		free(hdf5_data_object->sub_directory);
	if (hdf5_data_object->uuid_str != NULL)
		free(hdf5_data_object->uuid_str);

	/* free all Hdf5 resources */
	if (hdf5_data_object->dataset)
		H5Dclose (hdf5_data_object->dataset);
	if (hdf5_data_object->dataset_prop)
		H5Pclose (hdf5_data_object->dataset_prop);
	if (hdf5_data_object->dataspace)
		H5Sclose (hdf5_data_object->dataspace);
	if (hdf5_data_object->filespace)
		H5Sclose (hdf5_data_object->filespace);
	if (hdf5_data_object->memspace)
		H5Sclose (hdf5_data_object->memspace);
	if (hdf5_data_object->index_dataset)
		H5Dclose (hdf5_data_object->index_dataset);
	if (hdf5_data_object->index_prop)
		H5Pclose (hdf5_data_object->index_prop);
	if (hdf5_data_object->hdf5_file)
		H5Fclose (hdf5_data_object->hdf5_file);
	free(hdf5_data_object);

	return(0);
}

int digital_rf_check_hdf5_directory(char * directory)
/* digital_rf_check_hdf5_directory checks if directory exists.  If it does not
 * exist, returns -1.  If okay, returns 0.
 *
 */
{
	/* local variable */
	struct stat stat_obj = {0}; /* used to run stat to determine if directory exists */

	/* see if directory needs to be created */
	if (stat(directory, &stat_obj))
	{
		return(-1);
	} else {
		/* verify it's a directory */
		if(!S_ISDIR(stat_obj.st_mode))
		{
			return(-1);
		}
	}
	return(0);
}


uint64_t digital_rf_write_samples_to_file(Digital_rf_write_object *hdf5_data_object, uint64_t samples_written, uint64_t * global_index_arr,
		uint64_t * data_index_arr, uint64_t index_len, void * vector, uint64_t vector_length)
/* digital_rf_write_samples_to_file writes some or all of the data to a single Hdf5 file.
 *
 * Returns the number of samples written. Returns 0 if error.
 *
 * Inputs:
 * 	Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 * 	uint64_t samples_written - the number of samples written to previous files during this particular user write call
 * 	uint64_t * global_index_arr - an array of global indices into the samples being written.  The global
 * 		index is the total number of sample periods since data taking began, including gaps.  Error
 * 		is raised if any value is before its expected value (meaning repeated data).
 * 	uint64_t * data_index_arr - an array of len = len(global_index_arr), where the indices are related to which
 * 		sample in the vector being passed in by the user, so that the first value is always 0,
 * 		or error raised.  Values must be increasing, and cannot be equal or greater than index_len or error raised.
 * 	uint_64 index_len - the len of both global_index_arr and data_index_arr.  Must be greater than 0.
 * 	void * vector - the vector (either single valued or complex) containing the data, or NULL pointer if no data to be written.
 * 	uint64_t vector_length - the total number of samples to write from vector.
 *
 */
{
	/* local variables */
	int64_t last_global_index; /* may possibly be negative if first write not to first sample in file */
	uint64_t samples_to_write, samples_after_last_global_index;
	uint64_t next_global_index;
	int block_index_len; /* len of /rf_data_index dataset needed for this particular write */
	uint64_t * rf_data_index_arr = NULL; /* will be malloced and filled out with all data needed for rf_data_index table */
	int result;
	hsize_t size[2] = {0, hdf5_data_object->num_subchannels}; /* will be set to size of full dataset in file */
	hsize_t offset[2] = {0,0};        /* will be set to the index in file writing to */
	herr_t  status;                   /* Hdf5 error status */
	int file_exists;                  /* set to 1 if file being written to already exists, 0 if not */
	time_t  computer_time;            /* these two variables used to update last_unix_time */
	int64_t u_computer_time;

	char subdir[BIG_HDF5_STR] = ""; 	/* to be set to the subdirectory to write to */
	char basename[SMALL_HDF5_STR] = ""; /* to be set to the file basename to write to */
	char subdir_with_trailing_slash[SMALL_HDF5_STR] = "";
	uint64_t samples_left = 0;				/* to be set to the number of samples available to write to this file */
	uint64_t max_samples_this_file;			/* to be set to the maximum total number of samples that can be in this file */


	/* verify inputs are sensible */
	if (index_len < 1)
	{
		fprintf(stderr, "Illegal index_len %" PRIu64 " in digital_rf_write_samples_to_file\n", index_len);
		return(0);
	}
	if (data_index_arr[0] != 0)
	{
		fprintf(stderr, "Illegal first value %" PRIu64 " in data_index_arr, must be 0\n", data_index_arr[0]);
				return(0);
	}

	/* get needed subdir, basename, and samples_left */
	next_global_index = digital_rf_get_global_sample(samples_written, global_index_arr, data_index_arr, index_len);
	result = digital_rf_get_subdir_file(hdf5_data_object, next_global_index, subdir, basename, &samples_left, &max_samples_this_file);
	if (result) return(0);

	/* We need to see if we need to open a new file or expand an existing one */
	strcpy(subdir_with_trailing_slash, subdir);
	strcat(subdir_with_trailing_slash, "/"); /* allows us to compare subdir with sub_directory */
	if (hdf5_data_object->sub_directory == NULL || strcmp(hdf5_data_object->sub_directory, subdir_with_trailing_slash)
			|| strcmp(hdf5_data_object->basename, basename))
		file_exists = 0;
	else
		file_exists = 1;

	/* get all the info needed to create (or expand) /rf_data_index and fill it out, along with calculating samples_to_write  */
	rf_data_index_arr = digital_rf_create_rf_data_index(hdf5_data_object, samples_written, samples_left, max_samples_this_file, global_index_arr,
				data_index_arr, index_len, vector_length, next_global_index, &block_index_len, &samples_to_write, file_exists);
	assert(samples_to_write <= max_samples_this_file);
	if (!rf_data_index_arr && block_index_len == -1)
		return(0);

	/* Either create new file or expand old one or if contiguous continue writing to old one */
	if (!file_exists)
	{
		result = digital_rf_create_hdf5_file(hdf5_data_object, subdir, basename, samples_to_write, samples_left, max_samples_this_file);
		if (result)
		{
			fprintf(stderr, "failed to create subdir %s, basename %s\n", subdir, basename);
			if (rf_data_index_arr)
				free(rf_data_index_arr);
			return(0);
		}
	}
	else
	{
		if (hdf5_data_object->needs_chunking)
		{
			/* expand this file's dataset to hold the new data */
			digital_rf_extend_dataset(hdf5_data_object, samples_to_write);
		}
		else
		{
			/* set global and dataset indices in case we skipped samples since
			 * previous write, samples_left calculated earlier based on next_global_index */
			hdf5_data_object->global_index = next_global_index;
			hdf5_data_object->dataset_index = max_samples_this_file - samples_left;
		}
		assert(hdf5_data_object->dataset_index + samples_to_write <= max_samples_this_file);
	}

	/* create dataspace hyperslab to write to */
	if (hdf5_data_object->filespace)
		H5Sclose (hdf5_data_object->filespace);
	hdf5_data_object->filespace = H5Dget_space(hdf5_data_object->dataset);
	offset[0] = hdf5_data_object->dataset_index;
	size[0] = samples_to_write;
	H5Sselect_hyperslab(hdf5_data_object->filespace, H5S_SELECT_SET,
						offset, NULL, size, NULL);

	/* set up memspace to control write */
	if (hdf5_data_object->memspace)
		H5Sclose (hdf5_data_object->memspace);
	hdf5_data_object->memspace = H5Screate_simple(hdf5_data_object->rank, size, NULL);

	/* write rf_data */
	if (hdf5_data_object->is_complex == 0)
		status = H5Dwrite(hdf5_data_object->dataset, hdf5_data_object->dtype_id, hdf5_data_object->memspace,
						  hdf5_data_object->filespace, H5P_DEFAULT,
						  (char *)vector + (samples_written * H5Tget_size(hdf5_data_object->dtype_id) * hdf5_data_object->num_subchannels));
	else /* complex */
		status = H5Dwrite(hdf5_data_object->dataset, hdf5_data_object->complex_dtype_id, hdf5_data_object->memspace,
						  hdf5_data_object->filespace, H5P_DEFAULT,
						  (char *)vector + (samples_written * H5Tget_size(hdf5_data_object->dtype_id) * 2* hdf5_data_object->num_subchannels));

	if (status < 0)
	{
		H5Eprint(H5E_DEFAULT, stderr);
		hdf5_data_object->has_failure = 1;
		free(rf_data_index_arr);
		return(0);
	}

	/* write rf_data_index dataset */
	if (block_index_len > 0)
	{
		if (digital_rf_write_rf_data_index(hdf5_data_object, rf_data_index_arr, block_index_len))
		{
			free(rf_data_index_arr);
			return(0);
		}
	}

	/* advance state */
	hdf5_data_object->dataset_index += samples_to_write;
	hdf5_data_object->dataset_avail -= samples_to_write;
	/* update global_index - see meaning of rf_data_index_arr and block_index_len for logic.
	 * Basically using last line and counting extra samples after last global index */
	if (block_index_len > 0)
	{
		/* last_global_index is relative to global_start_sample, and it could be
		 * negative if write is continuous + non-chunked and starts in middle of file */
		last_global_index = rf_data_index_arr[block_index_len*2 - 2] - hdf5_data_object->global_start_sample;
		/* dataset_index is adjusted forward when write starts in middle of file
		 * so any negativity in last_global_index is made up here */
		samples_after_last_global_index = hdf5_data_object->dataset_index - rf_data_index_arr[block_index_len*2 - 1];
		hdf5_data_object->global_index = last_global_index + samples_after_last_global_index;
		free(rf_data_index_arr);
	}
	else
	{
		/* if no indices were used, then all data written must have been continuous */
		hdf5_data_object->global_index += samples_to_write;
	}

	assert(hdf5_data_object->dataset_index <= max_samples_this_file); /* debug */

	/* update hdf5_data_object->last_unix_time */
	computer_time = time(NULL);
	u_computer_time = (uint64_t)computer_time;
	hdf5_data_object->last_utc_timestamp = u_computer_time;

	return(samples_to_write);
}


int digital_rf_create_hdf5_file(Digital_rf_write_object *hdf5_data_object, char * subdir, char * basename,
								uint64_t samples_to_write, uint64_t samples_left, uint64_t max_samples_this_file)
/* digital_rf_create_hdf5_file opens a new Hdf5 file
 *
 * Inputs:
 * 	Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 * 	subdir - base subdir to create file in.  May need to be created.
 * 	basename - basename of file to create.
 * 	samples_to_write - number of samples to write in this write call (may need to be expanded later)
 * 	samples_left - number of samples that can possible be written in this file.  If not max_samples_this_file, affects
 * 		dataset_index if not needs_chunking
 *  max_samples_this_file - total number of samples that can be in this file
 *
 * 	Creates a file with /rf_data dataset of size (*, 2)
 *
 * 	Returns 0 if success, -1 if failure
 *
 */
{
	/* local variables */
	char datasetname[] = "rf_data";
	char fullname[BIG_HDF5_STR] = "";
	char finished_fullname[BIG_HDF5_STR] = "";
	char subdir_with_trailing_slash[SMALL_HDF5_STR] = "";
	char error_str[BIG_HDF5_STR] = "";
	uint64_t num_rows = 0;
	hsize_t  dims[2]  = {0, hdf5_data_object->num_subchannels};
	hsize_t  maxdims[2] = {max_samples_this_file, hdf5_data_object->num_subchannels};

    if (hdf5_data_object->marching_dots)
    {
		printf(".");
		fflush(stdout);
    }

    if (hdf5_data_object->hdf5_file != 0)
	{
		/* close previous file */
		H5Dclose (hdf5_data_object->dataset);
		hdf5_data_object->dataset = 0;
		H5Dclose (hdf5_data_object->index_dataset);
		hdf5_data_object->index_dataset = 0;
		H5Sclose (hdf5_data_object->dataspace);
		hdf5_data_object->dataspace = 0;
		if (hdf5_data_object->filespace)
		{
			H5Sclose (hdf5_data_object->filespace);
			hdf5_data_object->filespace = 0;
		}
		if (hdf5_data_object->memspace)
		{
			H5Sclose (hdf5_data_object->memspace);
			hdf5_data_object->memspace = 0;
		}
		H5Fclose (hdf5_data_object->hdf5_file);
		hdf5_data_object->hdf5_file = 0;
		hdf5_data_object->dataset_index = 0;

		/* now rename this closed file */
		digital_rf_close_hdf5_file(hdf5_data_object);

	}

	hdf5_data_object->present_seq++; /* indicates the creation of a new file */

	strcpy(subdir_with_trailing_slash, subdir);
	strcat(subdir_with_trailing_slash, "/"); /* allows us to compare subdir with sub_directory */

	/* create new directory if needed */
	if (hdf5_data_object->sub_directory == NULL || digital_rf_check_hdf5_directory(subdir)
			|| strcmp(hdf5_data_object->sub_directory, subdir_with_trailing_slash))
	{
		if (digital_rf_create_new_directory(hdf5_data_object, subdir))
			return(-1);
	}


	strcpy(fullname, hdf5_data_object->directory); /* previous check ensures these three commands succeed */
	strcat(fullname, hdf5_data_object->sub_directory);
	strcpy(hdf5_data_object->basename, basename);
	strcat(fullname, hdf5_data_object->basename);

	/* check if file exists with the finished name, fail if it does */
	strcpy(finished_fullname, hdf5_data_object->directory);
	strcat(finished_fullname, hdf5_data_object->sub_directory);
	strcat(finished_fullname, strstr(hdf5_data_object->basename, "rf"));
	if( access( finished_fullname, F_OK ) != -1 )
	{
		snprintf(error_str, BIG_HDF5_STR, "The following Hdf5 file already exists: %s\n", finished_fullname);
		fprintf(stderr, "%s", error_str);
		return(-1);
	}

	/* Create a new file. If file exists will fail. */
	hdf5_data_object->hdf5_file = H5Fcreate (fullname, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
	if (hdf5_data_object->hdf5_file < 0)
	{
		snprintf(error_str, BIG_HDF5_STR, "The following Hdf5 file could not be created, or already exists: %s\n", fullname);
		fprintf(stderr, "%s", error_str);
		hdf5_data_object->has_failure = 1;
		hdf5_data_object->hdf5_file = 0;
		return(-1);
	}

	/* now we add the dataset to create */
	if (hdf5_data_object->needs_chunking)
		num_rows = samples_to_write;
	else
		num_rows = max_samples_this_file;
	dims[0] = num_rows;
	/* Create the data space with set dimensions. */
	if (hdf5_data_object->dataspace)
			H5Sclose (hdf5_data_object->dataspace);
	hdf5_data_object->dataspace = H5Screate_simple (hdf5_data_object->rank, dims, maxdims);
	/* create the dataset */
	if (hdf5_data_object->dataset)
			H5Dclose (hdf5_data_object->dataset);

	if (hdf5_data_object->is_complex == 0)
		hdf5_data_object->dataset = H5Dcreate2 (hdf5_data_object->hdf5_file, datasetname,
												hdf5_data_object->dtype_id,
												hdf5_data_object->dataspace, H5P_DEFAULT,
												hdf5_data_object->dataset_prop, H5P_DEFAULT);
	else
		hdf5_data_object->dataset = H5Dcreate2 (hdf5_data_object->hdf5_file, datasetname,
												hdf5_data_object->complex_dtype_id,
												hdf5_data_object->dataspace, H5P_DEFAULT,
												hdf5_data_object->dataset_prop, H5P_DEFAULT);

	if (hdf5_data_object->needs_chunking)
		hdf5_data_object->dataset_index = 0;        /* next write will be to first row */
	else
		hdf5_data_object->dataset_index = max_samples_this_file - samples_left;

	hdf5_data_object->dataset_avail = num_rows; /* size available to next write */

	/* last we add metadata */
	digital_rf_write_metadata(hdf5_data_object);
	return(0);
}


int digital_rf_close_hdf5_file(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_close_hdf5_file closes the present hdf5 file by removing the tmp. part of the file name
 *
 * Inputs:
 * 	Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 *
 * 	Renames last file written to by removing tmp. at beginning of basename, if it exists.  Returns success if it does not exist.
 *
 * 	Returns 0 if success, -1 if failure
 *
 */
{
	/* local variables */
	char fullname[BIG_HDF5_STR] = "";
	char new_fullfilename[BIG_HDF5_STR] = "";

	if (hdf5_data_object->directory == NULL ||
			hdf5_data_object->sub_directory == NULL)
		return(0); /* nothing to close */

	strcpy(fullname, hdf5_data_object->directory);
	strcat(fullname, hdf5_data_object->sub_directory);
	strcat(fullname, hdf5_data_object->basename);

	strcpy(new_fullfilename, hdf5_data_object->directory);
	strcat(new_fullfilename, hdf5_data_object->sub_directory);
	strcat(new_fullfilename, strstr(hdf5_data_object->basename, "rf"));

	if( access( fullname, F_OK ) != -1 )
		/* remove file if error has occurred, rename otherwise */
		if (hdf5_data_object->has_failure)
			return(remove(fullname));
		else
			return(rename(fullname, new_fullfilename));
	else
		return(0); /* file already closed */
}


int digital_rf_create_new_directory(Digital_rf_write_object *hdf5_data_object, char * subdir)
/* digital_rf_create_new_directory creates a new subdirectory to store Hdf5 files in
 *
 * 	Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 * 	subdir - base subdir to create.
 *
 * Affects: sets hdf5_data_object->sub_directory.
 *
 * Returns 0 if success, -1 if failure. Fails if this directory can't be written.
 *
 */
{
	/* local variables */
	char full_directory[BIG_HDF5_STR] = "";
	int result;

	strcpy(full_directory, hdf5_data_object->directory); /* directory ends with "/" */
	strcat(full_directory, subdir);

	#if defined(_WIN32)
		result = _mkdir(full_directory);
	#else
		result = mkdir(full_directory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	#endif
	if (result && errno != EEXIST)
	{
		fprintf(stderr, "Unable to create directory %s\n", full_directory);
		hdf5_data_object->has_failure = 1;
		return(-1);
	}

	if (hdf5_data_object->sub_directory != NULL)
		free(hdf5_data_object->sub_directory);

	if ((hdf5_data_object->sub_directory = (char *)malloc(sizeof(char) * (strlen(subdir)+2)))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}
	strcpy(hdf5_data_object->sub_directory, subdir);
	strcat(hdf5_data_object->sub_directory, "/"); /* will always end with "/" */
	return(0);
}



int digital_rf_set_fill_value(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_set_fill_value sets the fill value property in hdf5_data_object->dataset_prop according to dtype_id.
 *
 * Returns 0 if success, -1 if failure
 */
{
	/* local variables */

	/* integer minimum values, first value is matching byte order, second is reversed byte order */

	int minUnsignedInt = 0;

	/* char */
	char minChar = -128;
	struct complex_char_fill_type { char r, i; };
	struct complex_char_fill_type complex_char_fill = { minChar, minChar };
	struct complex_uchar_fill_type { unsigned char r; unsigned char i; };
	struct complex_uchar_fill_type complex_uchar_fill = { 0, 0 };

	/* short */
	short minShort[2] = {INT16_MIN, 128};
	struct complex_short_fill_type { short r, i; };
	struct complex_short_fill_type complex_short_fill[2] = { {minShort[0], minShort[0]}, {minShort[1], minShort[1]} };
	struct complex_ushort_fill_type { unsigned short r, i; };
	struct complex_ushort_fill_type complex_ushort_fill = { 0, 0 };

	/* int */
	int minInt[2] = {INT32_MIN, 128};
	struct complex_int_fill_type { int r, i; };
	struct complex_int_fill_type complex_int_fill[2] = { {minInt[0], minInt[0]}, {minInt[1], minInt[1]} };
	struct complex_uint_fill_type { unsigned int r, i; };
	struct complex_uint_fill_type complex_uint_fill = { 0, 0 };

	/* int64 */
	int64_t minLLong[2] = {INT64_MIN, 128};
	struct complex_long_fill_type { int64_t r, i; };
	struct complex_long_fill_type complex_long_fill[2] = { {minLLong[0], minLLong[0]}, {minLLong[1], minLLong[1]} };
	struct complex_ulong_fill_type { uint64_t r, i; };
	struct complex_ulong_fill_type complex_ulong_fill = { 0, 0 };

	// float
	float float_fill = (float) HUGE_VAL * 0;  /* NAN but works in C89 (MSVC++ 2008)*/
	struct complex_float_fill_type { float r, i; };
	struct complex_float_fill_type complex_float_fill = { float_fill, float_fill };

	// double
	double double_fill = HUGE_VAL * 0;  /* NAN but works in C89 (MSVC++ 2008)*/
	struct complex_double_fill_type { double r, i; };
	struct complex_double_fill_type complex_double_fill = { double_fill, double_fill };

	char error_str[SMALL_HDF5_STR] = "";
	H5T_class_t classType;
	H5T_sign_t signType;
	int numBytes;
	int endian_flip, write_endian;

	/* found out if the output endian differs from the host endian */
	endian_flip = 0;
	write_endian = H5Tget_order(hdf5_data_object->dtype_id);
	if (digital_rf_is_little_endian() && (write_endian == H5T_ORDER_BE))
		endian_flip = 1;
	else if ((!digital_rf_is_little_endian()) && (write_endian == H5T_ORDER_LE))
		endian_flip = 1;

	/* find out whether we are using integers or floats */
	classType = H5Tget_class( hdf5_data_object->dtype_id );
	/* find out if integer is signed, and the number of bytes */
	signType = H5Tget_sign(hdf5_data_object->dtype_id);
	numBytes = (int)H5Tget_size(hdf5_data_object->dtype_id);

	if (classType == H5T_FLOAT && hdf5_data_object->is_complex == 0)
	{
		H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->dtype_id, &double_fill);
	}
	else if (classType == H5T_FLOAT && hdf5_data_object->is_complex != 0 && numBytes == 4)
	{
		H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id, &complex_float_fill);
	}
	else if (classType == H5T_FLOAT && hdf5_data_object->is_complex != 0 && numBytes == 8)
	{
		H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id, &complex_double_fill);
	}
	else if (classType == H5T_INTEGER)
	{
		if (hdf5_data_object->is_complex == 0)
		{
			if (signType == H5T_SGN_NONE)
				H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->dtype_id, &minUnsignedInt);
			else
			{
				switch(numBytes)
				{
					case 1:
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->dtype_id, &minChar);
						break;
					case 2:
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->dtype_id, &minShort[endian_flip]);
						break;
					case 4:
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->dtype_id, &minInt[endian_flip]);
						break;
					case 8:
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->dtype_id, &minLLong[endian_flip]);
						break;
					default:
						snprintf(error_str, SMALL_HDF5_STR, "Integer type has unexpected number of bytes: %i\n", numBytes);
						fprintf(stderr, "%s", error_str);
						return(-1);
				}
			}
		}
		else /* complex data */
		{
			switch(numBytes)
			{
				case 1:
					if (signType == H5T_SGN_NONE)
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id, &complex_uchar_fill);
					else
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id, &complex_char_fill);
					break;
				case 2:
					if (signType == H5T_SGN_NONE)
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id,
								&complex_ushort_fill);
					else
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id,
								&complex_short_fill[endian_flip]);
					break;
				case 4:
					if (signType == H5T_SGN_NONE)
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id,
								&complex_uint_fill);
					else
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id,
								&complex_int_fill[endian_flip]);
					break;
				case 8:
					if (signType == H5T_SGN_NONE)
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id,
								&complex_ulong_fill);
					else
						H5Pset_fill_value(hdf5_data_object->dataset_prop, hdf5_data_object->complex_dtype_id,
								&complex_long_fill[endian_flip]);
					break;
				default:
					snprintf(error_str, SMALL_HDF5_STR, "Integer type has unexpected number of bytes: %i\n", numBytes);
					fprintf(stderr, "%s", error_str);
					return(-1);
			}
		}

	}
	else
	{
		fprintf(stderr, "Hdf5 datatype passed into dtype_id is neither integer nor float - aborting\n");
		return(-1);
	}
	return(0);

}


void digital_rf_write_metadata(Digital_rf_write_object *hdf5_data_object)
/* digital_rf_write_metadata writes the following metadata to the open file:
 *  Fields that match those in the drf_properties.h5 file
 * 	1. uint64_t H5Tget_class (result of H5Tget_class(hdf5_data_object->hdf5_data_object)
 * 	2. uint64_t H5Tget_size (result of H5Tget_size(hdf5_data_object->hdf5_data_object)
 * 	3. uint64_t H5Tget_order (result of H5Tget_order(hdf5_data_object->hdf5_data_object)
 * 	4. uint64_t H5Tget_precision (result of H5Tget_precision(hdf5_data_object->hdf5_data_object)
 * 	5. uint64_t H5Tget_offset (result of H5Tget_offset(hdf5_data_object->hdf5_data_object)
 * 	6. uint64_t subdir_cadence_secs,
 * 	7. uint64_t file_cadence_millisecs
 * 	8. uint64_t sample_rate_numerator
 * 	9. uint64_t sample_rate_denominator
 * 	10. int is_complex
 * 	11. int num_subchannels
 * 	12. int is_continuous
 * 	13. char[] epoch
 * 	14. char[] digital_rf_time_description
 * 	15. char[] digital_rf_version
 *
 * 	Not found in drf_properties.h5 because possibly unique to each file
 * 	1. int sequence_num (incremented for each file)
 * 	2. uint64_t init_utc_timestamp (changes at each restart of the recorder - needed if leapseconds correction applied)
 * 	3. uint64_t computer_time (time of initial file creation)
 * 	4. char[] uuid_str - set independently at each restart of the recorder
 *
 * 	Must match drf_properties.h5, except adds sequence_num
 *
 * Inputs:
 * 	Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 *
 */
{
	/* local variables */
	hid_t       attribute_id, dataspace_id;  /* identifiers */
	hid_t       str_dataspace, str_type, str_attribute;
	hsize_t     dims = 1;
	time_t      computer_time;
	int64_t    u_computer_time;
	uint64_t result;

	dataspace_id = H5Screate_simple(1, &dims, NULL);

	/* sequence_num */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "sequence_num", H5T_NATIVE_INT, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->present_seq));
	H5Aclose(attribute_id);

	/* H5Tget_class */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "H5Tget_class", H5T_NATIVE_ULLONG, dataspace_id,
							   H5P_DEFAULT, H5P_DEFAULT);
	result = (uint64_t)H5Tget_class(hdf5_data_object->dtype_id);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
	H5Aclose(attribute_id);

	/* H5Tget_size */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "H5Tget_size", H5T_NATIVE_ULLONG, dataspace_id,
							   H5P_DEFAULT, H5P_DEFAULT);
	result = (uint64_t)H5Tget_size(hdf5_data_object->dtype_id);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
	H5Aclose(attribute_id);

	/* H5Tget_order */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "H5Tget_order", H5T_NATIVE_ULLONG, dataspace_id,
							   H5P_DEFAULT, H5P_DEFAULT);
	result = (uint64_t)H5Tget_order(hdf5_data_object->dtype_id);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
	H5Aclose(attribute_id);

	/* H5Tget_precision */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "H5Tget_precision", H5T_NATIVE_ULLONG, dataspace_id,
							   H5P_DEFAULT, H5P_DEFAULT);
	result = (uint64_t)H5Tget_precision(hdf5_data_object->dtype_id);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
	H5Aclose(attribute_id);

	/* H5Tget_offset */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "H5Tget_offset", H5T_NATIVE_ULLONG, dataspace_id,
							   H5P_DEFAULT, H5P_DEFAULT);
	result = (uint64_t)H5Tget_offset(hdf5_data_object->dtype_id);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
	H5Aclose(attribute_id);

	/* num_subchannels */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "num_subchannels", H5T_NATIVE_INT, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->num_subchannels));
	H5Aclose(attribute_id);

	/* is_complex */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "is_complex", H5T_NATIVE_INT, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->is_complex));
	H5Aclose(attribute_id);

	/* subdir_cadence_secs */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "subdir_cadence_secs", H5T_NATIVE_ULLONG, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->subdir_cadence_secs));
	H5Aclose(attribute_id);

	/* file_cadence_millisecs */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "file_cadence_millisecs", H5T_NATIVE_ULLONG, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->file_cadence_millisecs));
	H5Aclose(attribute_id);

	/* is_continuous */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "is_continuous", H5T_NATIVE_INT, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->is_continuous));
	H5Aclose(attribute_id);

	/* sample_rate_numerator */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "sample_rate_numerator", H5T_NATIVE_ULLONG, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->sample_rate_numerator));
	H5Aclose(attribute_id);

	/* sample_rate_denominator */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "sample_rate_denominator", H5T_NATIVE_ULLONG, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->sample_rate_denominator));
	H5Aclose(attribute_id);

	/* init_utc_timestamp */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "init_utc_timestamp", H5T_NATIVE_ULLONG, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->init_utc_timestamp));
	H5Aclose(attribute_id);


	/* computer time */
	attribute_id = H5Acreate2 (hdf5_data_object->dataset, "computer_time", H5T_NATIVE_ULLONG, dataspace_id,
								 H5P_DEFAULT, H5P_DEFAULT);
	computer_time = time(NULL);
	u_computer_time = (uint64_t)computer_time;
	H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(u_computer_time));
	H5Aclose(attribute_id);

	/* uuid_str */
	str_dataspace  = H5Screate(H5S_SCALAR);
    str_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(str_type, strlen(hdf5_data_object->uuid_str)+1);
    str_attribute = H5Acreate2(hdf5_data_object->dataset, "uuid_str", str_type, str_dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(str_attribute, str_type, hdf5_data_object->uuid_str);
    H5Aclose(str_attribute);

    /* epoch */
	H5Tset_size(str_type, strlen(DIGITAL_RF_EPOCH)+1);
	str_attribute = H5Acreate2(hdf5_data_object->dataset, "epoch", str_type, str_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(str_attribute, str_type, DIGITAL_RF_EPOCH);
	H5Aclose(str_attribute);

	/* digital_rf_time_description */
	H5Tset_size(str_type, strlen(DIGITAL_RF_TIME_DESCRIPTION)+1);
	str_attribute = H5Acreate2(hdf5_data_object->dataset, "digital_rf_time_description", str_type, str_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(str_attribute, str_type, DIGITAL_RF_TIME_DESCRIPTION);
	H5Aclose(str_attribute);

	/* digital_rf_version */
	H5Tset_size(str_type, strlen(DIGITAL_RF_VERSION)+1);
	str_attribute = H5Acreate2(hdf5_data_object->dataset, "digital_rf_version", str_type, str_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(str_attribute, str_type, DIGITAL_RF_VERSION);
	H5Aclose(str_attribute);

    /* free resources used */
	H5Tclose(str_type);
    H5Sclose(dataspace_id);
    H5Sclose(str_dataspace);


}


uint64_t * digital_rf_create_rf_data_index(Digital_rf_write_object *hdf5_data_object, uint64_t samples_written, uint64_t samples_left,
		uint64_t max_samples_this_file, uint64_t * global_index_arr, uint64_t * data_index_arr, uint64_t index_len, uint64_t vector_len,
		uint64_t next_global_sample, int * rows_to_write, uint64_t * samples_to_write, int file_exists)
/* digital_rf_create_rf_data_index returns a malloced block of rf_data_index index data to write into the existing Hdf5 file
 * also sets the number of rows to be written.  Number of rows to be written may be zero, in which case returns NULL and
 * rows_to_write set to 0.
 * With Digital RF 2.0, now also calculates samples_to_write whether or not rows_to_write zero or non-zero
 *
 *  Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 * 	uint64_t samples_written - the number of samples written to previous files during this particular user write call
 * 	uint64_t samples_left - the maximum continuous samples that can be written to this file
 *	uint64_t max_samples_this_file - the maximum total number of samples the can be in this file
 * 	uint64_t * global_index_arr - an array of global indices into the samples being written.  The global
 * 		index is the total number of sample periods since data taking began, including gaps.  Error
 * 		is raised if any value is before its expected value (meaning repeated data).
 * 	uint64_t * data_index_arr - an array of len = len(global_index_arr), where the indices are related to which
 * 		sample in the vector being passed in by the user, so that the first value is always 0,
 * 		or error raised.  Values must be increasing, and cannot be equal or greater than index_len or error raised.
 * 	uint_64 index_len - the len of data_index_arr.  Must be greater than 0.
 * 	uint64_t vector_length - the total number of samples to write from vector.
 * 	uint64_t next_global_sample - first global sample to write in this file.
 * 	int * rows_to_write - this int will be set to the number of rows malloced in returned data
 * 	uint64_t * samples_to_write - total number of samples to write to this file given the file boundaries
 * 	int file_exists - 1 if file being written to already exists, 0 if not
 *
 * 	Returns a malloced uint64_t array of size (rows_to_write, 2) with columns
 * 		dataset for this particular file, or -1 if an error detected. Used to allocate or increase size of /rf_data_index.
 * 		Returned array adds hdf5_data_object->global_start_sample to all global indices so that all are zero at 0UT 1970-01-01
 * 		May be NULL if rows_to_write is 0. Caller is responsible for freeing this.
 * 		Returns NULL and rows_to_write = -1 and error printed to stderr if error detected
 */
{
	int i; /* block loop variable */
	uint64_t this_index, this_sample;
	uint64_t last_global_sample;  /* last possible global sample that be written to this file */
	int64_t top_index = -1, bottom_index = -1; /* temp indecies used to calc samples_to write */
	int row_count = 0;
	uint64_t prev_index = 0; /* make sure indices are increasing at least as much as global_index_arr */
	uint64_t prev_sample = 0;
	char error_str[BIG_HDF5_STR] = "";
	uint64_t * ret_arr; /* will hold malloced data to be returned */
	int rows_written = 0; /* keeps tracks of rows written */

	if (index_len < 1)
	{
		snprintf(error_str, BIG_HDF5_STR, "index_len (%" PRIu64 ") must be greater than 0\n",
				index_len);
		fprintf(stderr, "%s", error_str);
		*rows_to_write = -1;
		return(NULL);
	}

	/* init samples_to_write */
	*samples_to_write = 0;
	last_global_sample = next_global_sample + samples_left;

	/* this first pass is just to count the number of rows needed before the malloc, and to valid data is reasonable */
	if (samples_written == 0 && global_index_arr[0] < hdf5_data_object->global_index)
	{
		snprintf(error_str, BIG_HDF5_STR, "global_index_arr passed in %" PRIu64 " before minimum value of %" PRIu64 "\n",
				global_index_arr[0], hdf5_data_object->global_index);
		fprintf(stderr, "%s", error_str);
		*rows_to_write = -1;
		return(NULL);
	}

	for (i=0; i<index_len; i++)
	{
		this_index = data_index_arr[i];
		this_sample = global_index_arr[i];

		/* more input data sanity checks */
		if (i>0 && prev_index >= this_index)
		{
			snprintf(error_str, BIG_HDF5_STR, "indices in data_index_arr out of order - index %i and %i\n", i-1,i);
			fprintf(stderr, "%s", error_str);
			*rows_to_write = -1;
			return(NULL);
		}
		if (i>0 && ((this_index - prev_index) > (global_index_arr[i] - global_index_arr[i-1])))
		{
			snprintf(error_str, BIG_HDF5_STR, "error - indices advancing faster than global index at index %i, illegal\n", i);
			fprintf(stderr, "%s", error_str);
			*rows_to_write = -1;
			return(NULL);
		}

		/* see if these indices are needed for this file */
		if (i > 0)
		{
			if (next_global_sample < prev_sample + (this_index - prev_index) && last_global_sample >= this_sample)
				row_count++;
		}
		else if (!file_exists || hdf5_data_object->needs_chunking)
			row_count++;

		/* determine range of data [bottom_index, top_index] that will be written to this file */
		/* set bottom_index if unset and the global sample for this block is past the next sample to be written */
		if (this_sample > next_global_sample && bottom_index == -1)
		{
			/* calculate bottom index */
			if (i>0)
			{
				if (prev_sample + (this_index-prev_index) < next_global_sample)
				{
					/* next_global_sample falls in a gap */
					bottom_index = this_index;
				}
				else
					bottom_index = prev_index + (next_global_sample - prev_sample);
			}
			else
				bottom_index = 0;
		}
		/* set top_index if unset and the global sample for this block is past the last sample to be in the file */
		if (this_sample > last_global_sample && top_index == -1)
		{
			/* calculate top index */
			if (last_global_sample > prev_sample + (this_index - prev_index))
			{
				/* last sample falls in a gap */
				top_index = this_index;
			}
			else
				top_index = prev_index + (last_global_sample - prev_sample);
		}

		prev_index = this_index;
		prev_sample = this_sample;
	}

	/* now handle write data beyond last gap */
	this_index = vector_len;
	this_sample = prev_sample + (this_index-prev_index);

	/* set bottom_index according to last data block if unset */
	if (bottom_index == -1)
		bottom_index = prev_index + (next_global_sample-prev_sample);

	/* set top_index according to last data block if unset */
	if (top_index == -1)
	{
		if (last_global_sample < this_sample)
			/* file ends before last sample in data block */
			top_index = prev_index + (last_global_sample - prev_sample);
		else
			/* write includes last sample in data block */
			/* works out to top_index == vector_len */
			top_index = prev_index + (this_sample - prev_sample);
	}
	/* now we know how many samples to write from the data to this file */
	*samples_to_write = top_index - bottom_index;

	/* if no indices need to be malloced, return now */
	if (row_count == 0)
	{
		*rows_to_write = 0;
		return(NULL);
	}

	/* now that we know how many rows to malloc, malloc them */
	/* allocate overall object */
	if ((ret_arr = (uint64_t *)malloc(sizeof(uint64_t)*row_count*2))==0)
	{
		fprintf(stderr, "malloc failure - unrecoverable\n");
		exit(-1);
	}

	/* next pass is to fill out ret_arr */
	/* ret_arr is [global_sampleN, data_indexN, ...] for blocks in file */
	for (i=0; i<index_len; i++)
	{
		this_index = data_index_arr[i];
		this_sample = global_index_arr[i];

		/* only thing we do on the first pass is add to ret_arr for start of
		 * data if it's a new file or the data is written in chunked format */
		if (i == 0 && (!file_exists || hdf5_data_object->needs_chunking))
		{
			ret_arr[0] = next_global_sample + hdf5_data_object->global_start_sample;
			/* handle the case of the first sample written where it may not be the first sample in the file */
			/* and we're writing the data in continuous format */
			if (hdf5_data_object->is_continuous && !hdf5_data_object->needs_chunking)
				ret_arr[0] -= max_samples_this_file - samples_left;
			    /* since ret_arr is referenced to first global sample, may actually be negative */
			/* ret_arr[1] will have hdf5_data_object->dataset_index added later
			 * upon expansion to account for existing data in file */
			ret_arr[1] = 0;
			rows_written++;
		}
		/* on subsequent passes, add the block to ret_arr when it falls in the file */
		if (i > 0)
		{
			if (next_global_sample < prev_sample + (this_index - prev_index) && last_global_sample >= this_sample)
			{
				ret_arr[2*rows_written] = this_sample + hdf5_data_object->global_start_sample;
				/* data index for data in file is different from data index of the data in the write call */
				/* by the number of samples previously written from the data (to previous files) */
				ret_arr[2*rows_written + 1] = this_index - samples_written;
				rows_written++;
			}
		}

		prev_index = this_index;
		prev_sample = this_sample;
	}
	assert(rows_written==row_count);  /* or else there's a bug in my logic */
	*rows_to_write = rows_written;

	return(ret_arr);

}


int digital_rf_write_rf_data_index(Digital_rf_write_object * hdf5_data_object, uint64_t * rf_data_index_arr, int block_index_len)
/* digital_rf_write_rf_data_index writes rf_data_index to the open Hdf5 file
 *
 * Inputs:
 *  Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 *  uint64_t * rf_data_index_arr - uint64_t array of size (block_index_len * 2) to write - all values are already set
 *
 *  Returns 0 if success, non-zero if error
 */
{
	int i;
	/* variables for /rf_data_index */
	char index_datasetname[] = "rf_data_index";
	hsize_t  index_dims[2]  = {0, 2};
	hsize_t  dimsext[2] = {block_index_len, 2};
	hsize_t  index_maxdims[2] = {H5S_UNLIMITED, 2};
	hsize_t  offset[2] = {0, 0};
	hid_t    index_dataspace, filespace, memspace;
	herr_t      status;

	/* find out if we need to create a new dataset, or expand and existing one */
	if (hdf5_data_object->index_dataset == 0)
	{
		/* create new dataset */
		index_dims[0] = block_index_len;
		index_dataspace = H5Screate_simple (2, index_dims, index_maxdims);
		hdf5_data_object->index_dataset = H5Dcreate2 (hdf5_data_object->hdf5_file, index_datasetname,
												H5T_NATIVE_ULLONG,
												index_dataspace, H5P_DEFAULT,
												hdf5_data_object->index_prop, H5P_DEFAULT);
		status = H5Dwrite(hdf5_data_object->index_dataset, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
					rf_data_index_arr);
		if (status < 0)
			return(status);

		H5Sclose(index_dataspace);
		hdf5_data_object->next_index_avail = block_index_len;
	}
	else
	{
		/* expand dataset */
		/* adjust data index in rf_data_index_arr to account for samples already in file */
		for (i=0; i<block_index_len; i++)
		{
			/* data index are odd terms in rf_data_index_arr,
			 * dataset_index adds number of samples already in file */
			rf_data_index_arr[2*i + 1] += hdf5_data_object->dataset_index;
		}
		/* write to existing index */
		index_dims[0] = hdf5_data_object->next_index_avail + block_index_len;
		status = H5Dset_extent (hdf5_data_object->index_dataset, index_dims);
		filespace = H5Dget_space (hdf5_data_object->index_dataset);
		offset[0] = hdf5_data_object->next_index_avail;
		status = H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
				dimsext, NULL);
		if (status < 0)
			return(status);
		memspace = H5Screate_simple (2, dimsext, NULL);
		status = H5Dwrite (hdf5_data_object->index_dataset, H5T_NATIVE_ULLONG, memspace, filespace,
		                       H5P_DEFAULT, rf_data_index_arr);
		if (status < 0)
			return(status);

		H5Sclose (memspace);
		H5Sclose (filespace);
		hdf5_data_object->next_index_avail += block_index_len;
	}
	return(0);
}


uint64_t digital_rf_get_global_sample(uint64_t samples_written, uint64_t * global_index_arr, uint64_t * data_index_arr,
		                              uint64_t index_len)
/* digital_rf_get_global_sample calculates the global_sample given samples_written using global_index_arr and data_index_arr
 *
 *  uint64_t samples_written - the number of samples written to previous files during this particular user write call
 * 	uint64_t * global_index_arr - an array of global indices into the samples being written.  The global
 * 		index is the total number of sample periods since data taking began, including gaps.
 * 	uint64_t * data_index_arr - an array of len = len(global_index_arr), where the indices are related to which
 * 		sample in the vector being passed in by the user, so that the first value is always 0
 * 	uint_64 index_len - the len of both global_index_arr and data_index_arr.  Must be greater than 0.
 *
 * 	This method tells the global index when some samples have already been written.  If samples_written == 0,
 * 	returns global_index_arr[0].  Else returns global_index_arr[i] + (samples_written-data_index_arr[i]) where
 * 	i is the highest row where data_index_arr[i] <= samples_written
 *
 */
{
	/* local variables */
	int i;
	uint64_t ret_value;

	ret_value = global_index_arr[0] + (samples_written - data_index_arr[0]);
	for (i=1; i<index_len; i++)
	{
		if (samples_written < data_index_arr[i])
			break;
		ret_value = global_index_arr[i] + (samples_written - data_index_arr[i]);
	}
	return(ret_value);
}

int digital_rf_extend_dataset(Digital_rf_write_object * hdf5_data_object, uint64_t samples_to_write)
/* digital_rf_extend_dataset is a method to extend the dataset of on opened Hdf5 file
 *
 * Digital_rf_write_object *hdf5_data_object - the Digital_rf_write_object created by digital_rf_create_write_hdf5
 * uint64_t - the number of samples to expand existing dataset
 *
 */
{
	/* local variables */
	herr_t   status;
	hsize_t  dims[2]  = {0, hdf5_data_object->num_subchannels};

	dims[0] = hdf5_data_object->dataset_index + samples_to_write;
	status = H5Dset_extent (hdf5_data_object->dataset, dims);
	return((int)status);
}

int digital_rf_handle_metadata(Digital_rf_write_object * hdf5_data_object)
/* digital_rf_handle_metadata is a method that either creates or verifies consistency with <channel>/drf_properties.h5
 *
 * drf_properties.h5 is a simple hdf5 file where the root group has the following attributes:
 * 	1. uint64_t H5Tget_class (result of H5Tget_class(hdf5_data_object->hdf5_data_object)
 * 	2. uint64_t H5Tget_size (result of H5Tget_size(hdf5_data_object->hdf5_data_object)
 * 	3. uint64_t H5Tget_order (result of H5Tget_order(hdf5_data_object->hdf5_data_object)
 * 	4. uint64_t H5Tget_precision (result of H5Tget_precision(hdf5_data_object->hdf5_data_object)
 * 	5. uint64_t H5Tget_offset (result of H5Tget_offset(hdf5_data_object->hdf5_data_object)
 * 	6. uint64_t subdir_cadence_secs,
 * 	7. uint64_t file_cadence_millisecs
 * 	8. uint64_t sample_rate_numerator
 * 	9. uint64_t sample_rate_denominator
 * 	10. int is_complex
 * 	11. int num_subchannels
 * 	12. int is_continuous
 * 	13. char[] epoch
 * 	14. char[] digital_rf_time_description
 * 	15. char[] digital_rf_version
 *
 * 	If drf_properties.h5 does not exist, creates it. Returns 0 if success, 1 if not.
 *
 * 	If drf_properties.h5 does exist, verifies that first 11 attributes match that in init.  If so, returns
 * 	0. If not, returns -1
 *
 */
{
	/* local variables */
	char metadata_file[BIG_HDF5_STR] = "";
	char error_str[BIG_HDF5_STR] = "";
	int metadata_exists = 0;
	hid_t hdf5_file, str_type, str_attribute;
	hid_t attribute_id, dataspace_id;  /* identifiers */
	hsize_t  dims = 1;
	uint64_t result;
	int int_result;

	/* find out if drf_properties.h5 exists */
	strcpy(metadata_file, hdf5_data_object->directory);
	strcat(metadata_file, "drf_properties.h5");
	if( access( metadata_file, R_OK ) != -1 )
	    metadata_exists = 1;
	else
		metadata_exists = 0;

	if (metadata_exists == 0)
	{
		dataspace_id = H5Screate_simple(1, &dims, NULL);
		hdf5_file = H5Fcreate (metadata_file, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
		if (hdf5_file < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The following metadata file could not be created: %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(1);
		}

		/* H5Tget_class */
		attribute_id = H5Acreate2 (hdf5_file, "H5Tget_class", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		result = (uint64_t)H5Tget_class(hdf5_data_object->dtype_id);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
		H5Aclose(attribute_id);

		/* H5Tget_size */
		attribute_id = H5Acreate2 (hdf5_file, "H5Tget_size", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		result = (uint64_t)H5Tget_size(hdf5_data_object->dtype_id);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
		H5Aclose(attribute_id);

		/* H5Tget_order */
		attribute_id = H5Acreate2 (hdf5_file, "H5Tget_order", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		result = (uint64_t)H5Tget_order(hdf5_data_object->dtype_id);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
		H5Aclose(attribute_id);

		/* H5Tget_precision */
		attribute_id = H5Acreate2 (hdf5_file, "H5Tget_precision", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		result = (uint64_t)H5Tget_precision(hdf5_data_object->dtype_id);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
		H5Aclose(attribute_id);

		/* H5Tget_offset */
		attribute_id = H5Acreate2 (hdf5_file, "H5Tget_offset", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		result = (uint64_t)H5Tget_offset(hdf5_data_object->dtype_id);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(result));
		H5Aclose(attribute_id);

		/* subdir_cadence_secs */
		attribute_id = H5Acreate2 (hdf5_file, "subdir_cadence_secs", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->subdir_cadence_secs));
		H5Aclose(attribute_id);

		/* file_cadence_millisecs */
		attribute_id = H5Acreate2 (hdf5_file, "file_cadence_millisecs", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->file_cadence_millisecs));
		H5Aclose(attribute_id);

		/* sample_rate_numerator */
		attribute_id = H5Acreate2 (hdf5_file, "sample_rate_numerator", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->sample_rate_numerator));
		H5Aclose(attribute_id);

		/* sample_rate_denominator */
		attribute_id = H5Acreate2 (hdf5_file, "sample_rate_denominator", H5T_NATIVE_ULLONG, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_ULLONG, &(hdf5_data_object->sample_rate_denominator));
		H5Aclose(attribute_id);

		/* is_complex */
		attribute_id = H5Acreate2 (hdf5_file, "is_complex", H5T_NATIVE_INT, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->is_complex));
		H5Aclose(attribute_id);

		/* num_subchannels */
		attribute_id = H5Acreate2 (hdf5_file, "num_subchannels", H5T_NATIVE_INT, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->num_subchannels));
		H5Aclose(attribute_id);

		/* is_continuous */
		attribute_id = H5Acreate2 (hdf5_file, "is_continuous", H5T_NATIVE_INT, dataspace_id,
								   H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attribute_id, H5T_NATIVE_INT, &(hdf5_data_object->is_continuous));
		H5Aclose(attribute_id);

		/* epoch */
		str_type = H5Tcopy(H5T_C_S1);
		H5Tset_size(str_type, strlen(DIGITAL_RF_EPOCH)+1);
		str_attribute = H5Acreate2(hdf5_file, "epoch", str_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(str_attribute, str_type, DIGITAL_RF_EPOCH);
		H5Aclose(str_attribute);

		/* digital_rf_time_description */
		H5Tset_size(str_type, strlen(DIGITAL_RF_TIME_DESCRIPTION)+1);
		str_attribute = H5Acreate2(hdf5_file, "digital_rf_time_description", str_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(str_attribute, str_type, DIGITAL_RF_TIME_DESCRIPTION);
		H5Aclose(str_attribute);

		/* digital_rf_version */
		H5Tset_size(str_type, strlen(DIGITAL_RF_VERSION)+1);
		str_attribute = H5Acreate2(hdf5_file, "digital_rf_version", str_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(str_attribute, str_type, DIGITAL_RF_VERSION);
		H5Aclose(str_attribute);

		/* free resources used */
		H5Tclose(str_type);
		H5Sclose(dataspace_id);
		H5Fclose(hdf5_file);
	}
	else
	{
		hdf5_file = H5Fopen(metadata_file, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (hdf5_file < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The following metadata file could not be opened: %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}

		/* read all attributes and compare */

		/* H5Tget_class attribute */
		attribute_id = H5Aopen(hdf5_file, "H5Tget_class", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The H5Tget_class attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != (uint64_t)H5Tget_class(hdf5_data_object->dtype_id))
		{
			fprintf(stderr, "Mismatching datatype found using H5Tget_class\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* H5Tget_size attribute */
		attribute_id = H5Aopen(hdf5_file, "H5Tget_size", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The H5Tget_size attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != (uint64_t)H5Tget_size(hdf5_data_object->dtype_id))
		{
			fprintf(stderr, "Mismatching datatype found using H5Tget_size\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* H5Tget_order attribute */
		attribute_id = H5Aopen(hdf5_file, "H5Tget_order", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The H5Tget_order attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != (uint64_t)H5Tget_order(hdf5_data_object->dtype_id))
		{
			fprintf(stderr, "Mismatching datatype found using H5Tget_order\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* H5Tget_precision attribute */
		attribute_id = H5Aopen(hdf5_file, "H5Tget_precision", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The H5Tget_precision attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != (uint64_t)H5Tget_precision(hdf5_data_object->dtype_id))
		{
			fprintf(stderr, "Mismatching datatype found using H5Tget_precision\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* H5Tget_offset attribute */
		attribute_id = H5Aopen(hdf5_file, "H5Tget_offset", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The H5Tget_offset attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != (uint64_t)H5Tget_offset(hdf5_data_object->dtype_id))
		{
			fprintf(stderr, "Mismatching datatype found using H5Tget_offset\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* subdir_cadence_secs attribute */
		attribute_id = H5Aopen(hdf5_file, "subdir_cadence_secs", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The subdir_cadence_secs attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != hdf5_data_object->subdir_cadence_secs)
		{
			fprintf(stderr, "Mismatching subdir_cadence_secs found\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* file_cadence_millisecs attribute */
		attribute_id = H5Aopen(hdf5_file, "file_cadence_millisecs", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The file_cadence_millisecs attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != hdf5_data_object->file_cadence_millisecs)
		{
			fprintf(stderr, "Mismatching file_cadence_millisecs found\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* sample_rate_numerator attribute */
		attribute_id = H5Aopen(hdf5_file, "sample_rate_numerator", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The sample_rate_numerator attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != hdf5_data_object->sample_rate_numerator)
		{
			fprintf(stderr, "Mismatching sample_rate_numerator found\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* sample_rate_denominator attribute */
		attribute_id = H5Aopen(hdf5_file, "sample_rate_denominator", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The sample_rate_denominator attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_ULLONG, &result);
		if (result != hdf5_data_object->sample_rate_denominator)
		{
			fprintf(stderr, "Mismatching sample_rate_denominator found\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* is_complex attribute */
		attribute_id = H5Aopen(hdf5_file, "is_complex", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The is_complex attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_INT, &int_result);
		if (int_result != hdf5_data_object->is_complex)
		{
			fprintf(stderr, "Mismatching is_complex found\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* num_subchannels attribute */
		attribute_id = H5Aopen(hdf5_file, "num_subchannels", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The num_subchannels attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_INT, &int_result);
		if (int_result != hdf5_data_object->num_subchannels)
		{
			fprintf(stderr, "Mismatching num_subchannels found\n");
			return(-1);
		}
		H5Aclose(attribute_id);

		/* is_continuous attribute */
		attribute_id = H5Aopen(hdf5_file, "is_continuous", H5P_DEFAULT);
		if (attribute_id < 0)
		{
			snprintf(error_str, BIG_HDF5_STR, "The is_continuous attribute not found in %s\n", metadata_file);
			fprintf(stderr, "%s", error_str);
			return(-1);
		}
		H5Aread(attribute_id, H5T_NATIVE_INT, &int_result);
		if (int_result != hdf5_data_object->is_continuous)
		{
			fprintf(stderr, "Mismatching is_continuous found\n");
			return(-1);
		}
		H5Aclose(attribute_id);


		H5Fclose(hdf5_file);
	}

	/* no problems found */
	return(0);

}


int digital_rf_is_little_endian(void)
/* digital_rf_is_little_endian returns 1 if local machine little-endian, 0 if big-endian
 *
 */
{
    volatile uint32_t i=0x01234567;
    // return 0 for big endian, 1 for little endian.
    return (*((uint8_t*)(&i))) == 0x67;
}
