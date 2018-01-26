/*
 * Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
 * All rights reserved.
 *
 * Distributed under the terms of the BSD 3-clause license.
 *
 * The full license is in the LICENSE file, distributed with this software.
*/
/*
 * Test driver/examples for rf_hdf5 public methods
 *
 * $Id$
 */

#include "digital_rf.h"

#define VECTOR_LEN 3200

int main (int argc, char *argv[])
{

	/* local variables */
	Digital_rf_write_object * data_object = NULL;
	uint64_t vector_leading_edge_index = 0;
	char * last_file_written;
	int is_continuous, i, result;

	/* datasets to write */
	short data_short[VECTOR_LEN][2];


	/* time variables */
	uint64_t picosecond;
	uint64_t sample_rate_numerator;
	uint64_t sample_rate_denominator;
	uint64_t global_index;

	/* init all datasets */
	for (i=0; i<VECTOR_LEN; i++)
	{
		if (i<VECTOR_LEN/2)
		{
			data_short[i][0] = 2;
			data_short[i][1] = 3;
		}
		else
		{
			data_short[i][0] = 4;
			data_short[i][1] = 6;
		}
	}



	sample_rate_numerator = 1000000;
	sample_rate_denominator = 1;
	global_index = (uint64_t)(1394368230 * (long double)sample_rate_numerator/sample_rate_denominator) + 1; /* should represent 2014-03-09 12:30:30  and 1E6 picoseconds*/


	printf("Test 0 - simple single write to multiple files, no compress, no checksum, 2 secs/subdir, 400 ms/file, - channel 0\n");
	is_continuous = 1;
	result = system("rm -rf /tmp/hdf5 ; mkdir /tmp/hdf5 ; mkdir /tmp/hdf5/junk0");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0", H5T_NATIVE_SHORT, 3600, 10000, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_0", 0, 0, 1, 1, is_continuous, 0);
	if (!data_object)
		exit(-1);
	for (i=0; i<7; i++)
	{
		printf("calling digital_rf_write_hdf5 with global_leading_edge_index %" PRIu64 " and vector_length %" PRIu64 "\n",
				vector_leading_edge_index + i*(VECTOR_LEN), (uint64_t)VECTOR_LEN);
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*(VECTOR_LEN), data_short, VECTOR_LEN);
		if (result)
			exit(-1);
	}
	last_file_written = digital_rf_get_last_file_written(data_object);
	printf("Last file written was %s\n", last_file_written);
	free(last_file_written);
	printf("Last write was at utc timestamp %" PRIu64 "\n", digital_rf_get_last_write_time(data_object));
	digital_rf_close_write_hdf5(data_object);
	printf("done test 0\n");


	printf("All tests completed successfully\n");
	return(0);

}
