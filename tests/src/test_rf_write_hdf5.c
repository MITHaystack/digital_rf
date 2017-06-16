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

#define ARR_SIZE 5

void init_block_indices(uint64_t * global_index_arr, uint64_t * block_index_arr,
		uint64_t first_global_value)
/* init_block_indices inits global_index_arr and block_index_arr to simulate writing 10 continuous
 * samples, then 10 missing samples, starting from first_global_value
 */
{
	int i;
	for (i=0; i<10; i++)
	{
		global_index_arr[i] = i*20 + first_global_value;
		block_index_arr[i] = i*10;
	}
}

void init_invalid_block_indices(uint64_t * global_index_arr, uint64_t * block_index_arr,
		uint64_t first_global_value)
/* init_invalid_block_indices is designed to test error handling by deliberately creating indices that
 * make ahead more quickly than the global index, which should trigger an error.
 */
{
	int i;
	for (i=0; i<10; i++)
	{
		global_index_arr[i] = i*10 + first_global_value;
		block_index_arr[i] = i*10;
		if (i>3)
			block_index_arr[i] = i*11; /* illegal - the block index must move ahead no faster than the global index did! */
	}
}

int main (int argc, char *argv[])
{

	/* local variables */
	Digital_rf_write_object * data_object = NULL;
	uint64_t vector_leading_edge_index = 0;
	char * last_file_written;
	int is_continuous;

	/* datasets to write */
	char data_char[100][2];
	char single_char[100];
	short data_short[100][2];
	int data_int[100][2];
	unsigned int data_uint[100][2];
	int64_t data_int64[100][2];
	int64_t single_int64[100];
	double data_double[100][2];
	char data_char_a[100][ARR_SIZE][2];
	char single_char_a[100][ARR_SIZE];
	int data_int_a[100][ARR_SIZE][2];

	/* data block indices to use */
	uint64_t global_index_arr[10];
	uint64_t block_index_arr[10];
	uint64_t vector_length = 100;
	int i, j, result;

	/* time variables */
	int year, month, day, hour, minute, second;
	uint64_t picosecond;
	uint64_t sample_rate_numerator;
	uint64_t sample_rate_denominator;
	long double sample_rate;
	uint64_t global_index;

	/* init all datasets */
	for (i=0; i<100; i++)
	{
		if (i<50)
		{
			data_char[i][0] = 2;
			data_short[i][0] = 2;
			data_int[i][0] = 2;
			data_uint[i][0] = 2;
			data_int64[i][0] = 2;
			data_double[i][0] = 2.0;
			single_char[i] = 2;
			single_int64[i] = 2.0;
			data_char[i][1] = 3;
			data_short[i][1] = 3;
			data_int[i][1] = 3;
			data_uint[i][1] = 3;
			data_int64[i][1] = 3;
			data_double[i][1] = 3.0;
		}
		else
		{
			data_char[i][0] = 4;
			data_short[i][0] = 4;
			data_int[i][0] = 4;
			data_uint[i][0] = 4;
			data_int64[i][0] = 4;
			data_double[i][0] = 4.0;
			single_char[i] = 4;
			single_int64[i] = 4.0;
			data_char[i][1] = 6;
			data_short[i][1] = 6;
			data_int[i][1] = 6;
			data_uint[i][1] = 6;
			data_int64[i][1] = 6;
			data_double[i][1] = 6.0;
		}
	}

	for (i=0; i<100; i++)
	{
		for (j=0; j<ARR_SIZE; j++)
		{
			if (i<50)
			{
				data_char_a[i][j][0] = 2;
				data_int_a[i][j][0] = 2;
				single_char_a[i][j] = 2;
				data_char_a[i][j][1] = 3 + j;
				data_int_a[i][j][1] = 3 + j;
			}
			else
			{
				data_char_a[i][j][0] = 4;
				data_int_a[i][j][0] = 4;
				single_char_a[i][j] = 4;
				data_char_a[i][j][1] = 6 + j;
				data_int_a[i][j][1] = 6 + j;
			}
		}
	}

	printf("Test of time library\n");
	sample_rate_numerator = (uint64_t)1.0E9;
	sample_rate_denominator = 1;
	sample_rate = ((long double)sample_rate_numerator)/sample_rate_denominator;
	global_index = (uint64_t)(1394368230 * sample_rate_numerator) + 1; /* should represent 2014-03-09 12:30:30  and 1E3 picoseconds*/
	if (digital_rf_get_unix_time(global_index, sample_rate, &year, &month, &day,
			                     &hour, &minute, &second,  &picosecond))
	{
		printf("Test failed at get_unix_time\n");
		exit(-1);
	}
	printf("%04i-%02i-%02i %02i:%02i:%02i pico: %" PRIu64 "\n", year, month, day, hour, minute, second, picosecond);

	/* change sample_rate to 66.67HZ so we can work with small files */
	sample_rate_numerator = 200;
	sample_rate_denominator = 3;
	sample_rate = ((long double)sample_rate_numerator)/sample_rate_denominator;
	global_index = (uint64_t)(1394368230 * sample_rate) + 1; /* should represent 2014-03-09 12:30:30  and 15E9 picoseconds*/


	printf("Test 0 - simple single write to multiple files, no compress, no checksum, 2 secs/subdir, 400 ms/file, - channel 0\n");
	is_continuous = 1;
	system("rm -rf /tmp/hdf5 ; mkdir /tmp/hdf5 ; mkdir /tmp/hdf5/junk0");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0", H5T_NATIVE_INT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_0", 0, 0, 1, 1, is_continuous, 0);
	if (!data_object)
		exit(-1);
	for (i=0; i<7; i++)
	{
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*100, data_int, vector_length);
		if (result)
			exit(-1);
	}
	last_file_written = digital_rf_get_last_file_written(data_object);
	printf("Last file written was %s\n", last_file_written);
	free(last_file_written);
	printf("Last write was at utc timestamp %" PRIu64 "\n", digital_rf_get_last_write_time(data_object));
	digital_rf_close_write_hdf5(data_object);
	printf("done test 0\n");

	printf("Test 0.1 - simple single array write to multiple files, no compress, no checksum - channel 0.1\n");
	system("rm -rf /tmp/hdf5/junk0.1 ; mkdir /tmp/hdf5/junk0.1");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0.1", H5T_NATIVE_INT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_0.1", 0, 0, 1, ARR_SIZE, is_continuous, 0);
	if (!data_object)
		exit(-1);
	for (i=0; i<7; i++)
	{
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*100, data_int_a, vector_length);
		if (result)
			exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 0.1\n");

	printf("Test 1 - use complex 1 byte ints with data gap, no compress, no checksum - channel 1\n");
	is_continuous = 0;
	system("rm -rf /tmp/hdf5/junk1 ; mkdir /tmp/hdf5/junk1");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk1", H5T_NATIVE_CHAR, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_1", 0, 0, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_char, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 1200, data_char, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 1\n");

	printf("Test 1.0 - use array of complex 1 byte ints with data gap, no compress, no checksum - channel 1.0\n");
	system("rm -rf /tmp/hdf5/junk1.0 ; mkdir /tmp/hdf5/junk1.0");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk1.0", H5T_NATIVE_CHAR, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_1.0", 0, 0, 1, ARR_SIZE, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_char_a, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, data_char_a, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 1.0\n");

	printf("Test 1.0.1 - try to write more to same channel\n");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk1.0", H5T_NATIVE_CHAR, 2, 400, global_index + 240, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_1.0", 0, 0, 1, ARR_SIZE, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 240, data_char_a, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 360, data_char_a, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 1.0.1\n");

	printf("Test 1.1 - use single 1 byte ints with no data gap, no compress, no checksum - channel 1.1\n");
	is_continuous = 1;
	system("rm -rf /tmp/hdf5/junk1.1 ; mkdir /tmp/hdf5/junk1.1");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk1.1", H5T_NATIVE_CHAR, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_1.1", 0, 0, 0, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	for (i=0; i<10; i++)
	{
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*100, single_char, vector_length);
		if (result)
			exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 1.1\n");

	printf("Test 1.1.0 - use array of single 1 byte ints with no data gap, no compress, no checksum - channel 1.1.0\n");
	system("rm -rf /tmp/hdf5/junk1.1.0 ; mkdir /tmp/hdf5/junk1.1.0");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk1.1.0", H5T_NATIVE_CHAR, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_1.1", 0, 0, 0, ARR_SIZE, is_continuous, 1);
	if (!data_object)
		exit(-1);
	for (i=0; i<10; i++)
	{
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*100, single_char_a, vector_length);
		if (result)
			exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 1.1\n");

	printf("Test 2 - use 2 byte ints with data gap, level 1 compress, but no checksum - channel 2\n");
	is_continuous = 0;
	system("rm -rf /tmp/hdf5/junk2 ; mkdir /tmp/hdf5/junk2");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk2", H5T_NATIVE_SHORT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_2", 1, 0, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_short, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, data_short, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 2\n");

	printf("Test 3 - use 4 byte ints with data gap, no compress, but with checksum - channel 3\n");
	system("rm -rf /tmp/hdf5/junk3 ; mkdir /tmp/hdf5/junk3");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk3", H5T_NATIVE_INT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_3", 0, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_int, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, data_int, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 3\n");

	printf("Test 4 - use 8 byte ints with data gap, both compress (level 6) and checksum - channel 4\n");
	system("rm -rf /tmp/hdf5/junk4 ; mkdir /tmp/hdf5/junk4");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk4", H5T_NATIVE_LLONG, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_4.1", 6, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_int64, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, data_int64, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 4\n");

	printf("Test 4.1 - use single 8 byte ints with 10 on/10 missing blocks, both compress (level 6) and checksum - channel 4.1\n");
	system("rm -rf /tmp/hdf5/junk4.1 ; mkdir /tmp/hdf5/junk4.1");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk4.1", H5T_NATIVE_LLONG, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_4.1", 6, 1, 0, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	init_block_indices(global_index_arr, block_index_arr, 0);
	result = digital_rf_write_blocks_hdf5(data_object, global_index_arr, block_index_arr, 10, single_int64, vector_length);
	if (result)
		exit(-1);
	init_block_indices(global_index_arr, block_index_arr, 205); /* last write was 200 samples, 5 additional missing samples */
	result = digital_rf_write_blocks_hdf5(data_object, global_index_arr, block_index_arr, 10, single_int64, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 4.1\n");

	printf("Test 4.2 - same as 4.1, except use both write methods intermixed - channel 4.2\n");
	system("rm -rf /tmp/hdf5/junk4.2 ; mkdir /tmp/hdf5/junk4.2");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk4.2", H5T_NATIVE_LLONG, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_4.2", 6, 1, 0, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	init_block_indices(global_index_arr, block_index_arr, 0);
	/* first write uses digital_rf_write_blocks_hdf5 */
	result = digital_rf_write_blocks_hdf5(data_object, global_index_arr, block_index_arr, 10, single_int64, vector_length);
	if (result)
		exit(-1);
	/* for the rest of the data, we write only 10 samples at a time using digital_rf_write_hdf5 */
	for (i=0; i<10; i++)
	{
		result = digital_rf_write_hdf5(data_object, 205 + 20*i, single_int64 + i*10, 10);
		if (result)
			exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 4.2\n");

	printf("Test 5 - use 4 byte unsigned ints with data gap, no compress, no checksum - channel 5\n");
	system("rm -rf /tmp/hdf5/junk5 ; mkdir /tmp/hdf5/junk5");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk5", H5T_NATIVE_UINT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_5", 0, 0, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_uint, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, data_uint, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 5\n");

	printf("Test 5.1 - use 4 byte unsigned ints without data gap, no compress, no checksum - channel 5.1\n");
	is_continuous = 1;
	system("rm -rf /tmp/hdf5/junk5.1 ; mkdir /tmp/hdf5/junk5.1");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk5.1", H5T_NATIVE_UINT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_5.1", 0, 0, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_uint, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 100, data_uint, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 5.1\n");

	printf("Test 5.2 - use 4 byte unsigned ints without data gap, no compress, no checksum - channel 5.2, smaller data writes\n");
	system("rm -rf /tmp/hdf5/junk5.2 ; mkdir /tmp/hdf5/junk5.2");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk5.2", H5T_NATIVE_UINT, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_5.2", 0, 0, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	for (i=0; i<5; i++)
	{
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*15, data_uint, 15);
		if (result)
			exit(-1);
	}
	for (i=0; i<5; i++)
	{
		/* here we create a 3 sample data gap mid file just to be mean */
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 78 + i*15, data_uint, 15);
		if (result)
			exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 5.2\n");

	printf("Test 6 - use doubles with data gap, both compress (level 9) and checksum - channel 6\n");
	is_continuous = 0;
	system("rm -rf /tmp/hdf5/junk6 ; mkdir /tmp/hdf5/junk6");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk6", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_6", 9, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_double, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, data_double, vector_length);
	if (result)
		exit(-1);
	digital_rf_close_write_hdf5(data_object);
	printf("done test 6\n");

	/* from here on we try to do illegal stuff, and make sure error checking catches it */

	printf("Test 7 - try to write backwards and get an error\n");
	system("rm -rf /tmp/hdf5/junk7 ; mkdir /tmp/hdf5/junk7");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk7", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_7", 9, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_double, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 90, data_double, vector_length);
	if (result)
		printf("Got expected error by trying to overwrite data\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 7\n");

	printf("Test 8 - try to write using a NULL pointer and get an error\n");
	system("rm -rf /tmp/hdf5/junk8 ; mkdir /tmp/hdf5/junk8");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk8", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_8", 9, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_double, vector_length);
	if (result)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + 120, NULL, vector_length);
	if (result)
		printf("Got expected error by trying to write using NULL pointer\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);
	printf("done test 8\n");

	printf("Test 9 - try to non-existant directory\n");
	data_object = digital_rf_create_write_hdf5("/nosuchdirectory/junk", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_9", 9, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		printf("Got expected error by trying to write to non-existant directory\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}

	printf("Test 10 - try to write to a file instead of a directory\n");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0/rf_data_seq_000000.hdf5", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_10", 9, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		printf("Got expected error by trying to write to a file\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}

	printf("Test 11 - try to write to a directory that already has data\n");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk6", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_6", 9, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_double, vector_length);
	if (result)
		printf("Got expected error by trying to write to a directory that already has data\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}
	digital_rf_close_write_hdf5(data_object);

	printf("Test 12 - try to use illegal compression level 10\n");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk6", H5T_NATIVE_DOUBLE, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_10", 10, 1, 1, 1, is_continuous, 1);
	if (!data_object)
		printf("Got expected error by trying to use compression level 10\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}

	printf("Test 13 - try to write using digital_rf_write_blocks_hdf5, but with overlapping data, both compress (level 6) and checksum - channel 13\n");
	system("rm -rf /tmp/hdf5/junk13 ; mkdir /tmp/hdf5/junk13");
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk13", H5T_NATIVE_LLONG, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_4", 6, 1, 0, 1, is_continuous, 1);
	if (!data_object)
		exit(-1);
	init_block_indices(global_index_arr, block_index_arr, 0);
	result = digital_rf_write_blocks_hdf5(data_object, global_index_arr, block_index_arr, 10, single_int64, vector_length);
	if (result)
		exit(-1);
	init_block_indices(global_index_arr, block_index_arr, 185); /* last write was 190 samples, illegal overlap should be thrown */
	result = digital_rf_write_blocks_hdf5(data_object, global_index_arr, block_index_arr, 10, single_int64, vector_length);
	if (result)
	{
		printf("Got expected overlap error\n");
		digital_rf_close_write_hdf5(data_object);
	}
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}
	printf("done test 13\n");

	printf("Test 14 - try to write using digital_rf_write_blocks_hdf5, but with block data indices that advance slower than the global index - channel 13\n");
		system("rm -rf /tmp/hdf5/junk14 ; mkdir /tmp/hdf5/junk14");
		data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk14", H5T_NATIVE_LLONG, 2, 400, global_index, sample_rate_numerator, sample_rate_denominator,
				"FAKE_UUID_4", 6, 1, 0, 1, is_continuous, 1);
		if (!data_object)
			exit(-1);
		init_invalid_block_indices(global_index_arr, block_index_arr, 0);
		result = digital_rf_write_blocks_hdf5(data_object, global_index_arr, block_index_arr, 10, single_int64, vector_length);
		if (result)
		{
			printf("Got expected bad block index error\n");
			digital_rf_close_write_hdf5(data_object);
		}
		else
		{
			printf("TEST FAILED!!!!! Error should have been thrown\n");
			exit(-1);
		}
		printf("done test 14\n");

	printf("Test 15 - try to write more to channel but with different data type - should throw error\n");
	is_continuous = 0;
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk1.0", H5T_NATIVE_INT, 2, 400, global_index + 800, sample_rate_numerator, sample_rate_denominator,
			"FAKE_UUID_1.0", 0, 0, 1, ARR_SIZE, is_continuous, 1);
	if (!data_object)
		printf("got expected error\n");
	else
	{
		printf("TEST FAILED!!!!! Error should have been thrown\n");
		exit(-1);
	}
	printf("done test 15\n");

	printf("All tests completed successfully\n");
	return(0);

}
