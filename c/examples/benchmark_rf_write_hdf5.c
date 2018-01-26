/*
 * Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
 * All rights reserved.
 *
 * Distributed under the terms of the BSD 3-clause license.
 *
 * The full license is in the LICENSE file, distributed with this software.
*/
/*
 * Benchmark write speed. digital rf 2.0
 */
#include <time.h>
#include <stdio.h>
#include "digital_rf.h"

void digital_rf_randomize_int16(int16_t *data, int len)
{
  int i;
  for(i=0 ; i<len ; i++)
  {
    data[i]=(i%32768)*(i+8192)*(i%13);
  }
}

void digital_rf_randomize_float32(float *data, int len)
{
  int i;
  for(i=0 ; i<len ; i++)
  {
    data[i]=((float)((int16_t)(i%32768)*(i+8192)*(i%13)))/32767.0;
  }
}

// length of random number buffer
#define NUM_SUBCHANNELS 4
#define RANDOM_BLOCK_SIZE 4194304 * NUM_SUBCHANNELS
// the last starting index used from buffer
#define N_SAMPLES 1048576
#define WRITE_BLOCK_SIZE 1000000
// set first time to be March 9, 2014
#define START_TIMESTAMP 1394368230
#define SAMPLE_RATE_NUMERATOR 1000000
#define SAMPLE_RATE_DENOMINATOR 1
#define SUBDIR_CADENCE 10
#define MILLISECS_PER_FILE 1000

// uncomment to enable specific tests
#define TEST_FWRITE
#define TEST_HDF5
#define TEST_HDF5_CHECKSUM
#define TEST_HDF5_CHECKSUM_COMPRESS

int main (int argc, char *argv[])
{
  /* 16-bit integer data */
  int16_t *data_int16;
  uint64_t i, result;
  uint64_t vector_length;
  int n_writes;
  clock_t begin, end;
  double time_spent;
  data_int16 = (int16_t *)malloc(RANDOM_BLOCK_SIZE*sizeof(int16_t));
  vector_length=WRITE_BLOCK_SIZE;
  n_writes = (int)1e8/WRITE_BLOCK_SIZE;
  printf("randomize data vector\n");
  digital_rf_randomize_int16(data_int16,RANDOM_BLOCK_SIZE);

  /* local variables */
  Digital_rf_write_object *data_object = NULL;
  uint64_t vector_leading_edge_index = 0;
  uint64_t global_start_sample = (uint64_t)(START_TIMESTAMP * ((long double)SAMPLE_RATE_NUMERATOR)/SAMPLE_RATE_DENOMINATOR);

#ifdef TEST_FWRITE
  int file_idx;
  FILE *f;
  char fname[4096];

  printf("Test -1 - fwrite raw data to disk\n");
  result = system("rm -rf /tmp/hdf5 ; mkdir /tmp/hdf5 ; mkdir /tmp/hdf5/junk0");
  printf("Start writing\n");
  begin = clock();
  file_idx=0;
  sprintf(fname,"/tmp/hdf5/junk0/file-%06d.bin",file_idx);
  f = fopen(fname, "w");

  for(i=0 ; i<2*n_writes ; i++)
  {
    fwrite(&data_int16[vector_leading_edge_index%N_SAMPLES], sizeof(int16_t), vector_length, f);
    if(i%2 == 0)
    {
      fclose(f);
      file_idx++;
      sprintf(fname,"/tmp/hdf5/junk0/file-%06d.bin",file_idx);
      f = fopen(fname, "wb");
    }
    vector_leading_edge_index+=vector_length;
  }
  fclose(f);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("done test %1.2f MB/s\n",((double)2.0*n_writes*2.0*vector_length)/time_spent/1e6);
#endif
#ifdef TEST_HDF5
  printf("Test 0 - simple single write to multiple files, no compress, no checksum - channel 0\n");
  result = system("rm -rf /tmp/hdf5/junk0 ; mkdir /tmp/hdf5/junk0");
  printf("Start writing\n");
  vector_leading_edge_index=0;
  data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0", H5T_NATIVE_SHORT, SUBDIR_CADENCE, MILLISECS_PER_FILE, global_start_sample, SAMPLE_RATE_NUMERATOR, SAMPLE_RATE_DENOMINATOR,
		  "FAKE_UUID_0", 0, 0, 1, NUM_SUBCHANNELS, 1, 1);
  if (!data_object)
  		exit(-1);
  begin = clock();

  if (!data_object)
    exit(-1);
  for(i=0 ; i<n_writes ; i++)
  {
	result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_int16, vector_length);
    vector_leading_edge_index+=vector_length;

    if (result)
      exit(-1);
  }
  digital_rf_close_write_hdf5(data_object);

  /* here, do your time-consuming job */
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("done test %1.2f MB/s\n",((double)n_writes*4*NUM_SUBCHANNELS*vector_length)/time_spent/1e6);
#endif
#ifdef TEST_HDF5_CHECKSUM
  printf("Test 1 - simple single write to multiple files, no compress, checksum - channel 0\n");
  result = system("rm -rf /tmp/hdf5/junk0 ; mkdir /tmp/hdf5/junk0");
  printf("Start writing\n");
  vector_leading_edge_index=0;
  data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0", H5T_NATIVE_SHORT, SUBDIR_CADENCE, MILLISECS_PER_FILE, global_start_sample, SAMPLE_RATE_NUMERATOR, SAMPLE_RATE_DENOMINATOR,
		  "FAKE_UUID_0", 0, 1, 1, NUM_SUBCHANNELS, 1, 1);
  if (!data_object)
  		exit(-1);
  begin = clock();

  if (!data_object)
    exit(-1);
  for(i=0 ; i<n_writes ; i++)
  {
    result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_int16, vector_length);
    vector_leading_edge_index+=WRITE_BLOCK_SIZE;

    if (result)
      exit(-1);
  }
  digital_rf_close_write_hdf5(data_object);

  /* here, do your time-consuming job */
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("done test %1.2f MB/s\n",((double)n_writes*4.0*NUM_SUBCHANNELS*vector_length)/time_spent/1e6);
#endif
#ifdef TEST_HDF5_CHECKSUM_COMPRESS
  printf("Test 2 - simple single write to multiple files, compress, checksum - channel 0\n");
  result = system("rm -rf /tmp/hdf5/junk0 ; mkdir /tmp/hdf5/junk0");
  printf("Start writing\n");
  vector_leading_edge_index=0;
  data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0", H5T_NATIVE_SHORT, SUBDIR_CADENCE, MILLISECS_PER_FILE, global_start_sample, SAMPLE_RATE_NUMERATOR, SAMPLE_RATE_DENOMINATOR,
		  "FAKE_UUID_0", 1, 1, 1, NUM_SUBCHANNELS, 1, 1);
  if (!data_object)
  		exit(-1);
  begin = clock();

  if (!data_object)
    exit(-1);
  for(i=0 ; i<n_writes ; i++)
  {
    result = digital_rf_write_hdf5(data_object, vector_leading_edge_index, data_int16, vector_length);
    vector_leading_edge_index+=WRITE_BLOCK_SIZE;

    if (result)
      exit(-1);
  }
  digital_rf_close_write_hdf5(data_object);

  /* here, do your time-consuming job */
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("done test %1.2f MB/s\n",((double)n_writes*4.0*NUM_SUBCHANNELS*vector_length)/time_spent/1e6);
#endif
  result = system("rm -rf /tmp/hdf5/junk0");
  free(data_int16);
  return(0);
}
