/*
 * Simple example of writing Digital RF 2.0 data with interrupt handling
 *
 * This simple example writes continuous complex data of short ints,
 * and also removes partially written files when a SIGINT occurs
 *
 * $Id$
 */

#include <signal.h>
#include <unistd.h>

#include "digital_rf.h"

/* the following code gives an example of writing a SIGINT handler that deletes the tmp file being written
 * Note that this assumes a non-threaded application because it uses a global to track the last file
 */


char global_last_rf_dir_written[BIG_HDF5_STR];

void intHandler(int dummy)
{
	char cmd[BIG_HDF5_STR] = "";
	sprintf(cmd, "rm %s/tmp.*", global_last_rf_dir_written);
	system(cmd);
    exit(-1);
}



int main (int argc, char *argv[])
{

	/* local variables */
	Digital_rf_write_object * data_object = NULL; /* main object created by init */
	uint64_t vector_leading_edge_index = 0; /* index of the sample being written starting at zero with the first sample recorded */
	uint64_t global_start_index; /* start sample (unix time * sample_rate) of first measurement - set below */
	int i, result;
	char * local_last_dir_written; /* used for interrupt handler */

	/* dummy dataset to write */
	short data_short[100][2];

	/* writing parameters */
	uint64_t sample_rate_numerator = 100; /* 100 Hz sample rate - typically MUCH faster */
	uint64_t sample_rate_denominator = 1;
	uint64_t subdir_cadence = 4; /* Number of seconds per subdirectory - typically longer */
	uint64_t millseconds_per_file = 400; /* Each subdirectory will have up to 10 400 ms files */
	int compression_level = 1; /* low level of compression */
	int checksum = 0; /* no checksum */
	int is_complex = 1; /* complex values */
	int is_continuous = 1; /* continuous data written */
	int num_subchannels = 1; /* only one subchannel */
	int marching_periods = 0; /* no marching periods when writing */
	char uuid[100] = "Fake UUID - use a better one!";
	uint64_t vector_length = 100; /* number of samples written for each call - typically MUCH longer */

	/* set up signal handling */
	signal(SIGINT, intHandler);

	/* init dataset */
	for (i=0; i<100; i++)
	{
		data_short[i][0] = 2*i;
		data_short[i][1] = 3*i;
	}

	/* start recording at global_start_sample */
	global_start_index = (uint64_t)(1394368230 * (long double)sample_rate_numerator/sample_rate_denominator) + 1; /* should represent 2014-03-09 12:30:30  and 10 milliseconds*/


	printf("Writing complex short to multiple files and subdirectores in /tmp/hdf5 channel junk0\n");
	system("rm -rf /tmp/hdf5 ; mkdir /tmp/hdf5 ; mkdir /tmp/hdf5/junk0");

	/* init */
	data_object = digital_rf_create_write_hdf5("/tmp/hdf5/junk0", H5T_NATIVE_INT, subdir_cadence, millseconds_per_file,
			global_start_index, sample_rate_numerator, sample_rate_denominator, uuid, compression_level, checksum, is_complex, num_subchannels,
			is_continuous, marching_periods);
	if (!data_object)
		exit(-1);

	/* write continuous data */
	for (i=0; i<10000; i++)
	{
		result = digital_rf_write_hdf5(data_object, vector_leading_edge_index + i*100, data_short, vector_length);
		if (result)
			exit(-1);
		/* update global last_rf_dir_written after every call */
		local_last_dir_written = digital_rf_get_last_dir_written(data_object);
		strcpy(global_last_rf_dir_written, local_last_dir_written);
		free(local_last_dir_written); /* digital_rf_get_last_dir_written dynamically allocates memory */
		if (i % 1000 == 0)
			printf("%i of 10000 written\n", i);
	}

	/* close */
	digital_rf_close_write_hdf5(data_object);

	printf("example done - examine /tmp/hdf5 for data\n");
	return(0);
}
