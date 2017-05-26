/* -*- c++ -*- */
/*
 * Copyright (c) 2017 Massachusetts Institute of Technology
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdexcept>
#include <gnuradio/io_signature.h>
#include "digital_rf_sink_impl.h"

extern "C" {
#include <digital_rf.h>
}

#define ZERO_BUFFER_SIZE 10000000

namespace gr {
  namespace drf {

    digital_rf_sink::sptr
    digital_rf_sink::make(char *dir, size_t sample_size,
                          uint64_t subdir_cadence_s, uint64_t file_cadence_ms,
                          uint64_t sample_rate_numerator,
                          uint64_t sample_rate_denominator,
                          char *uuid, bool is_complex,
                          int num_subchannels, bool stop_on_dropped_packet)
    {
      return gnuradio::get_initial_sptr
        (new digital_rf_sink_impl(dir, sample_size, subdir_cadence_s,
                                  file_cadence_ms, sample_rate_numerator,
                                  sample_rate_denominator, uuid,
                                  is_complex, num_subchannels,
                                  stop_on_dropped_packet));
    }


    /*
     * The private constructor
     */
    digital_rf_sink_impl::digital_rf_sink_impl(
            char *dir, size_t sample_size, uint64_t subdir_cadence_s,
            uint64_t file_cadence_ms, uint64_t sample_rate_numerator,
            uint64_t sample_rate_denominator, char* uuid,
            bool is_complex, int num_subchannels, bool stop_on_dropped_packet
    )
      : gr::sync_block("digital_rf_sink",
               gr::io_signature::make(1, 1, sample_size*num_subchannels),
               gr::io_signature::make(0, 0, 0)),
        d_sample_size(sample_size), d_subdir_cadence_s(subdir_cadence_s),
        d_file_cadence_ms(file_cadence_ms),
        d_sample_rate_numerator(sample_rate_numerator),
        d_sample_rate_denominator(sample_rate_denominator),
        d_is_complex(is_complex), d_num_subchannels(num_subchannels),
        d_stop_on_dropped_packet(stop_on_dropped_packet)
    {
      char command[4096];
      int i;

      d_sample_rate = ((long double)sample_rate_numerator /
                       (long double)sample_rate_denominator);

      if(d_is_complex)
      {
        // complex char (int8)
        if(d_sample_size == 2) {
          d_dtype = H5T_NATIVE_CHAR;
        }
        // complex short (int16)
        else if(d_sample_size == 4) {
          d_dtype = H5T_NATIVE_SHORT;
        }
        // complex float (float32)
        else if(d_sample_size == 8) {
          d_dtype = H5T_NATIVE_FLOAT;
        }
        // complex double (float64)
        else if(d_sample_size == 16) {
          d_dtype = H5T_NATIVE_DOUBLE;
        }
        else {
          std::invalid_argument("Item size not supported");
        }
      }
      else
      {
        // char (int8)
        if(d_sample_size == 1) {
          d_dtype = H5T_NATIVE_CHAR;
        }
        // short (int16)
        else if(d_sample_size == 2) {
          d_dtype = H5T_NATIVE_SHORT;
        }
        // float (float32)
        else if(d_sample_size == 4) {
          d_dtype = H5T_NATIVE_FLOAT;
        }
        // double (float64)
        else if(d_sample_size == 8) {
          d_dtype = H5T_NATIVE_DOUBLE;
        }
        else {
          std::invalid_argument("Item size not supported");
        }
      }

      strcpy(d_dir, dir);
      sprintf(command, "mkdir -p %s", d_dir);
      printf("%s\n", command);
      fflush(stdout);
      int ignore_this = system(command);

      strcpy(d_uuid, uuid);

      printf("subdir_cadence_s %lu file_cadence_ms %lu sample_size %d rate %1.2Lf\n",
             subdir_cadence_s, file_cadence_ms, (int)sample_size, d_sample_rate);

      d_zero_buffer = (char *)malloc(ZERO_BUFFER_SIZE*sizeof(char));
      for(i=0; i<ZERO_BUFFER_SIZE; i++) {
        d_zero_buffer[i] = 0;
      }

      d_first = 1;
      d_t0 = 1;
      d_local_index = 0;
      d_total_dropped = 0;
    }

    /*
     * Our virtual destructor.
     */
    digital_rf_sink_impl::~digital_rf_sink_impl()
    {
      digital_rf_close_write_hdf5(d_drfo);
      free(d_zero_buffer);
    }


    void
    digital_rf_sink_impl::get_rx_time(int n)
    {
      struct timeval tv;

      std::vector<gr::tag_t> rx_time_tags;
      get_tags_in_range(rx_time_tags, 0, 0, n, pmt::string_to_symbol("rx_time"));

      double t0_frac;
      uint64_t t0_sec;

      //print all tags
      BOOST_FOREACH(const gr::tag_t &rx_time_tag, rx_time_tags) {
        const uint64_t offset = rx_time_tag.offset;
        const pmt::pmt_t &value = rx_time_tag.value;

        t0_sec = pmt::to_uint64(pmt::tuple_ref(value, 0));
        t0_frac = pmt::to_double(pmt::tuple_ref(value, 1));
        d_t0 = (uint64_t)(d_sample_rate*t0_sec)
                + (uint64_t)(d_sample_rate*t0_frac);
        printf("Time tag @ sample %lu (%lu): %lu+%f\n", offset, d_t0, t0_sec, t0_frac);
      }
    }

    int
    digital_rf_sink_impl::detect_and_handle_overflow(uint64_t start,
                                                     uint64_t end,
                                                     char *in)
    {
      std::vector<gr::tag_t> rx_time_tags;
      uint64_t dt;
      uint64_t dropped = 0;
      uint64_t drop_index;
      int result;
      int consumed = 0;
      int filled;

      get_tags_in_range(rx_time_tags, 0, start, end, pmt::string_to_symbol("rx_time"));

      //print all tags
      BOOST_FOREACH(const gr::tag_t &rx_time_tag, rx_time_tags) {
        const uint64_t offset = rx_time_tag.offset;
        const pmt::pmt_t &value = rx_time_tag.value;

        uint64_t tt0_sec = pmt::to_uint64(pmt::tuple_ref(value, 0));
        double tt0_frac = pmt::to_double(pmt::tuple_ref(value, 1));

        // get sample index of drop (as opposed to packet index == offset)
        drop_index = offset + d_total_dropped;

        // we should have this many samples
        dt = ((uint64_t)(d_sample_rate*tt0_sec) + (uint64_t)(d_sample_rate*tt0_frac)
              - d_t0 - d_total_dropped);

        dropped = dt - offset;
        d_total_dropped += dropped;
        printf("\nDropped %lu packet(s) @ %lu, total_dropped %lu\n",
               dropped, drop_index, d_total_dropped);

        // write in-sequence data up to drop_index
        result = digital_rf_write_hdf5(d_drfo, d_local_index,
                                       in + consumed*d_sample_size*d_num_subchannels,
                                       drop_index - d_local_index);
        if(result) {
          throw std::runtime_error("Nonzero result on write");
        }
        consumed += drop_index - d_local_index;
        d_local_index = drop_index;

        if(d_stop_on_dropped_packet && dropped > 0) {
          printf("Stopping as requested\n");
          return WORK_DONE;
        }

        // if we've dropped packets, write zeros
        while(dropped > 0) {
          if(dropped*d_sample_size*d_num_subchannels <= ZERO_BUFFER_SIZE) {
            filled = dropped;
          }
          else {
            filled = ZERO_BUFFER_SIZE/d_sample_size/d_num_subchannels;
          }
          result = digital_rf_write_hdf5(d_drfo, d_local_index, d_zero_buffer, filled);
          if(result) {
            throw std::runtime_error("Nonzero result on write");
          }
          d_local_index += filled;
          dropped -= filled;
        }
      }
      return(consumed);
    }


    int
    digital_rf_sink_impl::work(int noutput_items,
                               gr_vector_const_void_star &input_items,
                               gr_vector_void_star &output_items)
    {
      char *in = (char *)input_items[0];
      int result, i;
      int samples_consumed = 0;

      if(d_first) {
        // sets start time d_t0
        get_rx_time(noutput_items);

        printf("Creating %s t0 %ld\n", d_dir, d_t0);
        fflush(stdout);
        /*      Digital_rf_write_object * digital_rf_create_write_hdf5(
                    char * directory, hid_t dtype_id, uint64_t subdir_cadence_secs,
                    uint64_t file_cadence_millisecs, uint64_t global_start_sample,
                    uint64_t sample_rate_numerator, uint64_t sample_rate_denominator,
                    char * uuid_str, int compression_level, int checksum, int is_complex,
                    int num_subchannels, int is_continuous, int marching_dots
                )
        */
        d_drfo = digital_rf_create_write_hdf5(
                d_dir, d_dtype, d_subdir_cadence_s, d_file_cadence_ms, d_t0,
                d_sample_rate_numerator, d_sample_rate_denominator, d_uuid,
                0, 0, d_is_complex, d_num_subchannels, 1, 1);
        if(!d_drfo) {
          throw std::runtime_error("Failed to create Digital RF writer object");
        }
        printf("done\n");
        d_first = 0;
      }
      else {
        samples_consumed = detect_and_handle_overflow(nitems_read(0),
                                                      nitems_read(0) + noutput_items,
                                                      in);
      }

      in += samples_consumed*d_sample_size*d_num_subchannels;
      result = digital_rf_write_hdf5(d_drfo, d_local_index, in,
                                     noutput_items - samples_consumed);
      if(result) {
        throw std::runtime_error("Nonzero result on write");
      }
      d_local_index += noutput_items - samples_consumed;

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace drf */
} /* namespace gr */
