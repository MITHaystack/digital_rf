/* -*- c++ -*- */
/*
 * Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
 * All rights reserved.
 *
 * Distributed under the terms of the BSD 3-clause license.
 *
 * The full license is in the LICENSE file, distributed with this software.
*/

#ifndef INCLUDED_GRDRF_DIGITAL_RF_SINK_IMPL_H
#define INCLUDED_GRDRF_DIGITAL_RF_SINK_IMPL_H

#include <gr_drf/digital_rf_sink.h>

extern "C" {
#include <digital_rf.h>
}

namespace gr {
  namespace drf {

    class digital_rf_sink_impl : public digital_rf_sink
    {
     private:
      char d_dir[4096];
      size_t d_sample_size;
      uint64_t d_subdir_cadence_s;
      uint64_t d_file_cadence_ms;
      uint64_t d_sample_rate_numerator;
      uint64_t d_sample_rate_denominator;
      long double d_sample_rate;
      char d_uuid[512];
      bool d_is_complex;
      int d_num_subchannels;
      bool d_stop_on_dropped_packet;
      bool d_ignore_tags;

      Digital_rf_write_object *d_drfo;
      hid_t d_dtype;
      uint64_t d_t0; // start time in samples from unix epoch
      bool d_t0_set;
      uint64_t d_local_index;
      uint64_t d_total_dropped;
      bool d_first;

      char *d_zero_buffer;

      // make copy constructor private with no implementation to prevent copying
      digital_rf_sink_impl(const digital_rf_sink_impl &that);

     public:
      digital_rf_sink_impl(char *dir, size_t sample_size,
                           uint64_t subdir_cadence_s, uint64_t file_cadence_ms,
                           uint64_t sample_rate_numerator,
                           uint64_t sample_rate_denominator,
                           char *uuid, bool is_complex,
                           int num_subchannels, bool stop_on_dropped_packet = 0,
                           uint64_t *start_sample_index = NULL,
                           bool ignore_tags = 0);
      ~digital_rf_sink_impl();

      int get_rx_time(int ninput_items);
      int detect_and_handle_overflow(uint64_t rel_start, uint64_t rel_end, char *in);

      // Where all the action really happens
      int work(int noutput_items,
               gr_vector_const_void_star &input_items,
               gr_vector_void_star &output_items);
    };

  } // namespace drf
} // namespace gr

#endif /* INCLUDED_GRDRF_DIGITAL_RF_SINK_IMPL_H */
