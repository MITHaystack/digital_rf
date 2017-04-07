/* -*- c++ -*- */
/*
 * Copyright (c) 2015 Massachusetts Institute of Technology
 *
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

      Digital_rf_write_object *d_drfo;
      hid_t d_dtype;
      uint64_t d_t0; // start time in samples from unix epoch
      uint64_t d_local_index;
      uint64_t d_total_dropped;
      bool d_first;

      char *d_zero_buffer;

      // make copy constructor private with no implementation to prevent copying
      digital_rf_sink_impl(const digital_rf_sink_impl& that);

     public:
      digital_rf_sink_impl(char *dir, size_t sample_size,
                           uint64_t subdir_cadence_s, uint64_t file_cadence_ms,
                           uint64_t sample_rate_numerator,
                           uint64_t sample_rate_denominator,
                           char* uuid, bool is_complex,
                           int num_subchannels, bool stop_on_dropped_packet);
      ~digital_rf_sink_impl();

      void get_rx_time(int n);
      int detect_and_handle_overflow(uint64_t start, uint64_t end, char *in);

      // Where all the action really happens
      int work(int noutput_items,
               gr_vector_const_void_star &input_items,
               gr_vector_void_star &output_items);
    };

  } // namespace drf
} // namespace gr

#endif /* INCLUDED_GRDRF_DIGITAL_RF_SINK_IMPL_H */
