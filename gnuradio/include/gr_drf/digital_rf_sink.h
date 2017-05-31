/* -*- c++ -*- */
/*
 * Copyright (c) 2015 Massachusetts Institute of Technology
 *
 */


#ifndef INCLUDED_GRDRF_DIGITAL_RF_SINK_H
#define INCLUDED_GRDRF_DIGITAL_RF_SINK_H

#include <gr_drf/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace drf {

    /*!
     * \brief Write data in Digital RF format.
     * \ingroup drf
     *
     */
    class GRDRF_API digital_rf_sink : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<digital_rf_sink> sptr;

      /*!
       * \brief Create a Digital RF sink block.
       *
       * \param dir Directory to write to.
       * \param sample_size Size of the input data items.
       * \param subdir_cadence_s Number of seconds of data per subdirectory.
       * \param file_cadence_ms Number of milliseconds of data per file.
       * \param sample_rate_numerator Numerator of sample rate in Hz.
       * \param sample_rate_denominator Denominator of sample rate in Hz.
       * \param uuid Unique ID to associate with this data, for pairing metadata.
       * \param is_complex True if the data samples are complex.
       * \param num_subchannels Number of subchannels (i.e. vector length).
       * \param stop_on_dropped_packet If True, stop when a packet is dropped.
       * \param start_sample_index Index in samples since epoch for first sample.
       * \param ignore_tags If True, do not use rx_time tag to set sample index.
       *
       */
      static sptr make(char *dir, size_t sample_size,
                       uint64_t subdir_cadence_s, uint64_t file_cadence_ms,
                       uint64_t sample_rate_numerator,
                       uint64_t sample_rate_denominator,
                       char *uuid, bool is_complex,
                       int num_subchannels, bool stop_on_dropped_packet = 0,
                       uint64_t *start_sample_index = NULL,
                       bool ignore_tags = 0);
    };

  } // namespace drf
} // namespace gr

#endif /* INCLUDED_GRDRF_DIGITAL_RF_SINK_H */
