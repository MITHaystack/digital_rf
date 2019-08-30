# ----------------------------------------------------------------------------
# Copyright (c) 2018 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module defining raster (periodic window) tools for GNU Radio."""
from __future__ import absolute_import, division, print_function

import numpy as np
import pmt
from gnuradio import gr

__all__ = ("raster_chunk", "raster_select_aggregate", "raster_tag")


class raster_chunk(gr.basic_block):
    """Block for chunking periodic rasters into fixed-size vectors."""

    def __init__(
        self,
        dtype=np.complex64,
        vlen=1,
        raster_length=10000,
        nperseg=1,
        noverlap=0,
        max_raster_length=None,
        max_noverlap=None,
    ):
        """Chunk periodic rasters into vectors with optional overlap.

        The input data is provided as samples with length `vlen` and type
        `dtype`. It is then divided into raster windows with a number of
        samples equal to `raster_length`. Each raster window is then broken
        into chunks of `nperseg` samples with an overlap of `noverlap` samples.
        The output may be zero-padded at the end to ensure that all of the
        samples in the raster window are included in an output chunk. Each
        chunk is output as a vector whose total length is ``nperseg * vlen``.

        The advantage of a raster of data is that its size can be changed in
        a running flowgraph, but it can be useful to interface raster data
        with fixed-size vectors (such as for FFTs).


        Parameters
        ----------
        dtype : np.dtype
            Data type of the input and output data.

        vlen : int
            Vector length of the *input* data (NOT the output vector length).

        raster_length : int
            Length of the raster window.

        nperseg : int
            Fixed length of each output chunk. If the input data is itself a
            vector, then each output vector will have a length of
            ``nperseg * vlen``.

        noverlap : int
            Number of samples to overlap for each output chunk.


        Other Parameters
        ----------------
        max_raster_length : int
            Maximum possible raster length, to allow for changes while the
            block is running. Knowing the maximum length allows for allocation
            of appropriately-sized buffers. If None, four times the initial
            `raster_length` will be used.

        max_noverlap : int
            Maximum possible number of overlap samples, to allow for changes
            while the block is running. Knowing the maximum number allows for
            allocation of appropriately-sized buffers. If None, two thirds of
            `nperseg` will be used.

        """
        if max_raster_length is None:
            max_raster_length = 4 * raster_length
        if max_noverlap is None:
            max_noverlap = 2 * nperseg // 3
        gr.basic_block.__init__(
            self,
            name="Raster Chunk",
            in_sig=[(dtype, vlen)],
            out_sig=[(dtype, vlen * nperseg)],
        )
        self._dtype = dtype
        self._vlen = vlen
        self._nperseg = max(1, nperseg)
        self._max_raster_length = max_raster_length
        self._max_noverlap = max_noverlap
        if raster_length < nperseg or raster_length > max_raster_length:
            errstr = "raster_length {0} must be between {1} and {2}"
            raise ValueError(errstr.format(raster_length, nperseg, max_raster_length))
        if noverlap < 0 or noverlap > min(nperseg - 1, max_noverlap):
            errstr = "noverlap {0} must be between 0 and {1}"
            raise ValueError(errstr.format(noverlap, min(nperseg - 1, max_noverlap)))

        # set parameters to max values to size buffer, then set to true values
        self.set_raster_length(max_raster_length)
        self.set_noverlap(max_noverlap)
        # makes sure the buffers have the max size
        self._set_params()
        # now the true values
        self.set_raster_length(raster_length)
        self.set_noverlap(noverlap)

        # tags become meaningless on vector output
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def _set_params(self):
        """Finalize given parameter values and calculate derived values."""
        self._raster_length = min(
            max(self._next_raster_length, self._nperseg), self._max_raster_length
        )
        self._noverlap = min(
            max(self._next_noverlap, 0), self._nperseg - 1, self._max_noverlap
        )
        nstep = self._nperseg - self._noverlap
        nchunks = int(np.ceil(float(self._raster_length) / nstep))
        self._nstep = nstep
        self._nchunks = nchunks
        # prepare zero-padded array for strided view of input raster
        padded_len = (self._nchunks - 1) * self._nstep + self._nperseg
        self._zeropadded = np.zeros((padded_len, self._vlen), dtype=self._dtype)
        self._in_raster = self._zeropadded[: self._raster_length]
        stride_shape = (self._nchunks, self._nperseg, self._vlen)
        strides = (
            self._nstep * self._zeropadded.strides[0],
        ) + self._zeropadded.strides
        self._strided = np.lib.stride_tricks.as_strided(
            self._zeropadded, stride_shape, strides
        )
        self._out_raster = self._strided.reshape(
            (self._nchunks, self._nperseg * self._vlen)
        )
        # set rate parameters
        self.set_output_multiple(self._nchunks)
        rate = float(self._nchunks) / self._raster_length
        self.set_relative_rate(rate)
        self._params_set = True

    def _adjust_params(self):
        """Check if the parameter values have changed and set them if so."""
        if not self._params_set:
            self._set_params()
            return True
        else:
            return False

    def set_raster_length(self, raster_length):
        """Set a new raster length."""
        self._next_raster_length = raster_length
        self._params_set = False

    def set_noverlap(self, noverlap):
        """Set a new number of overlap samples for the output chunks."""
        self._next_noverlap = noverlap
        self._params_set = False

    def forecast(self, noutput_items, ninput_items_required):
        """Determine number of input items required given an output number."""
        # since we set output_multiple, noutput_items is a multiple of
        # self._nchunks
        n = noutput_items // self._nchunks
        ninput_items_required[0] = n * self._raster_length

    def general_work(self, input_items, output_items):
        """Perform the block tasks on given input and output buffers."""
        in_arr = input_items[0].reshape((-1, self._vlen))
        out_arr = output_items[0].reshape((-1, self._nperseg * self._vlen))
        noutput_items = len(out_arr)

        # check if params changed, adjust and restart work if they have
        if self._adjust_params():
            return 0

        # noutput_items is a multiple of self._nchunks because we set
        # output_multiple to be self._nchunks
        nrasters = noutput_items // self._nchunks
        for k_raster in range(nrasters):
            in_idx = k_raster * self._raster_length
            out_idx = k_raster * self._nchunks

            # copy input raster into zeropadded memory
            self._in_raster[...] = in_arr[in_idx : (in_idx + self._raster_length), :]

            # copy strided chunks to output
            out_arr[out_idx : (out_idx + self._nchunks), :] = self._out_raster

        self.consume(0, nrasters * self._raster_length)
        return noutput_items


class raster_select_aggregate(gr.basic_block):
    """Block for selecting data from a raster and optionally aggregating it."""

    def __init__(
        self,
        dtype=np.complex64,
        vlen=1,
        raster_length=10000,
        select_start=0,
        select_length=None,
        nagg=1,
        agg_op="take",
        agg_op_args=(0,),
        max_raster_length=None,
        max_select_length=None,
        max_nagg=None,
    ):
        """Select data from a periodic raster window and optionally aggregate.

        The input data is provided as samples with length `vlen` and type
        `dtype`. It is then divided into raster windows with a number of
        samples equal to `raster_length`. Within and relative to each raster
        window, samples are selected to be output using `select_start` and
        `select_length`. The output rasters can optionally be aggregated
        together from `nagg` outputs to one using the specified operation.

        The advantage of a raster of data is that its size can be changed in
        a running flowgraph.


        Parameters
        ----------
        dtype : np.dtype
            Data type of the input and output data.

        vlen : int
            Vector length of the *input* data (NOT the output vector length).

        raster_length : int
            Length of the raster window.

        select_start : int
            Index relative to the start of the raster window that indicates the
            start of the output raster.

        select_length : int
            Number of samples to include in the selection from the raster
            window. The equivalent indexing of the raster window would then be
            ``raster[select_start:(select_start + select_length)]``. If None,
            then the length of entire remaining raster window from
            `select_start` will be used.

        nagg : int
            Number of output rasters to aggregate together. The output is thus
            downsampled by `nagg` in whole chunks of the selected raster
            window.

        agg_op : str
            String giving the name of a numpy array method to use for the
            aggregation operation. For `nagg` output rasters organized as an
            ``(nagg, select_length, vlen)``-shaped array called ``selections``,
            the aggregation operation would then be
            ``selections.agg_op(*agg_op_args, axis=0)``.

        agg_op_args : tuple
            Positional arguments to be passed to the aggregation operation
            method specified by `agg_op`. See above.


        Other Parameters
        ----------------
        max_raster_length : int
            Maximum possible raster length, to allow for changes while the
            block is running. Knowing the maximum length allows for allocation
            of appropriately-sized buffers. If None, four times the initial
            `raster_length` will be used.

        max_select_length : int
            Maximum possible selection length, to allow for changes while the
            block is running. Knowing the maximum length allows for allocation
            of appropriately-sized buffers. If None, four times the initial
            `select_length` will be used.

        max_nagg : int
            Maximum possible output aggregation, to allow for changes while the
            block is running. Knowing the maximum aggregation size allows for
            allocation of appropriately-sized buffers. If None, a default of
            four times the initial `nagg` will be used.

        """
        if max_raster_length is None:
            max_raster_length = 4 * raster_length
        if max_select_length is None:
            length = raster_length if select_length is None else select_length
            max_select_length = 4 * length
        if max_nagg is None:
            max_nagg = 4 * nagg
        gr.basic_block.__init__(
            self, name="Raster Select", in_sig=[(dtype, vlen)], out_sig=[(dtype, vlen)]
        )
        self._dtype = dtype
        self._vlen = vlen
        self._max_raster_length = max_raster_length
        self._max_select_length = max_select_length
        self._max_nagg = max_nagg
        self.set_agg_op(agg_op)
        self.set_agg_op_args(agg_op_args)
        # set parameters to max values to size buffer, then set to true values
        self.set_raster_length(max_raster_length)
        self.set_select_start(0)
        self.set_select_length(max_select_length)
        self.set_nagg(max_nagg)
        # makes sure the buffers have the max size
        self._adjust_params()
        # now the true values
        self.set_raster_length(raster_length)
        self.set_select_start(select_start)
        self.set_select_length(select_length)
        self.set_nagg(nagg)

        # we will propogate tags manually
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def _set_params(self):
        """Finalize given parameter values and calculate derived values."""
        # raster parameters
        self._raster_length = max(
            1, min(self._next_raster_length, self._max_raster_length)
        )
        self._nagg = max(1, min(self._next_nagg, self._max_nagg))
        self._ninput_multiple = self._raster_length * self._nagg

        # selection parameters
        self._select_start = self._next_select_start % self._raster_length
        if self._next_select_length is None:
            select_length = self._raster_length - self._select_start
        else:
            select_length = max(1, self._next_select_length)
        self._select_length = min(
            select_length,
            self._raster_length - self._select_start,
            self._max_select_length,
        )
        self._select_stop = self._select_start + self._select_length
        self.set_output_multiple(self._select_length)

        # hint to the scheduler and buffer allocator about rate ratio of output
        # to input
        rate = float(self._select_length) / self._ninput_multiple
        self.set_relative_rate(rate)

        self._params_set = True

    def _adjust_params(self):
        """Check if the parameter values have changed and set them if so."""
        if not self._params_set:
            self._set_params()
            return True
        else:
            return False

    def set_raster_length(self, raster_length):
        """Set a new raster length."""
        self._next_raster_length = raster_length
        self._params_set = False

    def set_select_start(self, select_start):
        """Set a new selection start index."""
        self._next_select_start = select_start
        self._params_set = False

    def set_select_length(self, select_length):
        """Set a new selection length."""
        self._next_select_length = select_length
        self._params_set = False
        self._rate_set = False

    def set_nagg(self, nagg):
        """Set a new aggregation size."""
        self._next_nagg = nagg
        self._params_set = False

    def set_agg_op(self, agg_op):
        """Set a new aggregation operation."""
        self._agg_op = agg_op

    def set_agg_op_args(self, agg_op_args):
        """Set new aggregation arguments."""
        self._agg_op_args = agg_op_args

    def forecast(self, noutput_items, ninput_items_required):
        """Determine number of input items required given an output number."""
        # since we set output_multiple, noutput_items is a multiple of
        # select_length
        nselects = noutput_items // self._select_length
        ninput_items_required[0] = self._ninput_multiple * nselects

    def general_work(self, input_items, output_items):
        """Perform the block tasks on given input and output buffers."""
        in_arr = input_items[0].reshape((-1, self._vlen))
        out_arr = output_items[0].reshape((-1, self._vlen))
        noutput_items = len(out_arr)
        nread = self.nitems_read(0)
        nwritten = self.nitems_written(0)

        # check if params changed, adjust and restart work if they have
        if self._adjust_params():
            return 0

        # noutput_items is a multiple of self._select_length because we set
        # output_multiple to be self._select_length
        nrasters = noutput_items // self._select_length
        for k_raster in range(nrasters):
            in_idx = k_raster * self._ninput_multiple
            out_idx = k_raster * self._select_length

            # forecast makes sure we have at least nagg rasters at input
            raster_samples = in_arr[in_idx : (in_idx + self._ninput_multiple)]
            in_rasters = raster_samples.reshape(
                (self._nagg, self._raster_length, self._vlen)
            )
            in_selects = in_rasters[:, self._select_start : self._select_stop, :]

            if self._nagg > 1:
                # perform operation on rasters
                op_method = getattr(in_selects, self._agg_op)
                out_rasters = op_method(*self._agg_op_args, axis=0)
            else:
                # no operation to perform if we're only aggregating one raster
                out_rasters = in_selects[0]

            # copy result to output
            out_arr[out_idx : (out_idx + self._select_length)] = out_rasters

            # read tags for selected input (only first raster if nagg > 1)
            tags = self.get_tags_in_window(
                0, in_idx + self._select_start, in_idx + self._select_stop
            )

            # write tags to output
            for tag in tags:
                offset_in_select = tag.offset - nread - in_idx - self._select_start
                offset = nwritten + out_idx + offset_in_select
                self.add_item_tag(0, offset, tag.key, tag.value)

        self.consume(0, nrasters * self._ninput_multiple)
        return noutput_items


class raster_tag(gr.sync_block):
    """Block for applying tags within a periodic raster window."""

    def __init__(
        self,
        dtype=np.complex64,
        vlen=1,
        raster_length=10000,
        tags=None,
        max_raster_length=None,
    ):
        """Add tags within a periodic raster window.

        The input data is provided as samples with length `vlen` and type
        `dtype`. It is then divided into raster windows with a number of
        samples equal to `raster_length`. The specified tags are periodically
        added to the output stream relative to the raster window at the given
        indices.

        The advantage of a raster of data is that its size can be changed in
        a running flowgraph. The added tags can be for informational purposes,
        or they could be used to trigger processing or plotting of the raster
        windows.


        Parameters
        ----------
        dtype : np.dtype
            Data type of the input and output data.

        vlen : int
            Vector length of the *input* data (NOT the output vector length).

        raster_length : int
            Length of the raster window.

        tags : list of tuples
            Tags to be added to the output relative to the specified raster
            window. Each tag is represented by a tuple item in the `tags` list
            with the following format:

                tag_item : (int, str, any) tuple
                    The first entry gives the index of the tag relative to the
                    start of each raster window. The second entry gives the
                    name of the tag. The third and final entry gives the tag's
                    value as a python object, to be converted to a pmt value
                    with :func:``pmt.to_pmt``.


        Other Parameters
        ----------------
        max_raster_length : int
            Maximum possible raster length, to allow for changes while the
            block is running. Knowing the maximum length allows for allocation
            of appropriately-sized buffers. If None, four times the initial
            `raster_length` will be used.

        """
        if tags is None:
            tags = [(0, "raster_start", True)]
        if max_raster_length is None:
            max_raster_length = 4 * raster_length
        gr.sync_block.__init__(
            self, name="Tag Raster", in_sig=[(dtype, vlen)], out_sig=[(dtype, vlen)]
        )
        self._dtype = dtype
        self._vlen = vlen
        self._max_raster_length = max_raster_length

        # set parameters to max values to size buffer, then set to true values
        self.set_raster_length(max_raster_length)
        self.set_tags(tags)
        # makes sure the buffers have the max size
        self._set_params()
        # now the true values
        self.set_raster_length(raster_length)
        self.set_tags(tags)

    def _set_params(self):
        """Finalize given parameter values and calculate derived values."""
        # raster length
        self._raster_length = self._next_raster_length
        self.set_output_multiple(self._raster_length)

        # tags
        t = []
        for idx, name, val in self._next_tags:
            o = idx % self._raster_length
            n = pmt.intern(name)
            v = pmt.to_pmt(val)
            t.append((o, n, v))
        self._tags = sorted(t)

        self._params_set = True

    def _adjust_params(self):
        """Check if the parameter values have changed and set them if so."""
        if not self._params_set:
            self._set_params()
            return True
        else:
            return False

    def set_raster_length(self, raster_length):
        """Set a new raster length."""
        self._next_raster_length = raster_length
        self._params_set = False

    def set_tags(self, tags):
        """Set new parameters for all of the tags to be added."""
        self._next_tags = tags
        self._params_set = False

    def work(self, input_items, output_items):
        """Perform the block tasks on given input and output buffers."""
        in_arr = input_items[0]
        out_arr = output_items[0]
        noutput_items = len(out_arr)
        nwritten = self.nitems_written(0)

        # check if params changed, adjust and restart work if they have
        if self._adjust_params():
            return 0

        # noutput_items is a multiple of self._select_length because we set
        # output_multiple to be self._raster_length
        nrasters = noutput_items // self._raster_length

        # copy data
        out_arr[...] = in_arr[: nrasters * self._raster_length]

        # add tags
        for k_raster in range(nrasters):
            out_idx = k_raster * self._raster_length
            for raster_offset, name, val in self._tags:
                self.add_item_tag(0, nwritten + out_idx + raster_offset, name, val)

        return noutput_items
