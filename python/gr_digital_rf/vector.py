# ----------------------------------------------------------------------------
# Copyright (c) 2018 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module defining vector tools for GNU Radio."""
from __future__ import absolute_import, division, print_function

import numpy as np
from gnuradio import gr

__all__ = ("vector_aggregate",)


class vector_aggregate(gr.basic_block):
    """Block for aggregate consecutive vectors together using an operation."""

    def __init__(
        self,
        dtype=np.complex64,
        vlen=1,
        nagg=1,
        agg_op="mean",
        agg_op_args=(),
        max_nagg=None,
    ):
        """Aggregate consecutive vectors together using a specified operation.

        Parameters
        ----------
        dtype : np.dtype
            Data type of the input and output data.

        vlen : int
            Vector length of the input and output data.

        nagg : int
            Number of output vectors to aggregate together. The output is thus
            downsampled by `nagg` in whole vector chunks.

        agg_op : str
            String giving the name of a numpy array method to use for the
            aggregation operation. For `nagg` output vectors organized as an
            ``(nagg, vlen)``-shaped array called ``vectors``, the aggregation
            operation would then be ``vectors.agg_op(*agg_op_args, axis=0)``.

        agg_op_args : tuple
            Positional arguments to be passed to the aggregation operation
            method specified by `agg_op`. See above.


        Other Parameters
        ----------------
        max_nagg : int
            Maximum possible output aggregation, to allow for changes while the
            block is running. Knowing the maximum aggregation size allows for
            allocation of appropriately-sized buffers. If None, a default of
            four times the initial `nagg` will be used.

        """
        if max_nagg is None:
            max_nagg = 4 * nagg
        gr.basic_block.__init__(
            self,
            name="Vector Aggregate",
            in_sig=[(dtype, vlen)],
            out_sig=[(dtype, vlen)],
        )
        self._dtype = dtype
        self._vlen = vlen
        self._max_nagg = max_nagg
        self.set_agg_op(agg_op)
        self.set_agg_op_args(agg_op_args)

        # set parameters using max values to allocate maximum buffers
        self.set_nagg(max_nagg)
        self._set_params()

        # now set parameters to their true values
        self.set_nagg(nagg)

    def _set_params(self):
        """Finalize given parameter values and calculate derived values."""
        self._nagg = max(1, min(self._next_nagg, self._max_nagg))
        self.set_relative_rate(1 / float(self._nagg))
        self._params_set = True

    def _adjust_params(self):
        """Check if the parameter values have changed and set them if so."""
        if not self._params_set:
            self._set_params()
            return True
        else:
            return False

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
        ninput_items_required[0] = noutput_items * self._nagg

    def general_work(self, input_items, output_items):
        """Perform the block tasks on given input and output buffers."""
        in_arr = input_items[0].reshape((-1, self._vlen))
        out_arr = output_items[0].reshape((-1, self._vlen))
        noutput_items = len(out_arr)

        if self._adjust_params():
            return 0

        nconsumed = noutput_items * self._nagg
        in_agg = in_arr[:nconsumed].reshape((noutput_items, self._nagg, self._vlen))
        op_method = getattr(in_agg, self._agg_op)
        out_arr[...] = op_method(*self._agg_op_args, axis=1)
        self.consume(0, nconsumed)
        return noutput_items
