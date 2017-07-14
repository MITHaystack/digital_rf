# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Utility functions for Digital RF and Digital Metadata."""
import ast
import datetime

import dateutil.parser
import numpy as np
import pytz
import six

__all__ = ('parse_sample_identifier',)


def parse_sample_identifier(iden, samples_per_second=None, ref_index=None):
    """Get a sample index from different forms of identifiers.

    Parameters
    ----------

    iden : None | int/long | float | string
        If None or '', None is returned to indicate that the index should
        be automatically determined.
        If an integer, it is returned as the sample index.
        If a float, it is interpreted as a timestamp (seconds since epoch)
        and the corresponding sample index is returned.
        If a string, three forms are permitted:
            1) a string which can be evaluated to an integer/float and
                interpreted as above,
            2) a string beginning with '+' and followed by an integer
                (float) expression, interpreted as samples (seconds) from
                `ref_index`, and
            3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

    samples_per_second : numpy.longdouble, required for float and time `iden`
        Sample rate in Hz used to convert a time to a sample index.

    ref_index : int/long, required for '+' string form of `iden`
        Reference index from which string `iden` beginning with '+' are
        offset.


    Returns
    -------

    sample_index : long | None
        Index to the identified sample given in the number of samples since
        the epoch (time_since_epoch*sample_per_second).

    """
    is_relative = False
    if iden is None or iden == '':
        return None
    elif isinstance(iden, six.string_types):
        if iden.startswith('+'):
            is_relative = True
            iden = iden.lstrip('+')
        try:
            # int/long or float
            iden = ast.literal_eval(iden)
        except (ValueError, SyntaxError):
            # convert datetime to float
            dt = dateutil.parser.parse(iden)
            epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            iden = (dt - epoch).total_seconds()

    if isinstance(iden, float):
        if samples_per_second is None:
            raise ValueError(
                'samples_per_second required when time identifier is used.'
            )
        idx = long(np.uint64(iden*samples_per_second))
    else:
        idx = long(iden)

    if is_relative:
        if ref_index is None:
            raise ValueError(
                'ref_index required when relative "+" identifier is used.'
            )
        return idx + ref_index
    else:
        return idx
