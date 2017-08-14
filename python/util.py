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

__all__ = (
    'epoch', 'parse_identifier_to_sample', 'parse_identifier_to_time',
    'sample_to_datetime', 'samples_to_timedelta',
)


epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


def parse_identifier_to_sample(iden, samples_per_second=None, ref_index=None):
    """Get a sample index from different forms of identifiers.

    Parameters
    ----------

    iden : None | int/long | float | string
        If None or '', None is returned to indicate that the index should
        be automatically determined.
        If an integer, it is returned as the sample index.
        If a float, it is interpreted as a UTC timestamp (seconds since epoch)
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
            if is_relative:
                raise ValueError(
                    '"+" identifier must be followed by an integer or float.'
                )
            # convert datetime to float
            dt = dateutil.parser.parse(iden)
            if dt.tzinfo is None:
                # assume UTC if timezone was not specified in the string
                dt = pytz.utc.localize(dt)
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


def parse_identifier_to_time(iden, samples_per_second=None, ref_datetime=None):
    """Get a time from different forms of identifiers.

    Parameters
    ----------

    iden : None | float | string | int/long
        If None or '', None is returned to indicate that the time should
        be automatically determined.
        If a float, it is interpreted as a UTC timestamp (seconds since epoch)
        and the corresponding datetime is returned.
        If an integer, it is interpreted as a sample index when
        `samples_per_second` is not None and a UTC timestamp otherwise.
        If a string, three forms are permitted:
            1) a string which can be evaluated to an integer/float and
                interpreted as above,
            2) a string beginning with '+' and followed by an integer
                (float) expression, interpreted as samples (seconds) from
                `ref_time`, and
            3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

    samples_per_second : numpy.longdouble, required for integer `iden`
        Sample rate in Hz used to convert a sample index to a time.

    ref_datetime : datetime, required for '+' string form of `iden`
        Reference time from which string `iden` beginning with '+' are
        offset. Must be timezone-aware.


    Returns
    -------

    dt : datetime
        Datetime object giving the indicated time.

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
            if is_relative:
                raise ValueError(
                    '"+" identifier must be followed by an integer or float.'
                )
            # parse string to datetime
            dt = dateutil.parser.parse(iden)
            if dt.tzinfo is None:
                # assume UTC if timezone was not specified in the string
                dt = pytz.utc.localize(dt)
            return dt

    if isinstance(iden, float) or samples_per_second is None:
        td = datetime.timedelta(seconds=iden)
    else:
        td = samples_to_timedelta(iden, samples_per_second)

    if is_relative:
        if ref_datetime is None:
            raise ValueError(
                'ref_datetime required when relative "+" identifier is used.'
            )
        elif (not isinstance(ref_datetime, datetime.datetime)
                or ref_datetime.tzinfo is None):
            raise ValueError(
                'ref_datetime must be a timezone-aware datetime.'
            )
        return td + ref_datetime
    else:
        return td + epoch


def sample_to_datetime(sample, samples_per_second):
    """Get datetime corresponding to the given sample index.

    Parameters
    ----------

    sample : int/long
        Sample index in number of samples since epoch.

    samples_per_second : numpy.longdouble
        Sample rate in Hz.


    Returns
    -------

    dt : datetime
        Datetime corresponding to the given sample index.

    """
    return epoch + samples_to_timedelta(sample, samples_per_second)


def samples_to_timedelta(samples, samples_per_second):
    """Get timedelta for a duration in number of samples given a sample rate.

    Parameters
    ----------

    samples : int/long
        Duration in number of samples.

    samples_per_second : numpy.longdouble
        Sample rate in Hz.


    Returns
    -------

    td : datetime.timedelta
        Timedelta corresponding to the number of samples.

    """
    # splitting into secs/frac lets us get a more accurate datetime
    secs = int(samples // samples_per_second)
    frac = (samples % samples_per_second) / samples_per_second
    microseconds = int(np.uint64(frac*1000000))

    return datetime.timedelta(seconds=secs, microseconds=microseconds)
