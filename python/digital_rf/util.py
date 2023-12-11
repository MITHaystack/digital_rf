# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Utility functions for Digital RF and Digital Metadata."""
from __future__ import absolute_import, division, print_function

import ast
import datetime
import fractions
import warnings

import dateutil.parser
import numpy as np

import six

__all__ = (
    "datetime_to_timedelta_tuple",
    "datetime_to_timestamp",
    "epoch",
    "get_samplerate_frac",
    "parse_identifier_to_sample",
    "parse_identifier_to_time",
    "sample_to_time_floor",
    "sample_to_datetime",
    "samples_to_timedelta",
    "time_to_sample",
    "time_to_sample_ceil",
)


_default_epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
epoch = _default_epoch


def get_samplerate_frac(sr_or_numerator, denominator=None):
    """Convert argument sample rate to a rational Fraction.

    Arguments are passed directly to the fractions.Fraction class, and the denominator
    of the result is limited to 32 bits.

    Parameters
    ----------
    sr_or_numerator : int | float | numpy.number | Rational | Decimal | str
        Sample rate in Hz, or the numerator of the sample rate if `denominator` is
        given. Most numeric types are accepted, falling back to evaluating the argument
        as a string if passing directly to fractions.Fraction fails. String arguments
        can represent the sample rate or a rational expression like "123/456".

    denominator: int | Rational, optional
        Denominator of the sample rate in Hz, if not None. Must be an integer or
        a Rational type, as expected by the `denominator` argument of
        fractions.Fraction.


    Returns
    -------
    frac : fractions.Fraction
        Rational representation of the sample rate.

    """
    try:
        frac = fractions.Fraction(sr_or_numerator, denominator)
    except TypeError:
        # try converting sr to str, then to fraction (works for np.longdouble)
        sr_frac = fractions.Fraction(str(sr_or_numerator))
        frac = fractions.Fraction(sr_frac, denominator)
    return frac.limit_denominator(2**32)


def time_to_sample_ceil(timedelta, sample_rate):
    """Convert a timedelta into a number of samples using a given sample rate.

    Ceiling rounding is used so that the value is the whole number of samples
    that spans *at least* the given `timedelta` but no more than
    ``timedelta + 1 / sample_rate``. This complements the flooring in
    `sample_to_time_floor`, so that::

        time_to_sample_ceil(sample_to_time_floor(sample, sr), sr) == sample


    Parameters
    ----------
    timedelta : (second, picosecond) tuple | np.timedelta64 | datetime.timedelta | float
        Time span to convert to a number of samples. To represent large time spans
        with high accuracy, pass a 2-tuple of ints containing the number of whole
        seconds and additional picoseconds. Float values are interpreted as a
        number of seconds.

    sample_rate : fractions.Fraction | first argument to ``get_samplerate_frac``
        Sample rate in Hz.


    Returns
    -------
    nsamples : int
        Number of samples in the `timedelta` time span at a rate of
        `sample_rate`, using ceiling rounding (up to the next whole sample).

    """
    if isinstance(timedelta, tuple):
        t_sec, t_psec = timedelta
    elif isinstance(timedelta, np.timedelta64):
        onesec = np.timedelta64(1, "s")
        t_sec = timedelta // onesec
        t_psec = (timedelta % onesec) // np.timedelta64(1, "ps")
    elif isinstance(timedelta, datetime.timedelta):
        t_sec = int(timedelta.total_seconds())
        t_psec = 1000000 * timedelta.microseconds
    else:
        t_sec = int(timedelta)
        t_psec = int(np.round((timedelta % 1) * 1e12))
    # ensure that sample_rate is a fractions.Fraction
    if not isinstance(sample_rate, fractions.Fraction):
        sample_rate = get_samplerate_frac(sample_rate)
    # calculate rational values for the second and picosecond parts
    s_frac = t_sec * sample_rate + t_psec * sample_rate / 10**12
    # get an integer value through ceiling rounding
    return int(s_frac) + ((s_frac % 1) != 0)


def sample_to_time_floor(nsamples, sample_rate):
    """Convert a number of samples into a timedelta using a given sample rate.

    Floor rounding is used so that the given whole number of samples spans
    *at least* the returned amount of time, accurate to the picosecond.
    This complements the ceiling rounding in `time_to_sample_ceil`, so that::

        time_to_sample_ceil(sample_to_time_floor(sample, sr), sr) == sample


    Parameters
    ----------
    nsamples : int
        Whole number of samples to convert into a span of time.

    sample_rate : fractions.Fraction | first argument to ``get_samplerate_frac``
        Sample rate in Hz.


    Returns
    -------
    seconds : int
        Number of whole seconds in the time span covered by `nsamples` at a rate
        of `sample_rate`.

    picoseconds : int
        Number of additional picoseconds in the time span covered by `nsamples`,
        using floor rounding (down to the previous whole number of picoseconds).

    """
    nsamples = int(nsamples)
    # ensure that sample_rate is a fractions.Fraction
    if not isinstance(sample_rate, fractions.Fraction):
        sample_rate = get_samplerate_frac(sample_rate)

    # get the timedelta as a Fraction
    t_frac = nsamples / sample_rate

    seconds = int(t_frac)
    picoseconds = int((t_frac % 1) * 10**12)

    return (seconds, picoseconds)


def time_to_sample(time, samples_per_second):
    """Get a sample index from a time using a given sample rate.

    Parameters
    ----------
    time : datetime | float
        Time corresponding to the desired sample index. If not given as a
        datetime object, the numeric value is interpreted as a UTC timestamp
        (seconds since epoch).

    samples_per_second : np.longdouble
        Sample rate in Hz.


    Returns
    -------
    sample_index : int
        Index to the identified sample given in the number of samples since
        the epoch (time_since_epoch*sample_per_second).

    """
    warnings.warn(
        "`time_to_sample` is deprecated. Use `time_to_sample_ceil` instead in"
        " combination with `datetime_to_timedelta_tuple` if necessary.",
        DeprecationWarning,
    )
    if isinstance(time, datetime.datetime):
        if time.tzinfo is None:
            # assume UTC if timezone was not specified
            time = time.replace(tzinfo=datetime.timezone.utc)
        td = time - epoch
        tsec = int(td.total_seconds())
        tfrac = 1e-6 * td.microseconds
        tidx = int(np.uint64(tsec * samples_per_second + tfrac * samples_per_second))
        return tidx
    else:
        return int(np.uint64(time * samples_per_second))


def sample_to_datetime(sample, sample_rate, epoch=None):
    """Get datetime corresponding to the given sample index.

    Parameters
    ----------

    sample : int
        Sample index in number of samples since epoch.

    sample_rate : fractions.Fraction | first argument to ``get_samplerate_frac``
        Sample rate in Hz.

    epoch : datetime, optional
        Epoch time. If None, the Digital RF default (the Unix epoch,
        January 1, 1970) is used.


    Returns
    -------

    dt : datetime
        Datetime corresponding to the given sample index.

    """
    if epoch is None:
        epoch = _default_epoch
    seconds, picoseconds = sample_to_time_floor(sample, sample_rate)
    td = datetime.timedelta(seconds=seconds, microseconds=picoseconds // 1000000)
    return epoch + td


def samples_to_timedelta(samples, samples_per_second):
    """Get timedelta for a duration in number of samples given a sample rate.

    Parameters
    ----------

    samples : int
        Duration in number of samples.

    samples_per_second : np.longdouble
        Sample rate in Hz.


    Returns
    -------

    td : datetime.timedelta
        Timedelta corresponding to the number of samples.

    """
    warnings.warn(
        "`samples_to_timedelta` is deprecated. Use `sample_to_time_floor` instead"
        " and create a timedelta object if necessary:"
        " `datetime.timedelta(seconds=seconds, microseconds=picoseconds // 1000000)`",
        DeprecationWarning,
    )
    # splitting into secs/frac lets us get a more accurate datetime
    secs = int(samples // samples_per_second)
    frac = (samples % samples_per_second) / samples_per_second
    microseconds = int(np.uint64(frac * 1000000))

    return datetime.timedelta(seconds=secs, microseconds=microseconds)


def datetime_to_timedelta_tuple(dt, epoch=None):
    """Return timedelta (seconds, picoseconds) tuple from epoch for a datetime object.

    Parameters
    ----------

    dt : datetime
        Time specified as a datetime object.

    epoch : datetime, optional
        Epoch time for converting absolute `dt` value to a number of seconds
        since `epoch`. If None, the Digital RF default (the Unix epoch,
        January 1, 1970) is used.


    Returns
    -------

    ts : float
        Time stamp (seconds since epoch).

    """
    if dt.tzinfo is None:
        # assume UTC if timezone was not specified
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    if epoch is None:
        epoch = _default_epoch
    timedelta = dt - epoch
    seconds = timedelta.seconds
    picoseconds = timedelta.microseconds * 1000000
    return (seconds, picoseconds)


def datetime_to_timestamp(dt, epoch=None):
    """Return time stamp (seconds since epoch) for a given datetime object.

    Parameters
    ----------

    dt : datetime
        Time specified as a datetime object.

    epoch : datetime, optional
        Epoch time for converting absolute `dt` value to a number of seconds
        since `epoch`. If None, the Digital RF default (the Unix epoch,
        January 1, 1970) is used.


    Returns
    -------

    ts : float
        Time stamp (seconds since epoch).

    """
    if dt.tzinfo is None:
        # assume UTC if timezone was not specified
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    if epoch is None:
        epoch = _default_epoch
    return (dt - epoch).total_seconds()


def parse_identifier_to_sample(iden, sample_rate=None, ref_index=None, epoch=None):
    """Get a sample index from different forms of identifiers.

    Parameters
    ----------

    iden : None | int | float | string | datetime
        If None or '', None is returned to indicate that the index should
        be automatically determined.
        If an integer, it is returned as the sample index.
        If a float, it is interpreted as a UTC timestamp (seconds since epoch)
        and the corresponding sample index is returned.
        If a string, four forms are permitted:
            1) a string which can be evaluated to an integer/float and
                interpreted as above,
            2) a string beginning with '+' and followed by an integer
                (float) expression, interpreted as samples (seconds) from
                `ref_index`, and
            3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'
            4) 'now' ('nowish'), indicating the current time (rounded up)

    sample_rate : fractions.Fraction | first argument to ``get_samplerate_frac``
        Sample rate in Hz used to convert a time to a sample index. Required
        when `iden` is given as a float or a time.

    ref_index : int
        Reference index from which string `iden` beginning with '+' are
        offset. Required when `iden` is a string that begins with '+'.

    epoch : datetime, optional
        Epoch time to use in converting an `iden` representing an absolute time.
        If None, the Digital RF default (the Unix epoch, January 1, 1970) is used.


    Returns
    -------

    sample_index : int | None
        Index to the identified sample given in the number of samples since
        the epoch (time_since_epoch*sample_per_second).

    """
    is_relative = False
    if iden is None or iden == "":
        return None
    elif isinstance(iden, six.string_types):
        if iden.startswith("+"):
            is_relative = True
            iden = iden.lstrip("+")
        try:
            # int or float
            iden = ast.literal_eval(iden)
        except (ValueError, SyntaxError):
            if is_relative:
                raise ValueError(
                    '"+" identifier must be followed by an integer or float.'
                )
            if iden.lower().startswith("now"):
                dt = datetime.datetime.now(tz=datetime.timezone.utc)
                if iden.lower().endswith("ish"):
                    dt = dt.replace(microsecond=0) + datetime.timedelta(seconds=1)
                iden = dt
            else:
                # parse to datetime
                iden = dateutil.parser.parse(iden)

    if not isinstance(iden, six.integer_types):
        if sample_rate is None:
            raise ValueError("sample_rate required when time identifier is used.")
        if epoch is None:
            epoch = _default_epoch
        if isinstance(iden, datetime.datetime):
            iden = iden - epoch
        elif not is_relative:
            # interpret float time as timestamp from unix epoch, so adjust from
            # unix epoch to specified sample epoch
            iden -= (
                epoch - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
            ).total_seconds()
        idx = time_to_sample_ceil(iden, sample_rate)
    else:
        idx = iden

    if is_relative:
        if ref_index is None:
            raise ValueError('ref_index required when relative "+" identifier is used.')
        return idx + ref_index
    else:
        return idx


def parse_identifier_to_time(iden, sample_rate=None, ref_datetime=None, epoch=None):
    """Get a time from different forms of identifiers.

    Parameters
    ----------

    iden : None | float | string | int
        If None or '', None is returned to indicate that the time should
        be automatically determined.
        If a float, it is interpreted as a UTC timestamp (seconds since epoch)
        and the corresponding datetime is returned.
        If an integer, it is interpreted as a sample index when
        `sample_rate` is not None and a UTC timestamp otherwise.
        If a string, four forms are permitted:
            1) a string which can be evaluated to an integer/float and
                interpreted as above,
            2) a string beginning with '+' and followed by an integer
                (float) expression, interpreted as samples (seconds) from
                `ref_time`, and
            3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'
            4) 'now' ('nowish'), indicating the current time (rounded up)

    sample_rate : fractions.Fraction | first argument to ``get_samplerate_frac``
        Sample rate in Hz used to convert a sample index to a time. Required
        when `iden` is given as an integer.

    ref_datetime : datetime, required for '+' string form of `iden`
        Reference time from which string `iden` beginning with '+' are
        offset. Must be timezone-aware.

    epoch : datetime, optional
        Epoch time to use in converting an `iden` representing a sample index.
        If None, the Digital RF default (the Unix epoch, January 1, 1970) is used.


    Returns
    -------

    dt : datetime
        Datetime object giving the indicated time.

    """
    is_relative = False
    if iden is None or iden == "":
        return None
    elif isinstance(iden, six.string_types):
        if iden.startswith("+"):
            is_relative = True
            iden = iden.lstrip("+")
        try:
            # int or float
            iden = ast.literal_eval(iden)
        except (ValueError, SyntaxError):
            if is_relative:
                raise ValueError(
                    '"+" identifier must be followed by an integer or float.'
                )
            if iden.lower().startswith("now"):
                dt = datetime.datetime.now(tz=datetime.timezone.utc)
                if iden.lower().endswith("ish"):
                    dt = dt.replace(microsecond=0) + datetime.timedelta(seconds=1)
            else:
                # parse string to datetime
                dt = dateutil.parser.parse(iden)
                if dt.tzinfo is None:
                    # assume UTC if timezone was not specified in the string
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt

    if isinstance(iden, float) or sample_rate is None:
        td = datetime.timedelta(seconds=iden)
        # timestamp is relative to unix epoch always
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        seconds, picoseconds = sample_to_time_floor(iden, sample_rate)
        td = datetime.timedelta(seconds=seconds, microseconds=picoseconds // 1000000)
        # identifier is a sample converted to a timedelta, now it should be
        # converted to an absolute time using the specified sample epoch
        if epoch is None:
            epoch = _default_epoch

    if is_relative:
        if ref_datetime is None:
            raise ValueError(
                'ref_datetime required when relative "+" identifier is used.'
            )
        elif (
            not isinstance(ref_datetime, datetime.datetime)
            or ref_datetime.tzinfo is None
        ):
            raise ValueError("ref_datetime must be a timezone-aware datetime.")
        return td + ref_datetime
    else:
        return td + epoch
