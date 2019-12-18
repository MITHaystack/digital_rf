# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for listing Digital RF/Metadata files in a directory."""
from __future__ import absolute_import, division, print_function

import bisect
import datetime
import os
import re
import shutil

import pytz

from . import util

__all__ = (
    "GLOB_DMDFILE",
    "GLOB_DMDPROPFILE",
    "GLOB_DRFFILE",
    "GLOB_DRFPROPFILE",
    "GLOB_SUBDIR",
    "RE_DMD",
    "RE_DMDPROP",
    "RE_DRF",
    "RE_DRFDMD",
    "RE_DRFDMDPROP",
    "RE_DRFPROP",
    "ilsdrf",
    "lsdrf",
    "sortkey_drf",
)

# timestamped subdirectory, e.g. 2016-09-21T20-00-00
GLOB_SUBDIR = (
    "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]" "T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
)
RE_SUBDIR = (
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2})-(?P<minute>[0-9]{2})-(?P<second>[0-9]{2})"
)
_RE_SUBDIR = re.compile("^" + RE_SUBDIR + "$")

# Digital RF file, e.g. rf@1474491360.000.h5
GLOB_DRFFILE = "*@[0-9]*.[0-9][0-9][0-9].h5"
RE_DRFFILE = r"(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.(?P<frac>[0-9]{3})\.h5$"
_RE_DRFFILE = re.compile("^" + RE_DRFFILE)

# Digital Metadata file, e.g. metadata@1474491360.h5
GLOB_DMDFILE = "*@[0-9]*.h5"
RE_DMDFILE = r"(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.h5$"
_RE_DMDFILE = re.compile("^" + RE_DMDFILE)

# either Digital RF or Digital Metadata file
RE_FILE = r"(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)(?:\.(?P<frac>[0-9]{3}))?\.h5$"
_RE_FILE = re.compile("^" + RE_FILE)

# Digital RF channel properties file, e.g. drf_properties.h5
# include pre-2.5-style metadata.h5 for now
GLOB_DRFPROPFILE = "*.h5"
RE_DRFPROPFILE = r"(?P<name>drf_properties|metadata)\.h5$"
_RE_DRFPROPFILE = re.compile("^" + RE_DRFPROPFILE)

# Digital RF channel properties file, e.g. dmd_properties.h5
# include pre-2.5-style metadata.h5 for now
GLOB_DMDPROPFILE = "*.h5"
RE_DMDPROPFILE = r"(?P<name>dmd_properties|metadata)\.h5$"
_RE_DMDPROPFILE = re.compile("^" + RE_DMDPROPFILE)

# either Digital RF or Digital Metadata channel properties file
RE_PROPFILE = r"(?P<name>(?:drf|dmd)_properties|metadata)\.h5$"
_RE_PROPFILE = re.compile("^" + RE_PROPFILE)

# Digital RF file in correct subdirectory structure
RE_DRF = re.escape(os.sep).join((r"(?P<chpath>.*?)", RE_SUBDIR, RE_DRFFILE))
# Digital Metadata file in correct subdirectory structure
RE_DMD = re.escape(os.sep).join((r"(?P<chpath>.*?)", RE_SUBDIR, RE_DMDFILE))
# either Digital RF or Digital Metadata in its correct subdirectory
RE_DRFDMD = re.escape(os.sep).join((r"(?P<chpath>.*?)", RE_SUBDIR, RE_FILE))
# properties file associated with Digital RF directory
RE_DRFPROP = re.escape(os.sep).join((r"(?P<chpath>.*?)", RE_DRFPROPFILE))
# properties file associated with Digital Metadata directory
RE_DMDPROP = re.escape(os.sep).join((r"(?P<chpath>.*?)", RE_DMDPROPFILE))
# properties file associated with Digital RF or Digital Metadata directory
RE_DRFDMDPROP = re.escape(os.sep).join((r"(?P<chpath>.*?)", RE_PROPFILE))


def sortkey_drf(filename, regexes=None):
    """Get key for a Digital RF filename to sort first by sample time."""
    if regexes is None:
        regexes = [_RE_FILE]
    for r in regexes:
        m = r.match(filename)
        if m:
            try:
                secs = int(m.group("secs"))
            except (IndexError, TypeError):
                # regex matched but there is no 'secs' in regex
                secs = 0
            try:
                frac = int(m.group("frac"))
            except (IndexError, TypeError):
                # regex matched but there is no 'frac' in regex
                frac = 0
            try:
                name = m.group("name")
            except (IndexError, TypeError):
                # regex matched but there is no 'name' in regex
                name = filename
            return (secs * 1000 + frac, name)
    return None


def _decorate_drf_files(subdir, filenames, file_regex):
    """Match Digital RF/Metadata filenames and decorate into (time, f)."""
    dec_files = []
    for filename in filenames:
        m = file_regex.match(filename)
        if m:
            try:
                frac = int(m.group("frac"))
            except (IndexError, TypeError):
                # regex matched but there is no 'frac' in regex (metadata)
                frac = 0
            time = datetime.timedelta(seconds=int(m.group("secs")), milliseconds=frac)
            dec_files.append((time, os.path.join(subdir, filename)))
    return dec_files


def _decorated_list_slice(dec_list, starttime=None, endtime=None, ffill=False):
    """Get slice for sorted list of tuples with (time, ...)."""
    ks = 0
    ke = len(dec_list)
    if starttime is not None:
        # (because of tuple comparison, all bisects are left)
        # dec_list[:ks][0] < starttime <= dec_list[ks:][0]
        ks = bisect.bisect_left(dec_list, (starttime,))
        # forward fill, first index where dec_list[ks][0] <= starttime
        if ffill and (ks == len(dec_list) or dec_list[ks][0] > starttime):
            ks = max(ks - 1, 0)
    if endtime is not None:
        # (because of tuple comparison, all bisects are left)
        # dec_list[:ke][0] < endtime <= dec_list[ke:][0]
        ke = bisect.bisect_left(dec_list, (endtime,), lo=ks)
        # make ke first index to exclude
        if ke < len(dec_list) and dec_list[ke][0] == endtime:
            ke = ke + 1
    return slice(ks, ke)


def _yield_matching_files(
    root,
    dirs,
    props,
    include_drf,
    include_dmd,
    starttime=None,
    endtime=None,
    reverse=False,
):
    """Yield matching files from a list of subdirectories in a channel dir."""
    yielding_drf_channel = any(_RE_DRFPROPFILE.match(f) for f in props) and include_drf
    yielding_dmd_channel = any(_RE_DMDPROPFILE.match(f) for f in props) and include_dmd
    if yielding_drf_channel and yielding_dmd_channel:
        file_regex = _RE_FILE
    elif yielding_drf_channel:
        file_regex = _RE_DRFFILE
    elif yielding_dmd_channel:
        file_regex = _RE_DMDFILE
    else:
        # not in a channel that we want to include
        return
    # get time-stamped subdirectories from dirs list
    dec_subdirs = []
    others = []
    for d in dirs:
        m = _RE_SUBDIR.match(d)
        if m:
            dt = datetime.datetime(
                year=int(m.group("year")),
                month=int(m.group("month")),
                day=int(m.group("day")),
                hour=int(m.group("hour")),
                minute=int(m.group("minute")),
                second=int(m.group("second")),
                tzinfo=pytz.utc,
            )
            time = dt - util.epoch
            dec_subdirs.append((time, d))
        else:
            others.append(d)
    # limit list of dirs for recursion by modifying in place
    dirs[:] = others

    # filter subdirectories by time
    dec_subdirs.sort()
    subdir_slice = _decorated_list_slice(
        dec_subdirs, starttime=starttime, endtime=endtime, ffill=True
    )

    for k, (_time, subdir) in (
        enumerate(dec_subdirs[subdir_slice])
        if not reverse
        else enumerate(list(reversed(dec_subdirs[subdir_slice])))
    ):
        # list potential files and get groups of all matching files
        try:
            subdir_files = os.listdir(os.path.join(root, subdir))
        except OSError:
            # directory failed to list (e.g. doesn't exist anymore), skip
            continue
        dec_files = _decorate_drf_files(
            os.path.join(root, subdir), subdir_files, file_regex
        )
        dec_files.sort()
        if (
            (k == 0)
            and yielding_dmd_channel
            and subdir_slice.start > 0
            and starttime
            and dec_files[0][0] > starttime
        ):
            # need to include files from earlier subdir so we can
            # forward fill and include the metadata file prior to starttime
            for k_subdir in range(subdir_slice.start - 1, -1, -1):
                prior_subdir = dec_subdirs[k_subdir][-1]
                prior_files = os.listdir(os.path.join(root, prior_subdir))
                dec_prior_files = _decorate_drf_files(
                    os.path.join(root, prior_subdir), prior_files, file_regex
                )
                if dec_prior_files:
                    dec_prior_files.extend(dec_files)
                    dec_files = dec_prior_files
                    dec_files.sort()
                    break

        # filter by time and yield files
        # forward fill if metadata since we assume metadata for an older
        # time is valid until the next metadata sample
        slc = _decorated_list_slice(
            dec_files,
            starttime=starttime,
            endtime=endtime,
            ffill=(k == 0) and yielding_dmd_channel,
        )
        for dec_file in dec_files[slc] if not reverse else reversed(dec_files[slc]):
            yield dec_file[-1]


def ilsdrf(
    path,
    recursive=True,
    reverse=False,
    starttime=None,
    endtime=None,
    include_drf=True,
    include_dmd=True,
    include_drf_properties=None,
    include_dmd_properties=None,
):
    """Yield Digital RF/Metadata files contained in a channel directory.

    Sub-directories will be traversed in alphabetical order and files from each
    directory will be yielded in Digital RF (time) order.


    Parameters
    ----------
    path : string
        Parent directory to list.

    recursive : bool
        If True, also list files recursively in channel subdirectories.

    reverse : bool
        If True, traverse directories and list files in those directories in
        reversed order.

    starttime : datetime.datetime
        Data covering this time or after will be included. This has no effect
        on property files.

    endtime : datetime.datetime
        Data covering this time or earlier will be included. This has no effect
        on property files.

    include_drf : bool
        If True, include Digital RF files in listing.

    include_dmd : bool
        If True, include Digital Metadata files in listing.

    include_drf_properties : bool | None
        If True, include the Digital RF properties file in listing.
        If None, use `include_drf` value.

    include_dmd_properties : bool | None
        If True, include the Digital Metadata properties file in listing.
        If None, use `include_dmd` value.


    Yields
    ------
    Digital RF/Metadata files contained in `path`.

    """
    # convert starttime and endtime to timedeltas for comparison
    if starttime is not None:
        if starttime.tzinfo is None:
            starttime = pytz.utc.localize(starttime)
        starttime = starttime - util.epoch
    if endtime is not None:
        if endtime.tzinfo is None:
            endtime = pytz.utc.localize(endtime)
        endtime = endtime - util.epoch

    if include_drf_properties is None:
        include_drf_properties = include_drf
    if include_dmd_properties is None:
        include_dmd_properties = include_dmd

    include_properties = True
    if include_drf_properties and include_dmd_properties:
        prop_regex = _RE_PROPFILE
    elif include_drf_properties:
        prop_regex = _RE_DRFPROPFILE
    elif include_dmd_properties:
        prop_regex = _RE_DMDPROPFILE
    else:
        include_properties = False

    path = os.path.abspath(path)
    if include_drf or include_dmd:
        # check if path is already a time-stamped subdir with files,
        # and yield the files if so
        root, subdir = os.path.split(path)
        if _RE_SUBDIR.match(subdir):
            any_props = [f for f in os.listdir(root) if _RE_PROPFILE.match(f)]
            if any_props:
                for f in _yield_matching_files(
                    root,
                    [subdir],
                    any_props,
                    include_drf,
                    include_dmd,
                    starttime=starttime,
                    endtime=endtime,
                    reverse=reverse,
                ):
                    yield f

    # now search for channel directories, starting with path
    for root, dirs, files in os.walk(path):
        # determine if root is a channel directory (i.e. has property file)
        any_props = [f for f in files if _RE_PROPFILE.match(f)]
        if any_props:
            if include_properties:
                # first yield property files
                props = [
                    os.path.join(root, f) for f in any_props if prop_regex.match(f)
                ]
                props.sort(reverse=reverse)
                for prop in props:
                    yield prop

            if include_drf or include_dmd:
                # list, match, filter, and yield files from subdirectories
                for f in _yield_matching_files(
                    root,
                    dirs,
                    any_props,
                    include_drf,
                    include_dmd,
                    starttime=starttime,
                    endtime=endtime,
                    reverse=reverse,
                ):
                    yield f

        if recursive:
            # walk through (non-ts-subdir) directories in sorted order
            dirs.sort(reverse=reverse)
        else:
            # don't go further if we're not recursive
            del dirs[:]


def lsdrf(*args, **kwargs):
    """Get list of Digital RF/Metadata files contained in a channel directory.

    Sub-directories will be traversed in alphabetical order and files from each
    directory will be yielded in Digital RF (time) order.

    Parameters
    ----------
    path : string
        Parent directory to list.

    recursive : bool
        If True, also list files recursively in channel subdirectories.

    reverse : bool
        If True, traverse directories and list files in those directories in
        reversed order.

    starttime : datetime.datetime
        Data covering this time or after will be included. This has no effect
        on property files.

    endtime : datetime.datetime
        Data covering this time or earlier will be included. This has no effect
        on property files.

    include_drf : bool
        If True, include Digital RF files in listing.

    include_dmd : bool
        If True, include Digital Metadata files in listing.

    include_drf_properties : bool | None
        If True, include the Digital RF properties file in listing.
        If None, use `include_drf` value.

    include_dmd_properties : bool | None
        If True, include the Digital Metadata properties file in listing.
        If None, use `include_dmd` value.


    Returns
    -------
    List of Digital RF/Metadata files contained in `path`.

    """
    return list(ilsdrf(*args, **kwargs))


def _add_time_group(parser):
    timegroup = parser.add_argument_group(title="time")
    timegroup.add_argument(
        "-s",
        "--starttime",
        dest="starttime",
        default=None,
        help="""Data covering this time or after will be included (no effect
                on property files). The time can be given as a datetime string
                (e.g. in ISO8601 format: 2016-01-01T15:24:00Z) or a float
                giving a Unix timestamp in seconds. (default: %(default)s)""",
    )
    timegroup.add_argument(
        "-e",
        "--endtime",
        dest="endtime",
        default=None,
        help="""Data covering this time or before will be included (no effect
                on property files). The time can be given as a datetime string
                (e.g. in ISO8601 format: 2016-01-01T15:24:00Z), a float
                giving a Unix timestamp in seconds, or a '+' followed by a time
                in seconds that will be taken relative to `starttime`.
                (default: %(default)s)""",
    )
    return parser


def _add_include_group(parser):
    includegroup = parser.add_argument_group(title="include")
    drfgroup = includegroup.add_mutually_exclusive_group()
    dmdgroup = includegroup.add_mutually_exclusive_group()
    drfpropsgroup = includegroup.add_mutually_exclusive_group()
    dmdpropsgroup = includegroup.add_mutually_exclusive_group()

    drfgroup.add_argument(
        "--drf",
        dest="include_drf",
        action="store_true",
        default=True,
        help="""Include Digital RF files.
                (default: True)""",
    )
    drfpropsgroup.add_argument(
        "--drfprops",
        dest="include_drf_properties",
        action="store_true",
        default=None,
        help="""Include drf_properties.h5 files. If unset, use value of --drf.
                (default: None)""",
    )
    drfgroup.add_argument(
        "--nodrf",
        dest="include_drf",
        action="store_false",
        help="""Do not include Digital RF files.
                (default: False)""",
    )
    drfpropsgroup.add_argument(
        "--nodrfprops",
        dest="include_drf_properties",
        action="store_false",
        help="""Do not include drf_properties.h5 files.
                (default: False)""",
    )
    dmdgroup.add_argument(
        "--dmd",
        dest="include_dmd",
        action="store_true",
        default=True,
        help="""Include Digital Metadata files.
                (default: True)""",
    )
    dmdpropsgroup.add_argument(
        "--dmdprops",
        dest="include_dmd_properties",
        action="store_true",
        default=None,
        help="""Include dmd_properties.h5 files. If unset, use value of --dmd.
                (default: None)""",
    )
    dmdgroup.add_argument(
        "--nodmd",
        dest="include_dmd",
        action="store_false",
        help="""Do not include Digital Metadata files.
                (default: False)""",
    )
    dmdpropsgroup.add_argument(
        "--nodmdprops",
        dest="include_dmd_properties",
        action="store_false",
        help="""Do not include dmd_properties.h5 files.
                (default: False)""",
    )
    return parser


def _build_ls_parser(Parser, *args):
    desc = "List Digital RF/Metadata files in a (channel) directory."
    parser = Parser(*args, description=desc)

    parser.add_argument(
        "dirs",
        nargs="*",
        default=".",
        help="""Data directory to list. (default: %(default)s)""",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="""Also list files recursively in channel subdirectories.
                (default: %(default)s)""",
    )
    parser.add_argument(
        "-R",
        "--reverse",
        action="store_true",
        help="""Traverse directories and list files in those directories in
                reversed order. (default: %(default)s)""",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="""Return absolute paths instead of relative paths.
                (default: %(default)s)""",
    )
    parser.add_argument(
        "--sortall",
        action="store_true",
        help="""Sort all files together before listing (normally only sorted
                within subdirectories). (default: %(default)s)""",
    )

    parser = _add_time_group(parser)
    parser = _add_include_group(parser)

    parser.set_defaults(func=_run_ls)

    return parser


def _run_ls(args):
    if args.starttime is not None:
        args.starttime = util.parse_identifier_to_time(args.starttime)
    if args.endtime is not None:
        args.endtime = util.parse_identifier_to_time(
            args.endtime, ref_datetime=args.starttime
        )

    if args.abs:

        def fixpath(path, start=None):
            return path

    else:
        fixpath = os.path.relpath

    kwargs = vars(args).copy()
    del kwargs["func"]
    del kwargs["dirs"]
    del kwargs["abs"]
    del kwargs["sortall"]

    if args.sortall:
        files = []
        for d in args.dirs:
            files.extend(lsdrf(d, **kwargs))
        files.sort(key=sortkey_drf)
        print("\n".join([fixpath(f, d) for f in files]))
    else:
        for d in args.dirs:
            for f in ilsdrf(d, **kwargs):
                print(fixpath(f, d))


def _add_srcdest_arguments(parser):
    parser.add_argument("src", help="Source directory.")
    parser.add_argument("dest", help="Destination directory.")

    parser.add_argument(
        "-c",
        "--channel",
        dest="chs",
        action="append",
        default=[],
        help="""Channel names to include from source directory.
                (default: Treat `src` as a channel directory)""",
    )
    parser.add_argument(
        "--only",
        dest="recursive",
        action="store_false",
        default=True,
        help="""Only copy the specified channel directory (do not recurse from
                `src` to find channel subdirectories).
                (default: False)""",
    )
    parser.add_argument(
        "-R",
        "--reverse",
        action="store_true",
        help="""Traverse directories and include files in those directories in
                reversed order. (default: %(default)s)""",
    )

    parser = _add_time_group(parser)
    parser = _add_include_group(parser)
    return parser


def _parse_srcdest_args(args):
    args.src = os.path.abspath(args.src)
    args.dest = os.path.abspath(args.dest)

    args.chs = [b.strip() for a in args.chs for b in a.strip().split(",")]
    args.srcdests = [
        (os.path.join(args.src, ch), os.path.join(args.dest, ch)) for ch in args.chs
    ]
    if not args.srcdests:
        args.srcdests = [(args.src, args.dest)]

    if args.starttime is not None:
        args.starttime = util.parse_identifier_to_time(args.starttime)
    if args.endtime is not None:
        args.endtime = util.parse_identifier_to_time(
            args.endtime, ref_datetime=args.starttime
        )

    kwargs = vars(args).copy()
    del kwargs["func"]
    del kwargs["src"]
    del kwargs["dest"]
    del kwargs["chs"]
    del kwargs["srcdests"]

    return args, kwargs


def _build_cp_parser(Parser, *args):
    desc = "Copy Digital RF/Metadata files from source to destination."
    parser = Parser(*args, description=desc)
    parser = _add_srcdest_arguments(parser)
    parser.set_defaults(func=_run_cp)
    return parser


def _run_cp(args):
    args, kwargs = _parse_srcdest_args(args)
    for (src, dest) in args.srcdests:
        for srcpath in ilsdrf(src, **kwargs):
            destpath = os.path.join(dest, os.path.relpath(srcpath, src))
            destdir = os.path.dirname(destpath)
            if not os.path.exists(destdir):
                os.makedirs(destdir)
            shutil.copy2(srcpath, destpath)


def _build_mv_parser(Parser, *args):
    desc = "Move Digital RF/Metadata files from source to destination."
    parser = Parser(*args, description=desc)
    parser = _add_srcdest_arguments(parser)
    parser.set_defaults(func=_run_mv)
    return parser


def _run_mv(args):
    args, kwargs = _parse_srcdest_args(args)
    for (src, dest) in args.srcdests:
        for srcpath in ilsdrf(src, **kwargs):
            destpath = os.path.join(dest, os.path.relpath(srcpath, src))
            destdir = os.path.dirname(destpath)
            if not os.path.exists(destdir):
                os.makedirs(destdir)
            shutil.move(srcpath, destpath)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ls_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
