# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for listing Digital RF/Metadata files in a directory."""
import bisect
import datetime
import os
import re
from collections import defaultdict

import pytz

from . import util

__all__ = (
    'GLOB_DMDFILE', 'GLOB_DMDPROPFILE', 'GLOB_DRFFILE', 'GLOB_DRFPROPFILE',
    'GLOB_SUBDIR', 'RE_DMD', 'RE_DMDPROP', 'RE_DRF', 'RE_DRFDMD',
    'RE_DRFDMDPROP', 'RE_DRFPROP',
    'ilsdrf', 'lsdrf', 'sortkey_drf'
)

# timestamped subdirectory, e.g. 2016-09-21T20-00-00
GLOB_SUBDIR = (
    '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
    'T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
)
RE_SUBDIR = (
    r'(?P<ymd>[0-9]{4}-[0-9]{2}-[0-9]{2})'
    r'T(?P<hms>[0-9]{2}-[0-9]{2}-[0-9]{2})'
)
_RE_SUBDIR = re.compile('^' + RE_SUBDIR + '$')

# Digital RF file, e.g. rf@1474491360.000.h5
GLOB_DRFFILE = '*@[0-9]*.[0-9][0-9][0-9].h5'
RE_DRFFILE = (
    r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.(?P<frac>[0-9]{3})\.h5$'
)
_RE_DRFFILE = re.compile('^' + RE_DRFFILE)

# Digital Metadata file, e.g. metadata@1474491360.h5
GLOB_DMDFILE = '*@[0-9]*.h5'
RE_DMDFILE = r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.h5$'
_RE_DMDFILE = re.compile('^' + RE_DMDFILE)

# either Digital RF or Digital Metadata file
RE_FILE = (
    r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)(?:\.(?P<frac>[0-9]{3}))?\.h5$'
)
_RE_FILE = re.compile('^' + RE_FILE)

# Digital RF channel properties file, e.g. drf_properties.h5
# include pre-2.5-style metadata.h5 for now
GLOB_DRFPROPFILE = '*.h5'
RE_DRFPROPFILE = r'(?P<name>drf_properties|metadata)\.h5$'
_RE_DRFPROPFILE = re.compile('^' + RE_DRFPROPFILE)

# Digital RF channel properties file, e.g. dmd_properties.h5
# include pre-2.5-style metadata.h5 for now
GLOB_DMDPROPFILE = '*.h5'
RE_DMDPROPFILE = r'(?P<name>dmd_properties|metadata)\.h5$'
_RE_DMDPROPFILE = re.compile('^' + RE_DMDPROPFILE)

# either Digital RF or Digital Metadata channel properties file
RE_PROPFILE = r'(?P<name>(?:drf|dmd)_properties|metadata)\.h5$'
_RE_PROPFILE = re.compile('^' + RE_PROPFILE)

# Digital RF file in correct subdirectory structure
RE_DRF = re.escape(os.sep).join((r'(?P<chpath>.*?)', RE_SUBDIR, RE_DRFFILE))
# Digital Metadata file in correct subdirectory structure
RE_DMD = re.escape(os.sep).join((r'(?P<chpath>.*?)', RE_SUBDIR, RE_DMDFILE))
# either Digital RF or Digital Metadata in its correct subdirectory
RE_DRFDMD = re.escape(os.sep).join((r'(?P<chpath>.*?)', RE_SUBDIR, RE_FILE))
# properties file associated with Digital RF directory
RE_DRFPROP = re.escape(os.sep).join((r'(?P<chpath>.*?)', RE_DRFPROPFILE))
# properties file associated with Digital Metadata directory
RE_DMDPROP = re.escape(os.sep).join((r'(?P<chpath>.*?)', RE_DMDPROPFILE))
# properties file associated with Digital RF or Digital Metadata directory
RE_DRFDMDPROP = re.escape(os.sep).join((r'(?P<chpath>.*?)', RE_PROPFILE))


def sortkey_drf(filename, regexes=[re.compile(RE_FILE)]):
    """Get key for a Digital RF filename to sort first by sample time."""
    for r in regexes:
        m = r.match(filename)
        if m:
            try:
                secs = int(m.group('secs'))
            except (IndexError, TypeError):
                # regex matched but there is no 'secs' in regex
                secs = 0
            try:
                frac = int(m.group('frac'))
            except (IndexError, TypeError):
                # regex matched but there is no 'frac' in regex
                frac = 0
            try:
                name = m.group('name')
            except (IndexError, TypeError):
                # regex matched but there is no 'name' in regex
                name = filename
            return (secs*1000 + frac, name)
    return None


def _group_drf(root, filenames):
    """Match Digital RF filenames and group by name into (time, f)."""
    groups = defaultdict(list)
    if not _RE_SUBDIR.match(os.path.basename(root)):
        return groups
    for filename in filenames:
        m = _RE_DRFFILE.match(filename)
        if m:
            time = datetime.timedelta(
                seconds=int(m.group('secs')),
                milliseconds=int(m.group('frac')),
            )
            groups[m.group('name')].append(
                (time, os.path.join(root, filename))
            )
    return groups


def _group_dmd(root, filenames):
    """Match Digital Metadata filenames and group by name into (time, f)."""
    groups = defaultdict(list)
    if not _RE_SUBDIR.match(os.path.basename(root)):
        return groups
    for filename in filenames:
        m = _RE_DMDFILE.match(filename)
        if m:
            time = datetime.timedelta(seconds=int(m.group('secs')))
            groups[m.group('name')].append(
                (time, os.path.join(root, filename))
            )
    return groups


def ilsdrf(
    path, recursive=True, reverse=False, starttime=None, endtime=None,
    include_drf=True, include_dmd=True,
    include_drf_properties=None, include_dmd_properties=None,
):
    """Generator of Digital RF/Metadata files contained in a channel directory.

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

    if not include_drf:
        # set empty groups so we can iterate over nothing
        rf_groups = {}
    if not include_dmd:
        # set empty groups so we can iterate over nothing
        md_groups = {}

    include_properties = True
    if include_drf_properties and include_dmd_properties:
        prop_regex = _RE_PROPFILE
    elif include_drf_properties:
        prop_regex = _RE_DRFPROPFILE
    elif include_dmd_properties:
        prop_regex = _RE_DMDPROPFILE
    else:
        include_properties = False
        # set empty props so we can iterate over nothing
        props = []

    path = os.path.abspath(path)
    for root, dirs, files in os.walk(path):
        last_level = (not recursive and root != path)
        if last_level:
            # don't go further if we're not recursive and we're already
            # on the first subdir level
            del dirs[:]
        else:
            # walk through directories in sorted order
            dirs.sort(reverse=reverse)

        # match files
        if include_drf:
            rf_groups = _group_drf(root, files)
        if include_dmd:
            md_groups = _group_dmd(root, files)
        if not rf_groups and not md_groups and last_level:
            # not in a matching subdir and we're not recursive, so don't
            # match anything else and continue on same level
            continue
        if include_properties:
            props = [
                os.path.join(root, f) for f in files if prop_regex.match(f)
            ]

        # sort within groups and yield files
        # first properties
        props.sort(reverse=reverse)
        for prop in props:
            yield prop
        # then DMD groups
        for name, dec_files in sorted(md_groups.items(), reverse=reverse):
            dec_files.sort(reverse=reverse)
            if starttime is not None:
                # we want to include the first file *at or before* starttime
                # since this is metadata and we assume previous metadata
                # holds if there is no file at a particular time
                # (because of tuple comparison, all bisects are left)
                # dec_files[:k][0] < starttime <= dec_files[k:][0]
                k = bisect.bisect_left(dec_files, (starttime,))
                # make k the first index to include
                if k == len(dec_files) or dec_files[k][0] != starttime:
                    k = max(k - 1, 0)
                del dec_files[:k]
            if endtime is not None:
                # (because of tuple comparison, all bisects are left)
                # dec_files[:k][0] < endtime <= dec_files[k:][0]
                k = bisect.bisect_left(dec_files, (endtime,))
                # make k first index to exclude
                if k < len(dec_files) and dec_files[k][0] == endtime:
                    k = k + 1
                del dec_files[k:]
            for dec_file in dec_files:
                yield dec_file[-1]
        # finally DRF groups
        for name, dec_files in sorted(rf_groups.items(), reverse=reverse):
            dec_files.sort(reverse=reverse)
            if starttime is not None:
                # (because of tuple comparison, all bisects are left)
                # dec_files[:k][0] < starttime <= dec_files[k:][0]
                k = bisect.bisect_left(dec_files, (starttime,))
                # k is the first index to include
                del dec_files[:k]
            if endtime is not None:
                # (because of tuple comparison, all bisects are left)
                # dec_files[:k][0] < endtime <= dec_files[k:][0]
                k = bisect.bisect_left(dec_files, (endtime,))
                # make k first index to exclude
                if k < len(dec_files) and dec_files[k][0] == endtime:
                    k = k + 1
                del dec_files[k:]
            for dec_file in dec_files:
                yield dec_file[-1]


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


def _build_ls_parser(Parser, *args):
    desc = 'List Digital RF/Metadata files in a (channel) directory.'
    parser = Parser(*args, description=desc)

    parser.add_argument(
        'dirs', nargs='*', default='.',
        help='''Data directory to list. (default: %(default)s)''',
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help='''Also list files recursively in channel subdirectories.
                (default: %(default)s)''',
    )
    parser.add_argument(
        '-R', '--reverse', action='store_true',
        help='''Traverse directories and list files in those directories in
                reversed order. (default: %(default)s)''',
    )
    parser.add_argument(
        '--abs', action='store_true',
        help='''Return absolute paths instead of relative paths.
                (default: %(default)s)''',
    )
    parser.add_argument(
        '--sortall', action='store_true',
        help='''Sort all files together before listing (normally only sorted
                within subdirectories). (default: %(default)s)''',
    )

    timegroup = parser.add_argument_group(title='time')
    timegroup.add_argument(
        '-s', '--starttime', dest='starttime', default=None,
        help='''Data covering this time or after will be included (no effect
                on property files). The time can be given as a datetime string
                (e.g. in ISO8601 format: 2016-01-01T15:24:00Z) or a float
                giving a Unix timestamp in seconds. (default: %(default)s)''',
    )
    timegroup.add_argument(
        '-e', '--endtime', dest='endtime', default=None,
        help='''Data covering this time or before will be included (no effect
                on property files). The time can be given as a datetime string
                (e.g. in ISO8601 format: 2016-01-01T15:24:00Z), a float
                giving a Unix timestamp in seconds, or a '+' followed by a time
                in seconds that will be taken relative to `starttime`.
                (default: %(default)s)''',
    )

    includegroup = parser.add_argument_group(title='include')
    drfgroup = includegroup.add_mutually_exclusive_group()
    dmdgroup = includegroup.add_mutually_exclusive_group()
    drfpropsgroup = includegroup.add_mutually_exclusive_group()
    dmdpropsgroup = includegroup.add_mutually_exclusive_group()

    drfgroup.add_argument(
        '--drf', dest='include_drf', action='store_true', default=True,
        help='''List Digital RF files.
                (default: True)''',
    )
    drfpropsgroup.add_argument(
        '--drfprops', dest='include_drf_properties', action='store_true',
        default=None,
        help='''List drf_properties.h5 files. If unset, use value of --drf.
                (default: None)''',
    )
    drfgroup.add_argument(
        '--nodrf', dest='include_drf', action='store_false',
        help='''Do not list Digital RF files.
                (default: False)''',
    )
    drfpropsgroup.add_argument(
        '--nodrfprops', dest='include_drf_properties', action='store_false',
        help='''Do not list drf_properties.h5 files.
                (default: False)''',
    )
    dmdgroup.add_argument(
        '--dmd', dest='include_dmd', action='store_true', default=True,
        help='''List Digital Metadata files.
                (default: True)''',
    )
    dmdpropsgroup.add_argument(
        '--dmdprops', dest='include_dmd_properties', action='store_true',
        default=None,
        help='''List dmd_properties.h5 files. If unset, use value of --dmd.
                (default: None)''',
    )
    dmdgroup.add_argument(
        '--nodmd', dest='include_dmd', action='store_false',
        help='''Do not list Digital Metadata files.
                (default: False)''',
    )
    dmdpropsgroup.add_argument(
        '--nodmdprops', dest='include_dmd_properties', action='store_false',
        help='''Do not list dmd_properties.h5 files.
                (default: False)''',
    )

    parser.set_defaults(func=_run_ls)

    return parser


def _run_ls(args):
    if args.starttime is not None:
        args.starttime = util.parse_identifier_to_time(args.starttime)
    if args.endtime is not None:
        args.endtime = util.parse_identifier_to_time(
            args.endtime, ref_datetime=args.starttime,
        )

    if args.abs:
        def fixpath(path, start=None):
            return path
    else:
        fixpath = os.path.relpath

    kwargs = vars(args).copy()
    del kwargs['func']
    del kwargs['dirs']
    del kwargs['abs']
    del kwargs['sortall']

    if args.sortall:
        files = []
        for d in args.dirs:
            files.extend(lsdrf(d, **kwargs))
        files.sort(key=sortkey_drf)
        print('\n'.join([fixpath(f, d) for f in files]))
    else:
        for d in args.dirs:
            for f in ilsdrf(d, **kwargs):
                print(fixpath(f, d))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ls_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
