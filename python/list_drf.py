# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for listing Digital RF/Metadata files in a directory."""
import os
import re

from six.moves import zip

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

# Digital RF file, e.g. rf@1474491360.000.h5
GLOB_DRFFILE = '*@[0-9]*.[0-9][0-9][0-9].h5'
RE_DRFFILE = (
    r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.(?P<frac>[0-9]{3})\.h5$'
)

# Digital Metadata file, e.g. metadata@1474491360.h5
GLOB_DMDFILE = '*@[0-9]*.h5'
RE_DMDFILE = r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.h5$'

# either Digital RF or Digital Metadata file
RE_FILE = (
    r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)(?:\.(?P<frac>[0-9]{3}))?\.h5$'
)

# Digital RF channel properties file, e.g. drf_properties.h5
# include pre-2.5-style metadata.h5 for now
GLOB_DRFPROPFILE = '*.h5'
RE_DRFPROPFILE = r'(?P<name>drf_properties|metadata)\.h5$'

# Digital RF channel properties file, e.g. dmd_properties.h5
# include pre-2.5-style metadata.h5 for now
GLOB_DMDPROPFILE = '*.h5'
RE_DMDPROPFILE = r'(?P<name>dmd_properties|metadata)\.h5$'

# either Digital RF or Digital Metadata channel properties file
RE_PROPFILE = r'(?P<name>(?:drf|dmd)_properties|metadata)\.h5$'

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


def ilsdrf(
    path, recursive=True, reverse=False, include_drf=True, include_dmd=True,
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
    if include_drf_properties is None:
        include_drf_properties = include_drf
    if include_dmd_properties is None:
        include_dmd_properties = include_dmd

    # regexes for all files, starting with DRF and DMD
    regexes = []
    if include_drf and include_dmd:
        regexes.append('^' + RE_FILE)
    elif include_drf:
        regexes.append('^' + RE_DRFFILE)
    elif include_dmd:
        regexes.append('^' + RE_DMDFILE)
    regexes = [re.compile(r) for r in regexes]

    # regex for subdirectory if DRF and DMD are included
    if regexes:
        subdir_regexes = [re.compile(RE_SUBDIR)]
    else:
        # match nothing
        subdir_regexes = []

    prop_regexes = []
    if include_drf_properties and include_dmd_properties:
        prop_regexes.append('^' + RE_PROPFILE)
    elif include_drf_properties:
        prop_regexes.append('^' + RE_DRFPROPFILE)
    elif include_dmd_properties:
        prop_regexes.append('^' + RE_DMDPROPFILE)
    prop_regexes = [re.compile(r) for r in prop_regexes]
    regexes.extend(prop_regexes)

    for root, dirs, files in os.walk(path):
        last_level = (not recursive and root != path)
        if last_level:
            # don't go further if we're not recursive and we're already
            # on the first subdir level
            del dirs[:]
        else:
            # walk through directories in sorted order
            dirs.sort(reverse=reverse)

        # yield files from this directory in sorted order and filter all at
        # once so the regex only has to be evaluated once per file
        if any(r.match(os.path.basename(root)) for r in subdir_regexes):
            keys = (sortkey_drf(f, regexes=regexes) for f in files)
        elif last_level:
            # not in a matching subdir and we're not recursive, so don't
            # match anything else and continue on same level
            continue
        else:
            keys = (sortkey_drf(f, regexes=prop_regexes) for f in files)
        decorated_files = [
            key + (os.path.join(root, f),)
            for key, f in zip(keys, files) if key is not None
        ]
        decorated_files.sort(reverse=reverse)
        for decorated_file in decorated_files:
            yield decorated_file[-1]


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
        '--sortall', action='store_true',
        help='''Sort all files together before listing (normally only sorted
                within subdirectories). (default: %(default)s)''',
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
    kwargs = vars(args).copy()
    del kwargs['func']
    del kwargs['dirs']
    del kwargs['sortall']

    if args.sortall:
        files = []
        for d in args.dirs:
            files.extend(lsdrf(d, **kwargs))
        files.sort(key=sortkey_drf)
        print('\n'.join(files))
    else:
        for d in args.dirs:
            for f in ilsdrf(d, **kwargs):
                print(f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ls_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
