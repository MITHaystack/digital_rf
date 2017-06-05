import os
import re

__all__ = (
    'GLOB_SUBDIR', 'RE_SUBDIR', 'GLOB_DRFFILE', 'RE_DRFFILE',
    'GLOB_DMDFILE', 'RE_DMDFILE', 'RE_FILE', 'GLOB_MDFILE', 'RE_MDFILE',
    'RE_DRF', 'RE_DMD', 'RE_DRFDMD', 'RE_METADATA',
    'lsdrf', 'sortkey_drf'
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
RE_DRFFILE = r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.(?P<frac>[0-9]{3})\.h5'

# Digital Metadata file, e.g. metadata@1474491360.h5
GLOB_DMDFILE = '*@[0-9]*.h5'
RE_DMDFILE = r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)\.h5'

# either Digital RF or Digital Metadata file
RE_FILE = (
    r'(?P<name>(?!tmp\.).+?)@(?P<secs>[0-9]+)(?:\.(?P<frac>[0-9]{3}))?\.h5'
)

# channel metadata file, e.g. metadata.h5
GLOB_MDFILE = 'metadata.h5'
RE_MDFILE = r'(?P<name>metadata)\.h5'

# Digital RF file in correct subdirectory structure
RE_DRF = re.escape(os.sep).join((r'.*?', RE_SUBDIR, RE_DRFFILE))
# Digital Metadata file in correct subdirectory structure
RE_DMD = re.escape(os.sep).join((r'.*?', RE_SUBDIR, RE_DMDFILE))
# either Digital RF or Digital Metadata in its correct subdirectory
RE_DRFDMD = re.escape(os.sep).join((r'.*?', RE_SUBDIR, RE_FILE))
# metadata file associated with Digital RF or Digital Metadata directory
RE_METADATA = re.escape(os.sep).join((r'.*?', RE_MDFILE))


def sortkey_drf(filename, _r=re.compile(RE_FILE)):
    """Get key for a Digital RF filename to sort first by sample time."""
    m = _r.match(filename)
    try:
        secs = int(m.group('secs'))
    except (AttributeError, IndexError, TypeError):
        # no match, or regex matched but there is no 'secs' in regex
        secs = 0
    try:
        frac = int(m.group('frac'))
    except (AttributeError, IndexError, TypeError):
        # no match, or regex matched but there is no 'frac' in regex
        frac = 0
    try:
        name = m.group('name')
    except (AttributeError, IndexError, TypeError):
        # no match, or regex matched but there is no 'name' in regex
        name = filename
    key = (secs*1000 + frac, name)
    return key


def lsdrf(path, include_drf=True, include_dmd=True, include_metadata=True):
    """Get list of Digital RF files contained in a directory."""
    regexes = []
    if include_drf and include_dmd:
        regexes.append(re.compile(RE_DRFDMD))
    elif include_drf:
        regexes.append(re.compile(RE_DRF))
    elif include_dmd:
        regexes.append(re.compile(RE_DMD))
    if include_metadata:
        regexes.append(re.compile(RE_METADATA))

    drf_files = []
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for filename in files:
            p = os.path.join(root, filename)
            if any(r.match(p) for r in regexes):
                drf_files.append(p)

    return drf_files


def _build_ls_parser(Parser, *args):
    desc = 'List Digital RF files in a directory.'
    parser = Parser(*args, description=desc)

    parser.add_argument('dir', nargs='?', default='.',
                        help='''Data directory to monitor.
                               (default: %(default)s)''')

    includegroup = parser.add_argument_group(title='include')
    includegroup.add_argument(
        '--nodrf', dest='include_drf', action='store_false',
        help='''Do not list Digital RF HDF5 files.
                (default: False)''',
    )
    includegroup.add_argument(
        '--nodmd', dest='include_dmd', action='store_false',
        help='''Do not list Digital Metadata HDF5 files.
                (default: False)''',
    )
    includegroup.add_argument(
        '--nometadata', dest='include_metadata', action='store_false',
        help='''Do not list metadata.h5 files.
                (default: False)''',
    )

    parser.set_defaults(func=_run_ls)

    return parser


def _run_ls(args):
    args.dir = os.path.abspath(args.dir)

    files = lsdrf(
        args.dir,
        include_drf=args.include_drf,
        include_dmd=args.include_dmd,
        include_metadata=args.include_metadata,
    )
    files.sort(key=sortkey_drf)
    print('\n'.join(files))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ls_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
