# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
from argparse import ArgumentParser

from .list_drf import _build_ls_parser
try:
    from .mirror import _build_mirror_parser
    from .ringbuffer import _build_ringbuffer_parser
    from .watchdog_drf import _build_watch_parser
except ImportError:
    # if no watchdog package, these fail to import, so just ignore
    _WATCHDOG = False
else:
    _WATCHDOG = True


def main(args=None):
    parser = ArgumentParser(
        description='Digital RF command line tools.',
        epilog=(
            'Type "drf <command> -h" to display help for a particular command.'
        )
    )
    subparsers = parser.add_subparsers(
        title='Available commands',
    )

    _build_ls_parser(subparsers.add_parser, 'ls')
    if _WATCHDOG:
        _build_mirror_parser(subparsers.add_parser, 'mirror')
        _build_ringbuffer_parser(subparsers.add_parser, 'ringbuffer')
        _build_watch_parser(subparsers.add_parser, 'watch')

    # parse the command line and/or function call arguments
    parsed = parser.parse_args(args)
    # use the function provided by the appropriate subparser to execute
    # the parsed arguments for that subparser
    parsed.func(parsed)
