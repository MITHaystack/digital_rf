# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for mirroring Digital RF files from on directory to another."""

import filecmp
import os
import shutil
import sys
import time

from .list_drf import lsdrf
from .ringbuffer import DigitalRFRingbufferHandler
from .watchdog_drf import DigitalRFEventHandler, DirWatcher

__all__ = (
    'DigitalRFMirrorHandler', 'DigitalRFMirror',
)


class DigitalRFMirrorHandler(DigitalRFEventHandler):
    """Event handler for mirroring Digital RF and Digital Metadata files.

    This handler mirrors new or modified files from a source directory to a
    destination directory. Moved and deleted files are ignored.

    """

    def __init__(
        self, src, dest, verbose=False, mirror_fun=shutil.copy2,
        include_drf=True, include_dmd=True, include_drf_properties=True,
        include_dmd_properties=True,
    ):
        """Create Digital RF mirror handler given source and destination."""
        self.src = os.path.abspath(src)
        self.dest = os.path.abspath(dest)
        self.verbose = verbose
        self.mirror_fun = mirror_fun
        super(DigitalRFMirrorHandler, self).__init__(
            include_drf=include_drf, include_dmd=include_dmd,
            include_drf_properties=include_drf_properties,
            include_dmd_properties=include_dmd_properties,
        )

    def _get_dest_path(self, src_path):
        rel_path = os.path.relpath(src_path, self.src)
        dest_path = os.path.join(self.dest, rel_path)
        return dest_path

    def mirror_to_dest(self, src_path):
        """Mirror file to its location in the destination directory."""
        dest_path = self._get_dest_path(src_path)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        if (not os.path.exists(dest_path)
                or not filecmp.cmp(src_path, dest_path)):
            if self.verbose:
                print('Mirroring {0}'.format(src_path))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
            self.mirror_fun(src_path, dest_path)

    def on_created(self, event):
        """Mirror newly-created file."""
        self.mirror_to_dest(event.src_path)

    def on_modified(self, event):
        """Mirror modified file."""
        self.mirror_to_dest(event.src_path)


class DigitalRFMirror(object):
    """Monitor a directory and mirror its Digital RF files to another.

    This class combines an event handler and a file system observer. It
    monitors a source directory and mirrors new and modified (and optionally,
    existing) Digital RF and Digital Metadata files to a given destination
    directory. Files that are moved or deleted in the source directory are
    ignored.

    """

    def __init__(
        self, src, dest, method='move', ignore_existing=False, verbose=False,
        include_drf=True, include_dmd=True,
    ):
        """Create Digital RF mirror object. Use start/run method to begin.

        Parameters
        ----------

        src : str
            Source directory to monitor.

        dest : str
            Destination directory.

        method : 'move' | 'copy'
            Mirroring method. New Digital RF files in the source directory will
            be moved or copied to the destination directory depending on this
            parameter. Metadata is always copied regardless of this parameter.


        Other Parameters
        ----------------

        ignore_existing : bool
            If True, existing Digital RF and Digital Metadata files will not be
            mirrored. Otherwise, they will be mirrored to the destination
            directory along with any existing channel metadata files
            when mirroring starts.

        verbose : bool
            If True, print the name of mirrored files.

        include_drf : bool
            If True, include Digital RF files. If False, ignore Digital RF
            files.

        include_dmd : bool
            If True, include Digital Metadata files. If False, ignore Digital
            Metadata files.

        """
        self.src = os.path.abspath(src)
        self.dest = os.path.abspath(dest)
        if method == 'move':
            mirror_fun = shutil.move
        elif method == 'copy':
            mirror_fun = shutil.copy2
        else:
            raise ValueError('Mirror method must be either "move" or "copy".')
        self.method = method
        self.ignore_existing = ignore_existing
        self.verbose = verbose
        self.include_drf = include_drf
        self.include_dmd = include_dmd

        if not self.include_drf and not self.include_dmd:
            errstr = 'One of `include_drf` or `include_dmd` must be True.'
            raise ValueError(errstr)

        self.observer = DirWatcher(self.src)

        self.prop_handler = DigitalRFMirrorHandler(
            self.src, self.dest, verbose=verbose, mirror_fun=shutil.copy2,
            include_drf=False, include_dmd=False,
            include_drf_properties=self.include_drf,
            include_dmd_properties=self.include_dmd,
        )
        self.observer.schedule(self.prop_handler, self.src, recursive=True)

        if self.include_drf:
            self.drf_handler = DigitalRFMirrorHandler(
                self.src, self.dest, verbose=verbose, mirror_fun=mirror_fun,
                include_drf=True, include_dmd=False,
                include_drf_properties=False, include_dmd_properties=False,
            )
            self.observer.schedule(
                self.drf_handler, self.src, recursive=True,
            )

        if self.include_dmd:
            self.md_handler = DigitalRFMirrorHandler(
                self.src, self.dest, verbose=verbose, mirror_fun=shutil.copy2,
                include_drf=False, include_dmd=True,
                include_drf_properties=False, include_dmd_properties=False,
            )
            self.observer.schedule(
                self.md_handler, self.src, recursive=True,
            )
            # set ringbuffer on Digital Metadata files so old ones are removed
            # (can't move since multiple writes can happen to a single file)
            if self.method == 'move':
                self.md_ringbuffer_handler = DigitalRFRingbufferHandler(
                    threshold=1, verbose=verbose, dryrun=False,
                    include_drf=False, include_dmd=True,
                )
                self.observer.schedule(
                    self.md_ringbuffer_handler, self.src, recursive=True,
                )

    def start(self):
        """Start mirror process and return when existing files are handled."""
        # start observer to mirror new and modified files
        self.observer.start()

        if os.path.isdir(self.src):
            # mirror existing files, all if desired or properties only
            proppaths = lsdrf(
                self.src, include_drf=False, include_dmd=False,
                include_drf_properties=self.include_drf,
                include_dmd_properties=self.include_dmd,
            )
            for p in proppaths:
                self.prop_handler.mirror_to_dest(p)

            if not self.ignore_existing:
                drfpaths = lsdrf(
                    self.src, include_drf=self.include_drf, include_dmd=False,
                    include_drf_properties=False, include_dmd_properties=False,
                )
                for p in drfpaths:
                    self.drf_handler.mirror_to_dest(p)

                mdpaths = lsdrf(
                    self.src, include_drf=False, include_dmd=self.include_dmd,
                    include_drf_properties=False, include_dmd_properties=False,
                )
                for p in mdpaths:
                    self.md_handler.mirror_to_dest(p)
                if self.method == 'move':
                    self.md_ringbuffer_handler.add_files(mdpaths)

    def join(self):
        """Wait until a KeyboardInterrupt is received to stop mirroring."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            sys.stdout.write('\n')
            sys.stdout.flush()
        self.observer.join()

    def run(self):
        """Start mirroring and wait for a KeyboardInterrupt to stop."""
        self.start()
        self.join()

    def stop(self):
        """Stop mirror process."""
        self.observer.stop()


def _build_mirror_parser(Parser, *args):
    desc = 'Mirror Digital RF files from one directory to another.'
    parser = Parser(*args, description=desc)

    parser.add_argument(
        'method', choices=['mv', 'cp'],
        help='Mirroring method.',
    )
    parser.add_argument('src', help='Source directory to monitor.')
    parser.add_argument('dest', help='Destination directory.')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print the name of mirrored files.',
    )
    parser.add_argument(
        '--ignore_existing', action='store_true',
        help='Ignore existing files in source directory.',
    )

    includegroup = parser.add_argument_group(title='include')
    includegroup.add_argument(
        '--nodrf', dest='include_drf', action='store_false',
        help='''Do not mirror Digital RF HDF5 files.
                (default: False)''',
    )
    includegroup.add_argument(
        '--nodmd', dest='include_dmd', action='store_false',
        help='''Do not mirror Digital Metadata HDF5 files.
                (default: False)''',
    )

    parser.set_defaults(func=_run_mirror)

    return parser


def _run_mirror(args):
    methods = {'mv': 'move', 'cp': 'copy'}
    args.method = methods[args.method]

    kwargs = vars(args).copy()
    del kwargs['func']
    mirror = DigitalRFMirror(**kwargs)
    print('Mirroring ({0}) {1} to {2}.'.format(
        args.method, args.src, args.dest,
    ))
    print('Type Ctrl-C to quit.')
    mirror.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_mirror_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
