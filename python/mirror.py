"""Module for mirroring Digital RF files from on directory to another."""

import filecmp
import os
import shutil
import sys
import time
from watchdog.observers import Observer

from .watchdog_drf import DigitalRFEventHandler, lsdrf

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
        include_drf=True, include_dmd=True, include_metadata=True,
    ):
        """Create Digital RF mirror handler given source and destination."""
        self.src = os.path.abspath(src)
        self.dest = os.path.abspath(dest)
        self.verbose = verbose
        self.mirror_fun = mirror_fun
        super(DigitalRFMirrorHandler, self).__init__(
            include_drf=include_drf, include_dmd=include_dmd,
            include_metadata=include_metadata,
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
            self.mirror_fun(src_path, dest_path)
            if self.verbose:
                print('Mirroring {0}'.format(src_path))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    def on_created(self, event):
        self.mirror_to_dest(event.src_path)

    def on_modified(self, event):
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

        """
        self.src = os.path.abspath(src)
        self.dest = os.path.abspath(dest)
        if method == 'move':
            mirror_fun = shutil.move
        elif method == 'copy':
            mirror_fun = shutil.copy2
        else:
            raise ValueError('Mirror method must be either "move" or "copy".')
        self.ignore_existing = ignore_existing
        self.verbose = verbose
        self.drf_handler = DigitalRFMirrorHandler(
            src, dest, verbose=verbose, mirror_fun=mirror_fun,
            include_drf=True, include_dmd=False, include_metadata=False,
        )
        self.md_handler = DigitalRFMirrorHandler(
            src, dest, verbose=verbose, mirror_fun=shutil.copy2,
            include_drf=False, include_dmd=True, include_metadata=True,
        )
        self.observer = Observer()
        self.observer.schedule(
            self.drf_handler, self.src, recursive=True,
        )
        self.observer.schedule(
            self.md_handler, self.src, recursive=True,
        )

    def start(self):
        """Start mirror process and return when existing files are handled."""
        # start observer to mirror new and modified files
        self.observer.start()

        # mirror existing files, all if desired or metadata only
        if self.ignore_existing:
            mdpaths = lsdrf(self.src, include_drf=False, include_dmd=False)
        else:
            mdpaths = lsdrf(self.src, include_drf=False)
        for p in mdpaths:
            self.md_handler.mirror_to_dest(p)

        if not self.ignore_existing:
            drfpaths = lsdrf(
                self.src, include_dmd=False, include_metadata=False,
            )
            for p in drfpaths:
                self.drf_handler.mirror_to_dest(p)

    def join(self):
        """Wait until a KeyboardInterrupt is received to stop mirroring."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
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

    parser.set_defaults(func=_run_mirror)

    return parser


def _run_mirror(args):
    methods = {'mv': 'move', 'cp': 'copy'}
    method = methods[args.method]

    mirror = DigitalRFMirror(
        args.src, args.dest, method, args.ignore_existing, args.verbose,
    )
    print('Mirroring ({0}) {1} to {2}.'.format(method, args.src, args.dest))
    print('Type Ctrl-C to quit.')
    mirror.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_mirror_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
