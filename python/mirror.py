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
import traceback
from itertools import chain

from watchdog.events import FileCreatedEvent

from .list_drf import ilsdrf
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
        dest_dir, dest_name = os.path.split(dest_path)
        tmp_dest_path = os.path.join(dest_dir, 'tmp.' + dest_name)
        try:
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            if (not os.path.exists(dest_path)
                    or not filecmp.cmp(src_path, dest_path)):
                if self.verbose:
                    print('Mirroring {0}'.format(src_path))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                # mirror to temporary name, then rename to final destination
                self.mirror_fun(src_path, tmp_dest_path)
                shutil.move(tmp_dest_path, dest_path)
        except OSError:
            if not os.path.isfile(src_path):
                # file doesn't exist anymore, no need to notify
                pass
            else:
                # otherwise, print the error but don't stop mirroring
                traceback.print_exc()

        # try to clean up source directory in case it is empty
        src_dir, src_name = os.path.split(src_path)
        try:
            os.rmdir(src_dir)
        except OSError:
            # directory not empty, just move on
            pass

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
        if method not in ('move', 'copy'):
            raise ValueError('Mirror method must be either "move" or "copy".')
        self.method = method
        self.ignore_existing = ignore_existing
        self.verbose = verbose
        self.include_drf = include_drf
        self.include_dmd = include_dmd

        if not self.include_drf and not self.include_dmd:
            errstr = 'One of `include_drf` or `include_dmd` must be True.'
            raise ValueError(errstr)

        self.event_handlers = []
        # have to copy properties files because static,
        # have to copy metadata because can be modified
        copy_handler = DigitalRFMirrorHandler(
            self.src, self.dest, verbose=verbose, mirror_fun=shutil.copy2,
            include_drf=(self.include_drf and self.method == 'copy'),
            include_dmd=self.include_dmd,
            include_drf_properties=self.include_drf,
            include_dmd_properties=self.include_dmd,
        )
        self.event_handlers.append(copy_handler)

        if self.include_drf and self.method == 'move':
            # move RF files with a separate handler
            drf_handler = DigitalRFMirrorHandler(
                self.src, self.dest, verbose=verbose, mirror_fun=shutil.move,
                include_drf=True, include_dmd=False,
                include_drf_properties=False, include_dmd_properties=False,
            )
            self.event_handlers.append(drf_handler)

        if self.include_dmd and self.method == 'move':
            # set ringbuffer on Digital Metadata files so old ones are removed
            # (can't move since multiple writes can happen to a single file)
            md_ringbuffer_handler = DigitalRFRingbufferHandler(
                count=1, verbose=verbose, dryrun=False,
                include_drf=False, include_dmd=True,
            )
            self.event_handlers.append(md_ringbuffer_handler)

        self._init_observer()

    def _init_observer(self):
        self.observer = DirWatcher(self.src)
        for handler in self.event_handlers:
            self.observer.schedule(handler, self.src, recursive=True)

    def start(self):
        """Start mirror process and return when existing files are handled."""
        # start observer to mirror new and modified files
        self.observer.start()

        if os.path.isdir(self.src):
            # don't need to pause dispatching because mirror ordering is not
            # critical and duplicate events are not harmful (we will either
            # copy again or fail to move because the source doesn't exist)
            # mirror properties at minimum
            paths = ilsdrf(
                self.src, include_drf=False, include_dmd=False,
                include_drf_properties=self.include_drf,
                include_dmd_properties=self.include_dmd,
            )

            if not self.ignore_existing:
                # mirror other files if desired
                more_paths = ilsdrf(
                    self.src, include_drf=self.include_drf,
                    include_dmd=self.include_dmd,
                    include_drf_properties=False,
                    include_dmd_properties=False,
                )
                paths = chain(paths, more_paths)

            # add events for existing files to queue
            for p in paths:
                event = FileCreatedEvent(p)
                for emitter in self.observer.emitters:
                    emitter.queue_event(event)

    def join(self):
        """Wait until a KeyboardInterrupt is received to stop mirroring."""
        try:
            while True:
                if not self.observer.all_alive():
                    # if not all threads of the observer are alive,
                    # reinitialize and restart
                    print(
                        'Found stopped thread, reinitializing and restarting.'
                    )
                    # make a new observer and start it ASAP
                    # (if we missed some events, can't help it)
                    self._init_observer()
                    self.observer.start()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
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
