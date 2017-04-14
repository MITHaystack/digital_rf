"""Module for montoring creation/modification/deletion of Digital RF files."""

import os
import re
import time

from watchdog.events import (FileCreatedEvent, FileDeletedEvent,
                             RegexMatchingEventHandler)
from watchdog.observers import Observer
from watchdog.utils import unicode_paths

__all__ = (
    'RE_SUBDIR', 'RE_DRFFILE', 'RE_DMDFILE', 'RE_MDFILE',
    'RE_DRF', 'RE_DMD', 'RE_METADATA',
    'DigitalRFEventHandler', 'lsdrf',
)

# e.g. 2016-09-21T20-00-00
RE_SUBDIR = (r'(?P<ymd>[0-9]{4}-[0-9]{2}-[0-9]{2})'
             r'T(?P<hms>[0-9]{2}-[0-9]{2}-[0-9]{2})')
# e.g. rf@1474491360.000.h5
RE_DRFFILE = r'(?!tmp\.).+?@(?P<secs>[0-9]+\.[0-9]{3})\.h5'
# e.g. metadata@1474491360.h5
RE_DMDFILE = r'(?!tmp\.).+?@(?P<secs>[0-9]+)\.h5'
# e.g. metadata.h5
RE_MDFILE = r'metadata\.h5'

# Digital RF file in correct subdirectory structure
RE_DRF = re.escape(os.sep).join((r'.*?', RE_SUBDIR, RE_DRFFILE))
# Digital Metadata file in correct subdirectory structure
RE_DMD = re.escape(os.sep).join((r'.*?', RE_SUBDIR, RE_DMDFILE))
# metadata file associated with Digital RF or Digital Metadata directory
RE_METADATA = re.escape(os.sep).join((r'.*?', RE_MDFILE))


def lsdrf(path, include_drf=True, include_dmd=True, include_metadata=True):
    """Get list of Digital RF files contained in a directory."""
    regexes = []
    if include_drf:
        regexes.append(re.compile(RE_DRF))
    if include_dmd:
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


class DigitalRFEventHandler(RegexMatchingEventHandler):
    """
    Event handler for Digital RF and Digital Metadata files.

    Events to files that match the Digital RF or Digital Metadata format and
    directory structure will trigger the corresponding method. Override these
    methods in a subclass to define their actions.

    The triggered methods are: `on_created`, `on_deleted`, `on_modified`, and
    `on_drf_moved`.

    """

    def __init__(
        self, include_drf=True, include_dmd=True, include_metadata=True,
    ):
        """Create event handler, optionally ignoring all metadata."""
        regexes = []
        if include_drf:
            regexes.append(RE_DRF)
        if include_dmd:
            regexes.append(RE_DMD)
        if include_metadata:
            regexes.append(RE_METADATA)
        super(DigitalRFEventHandler, self).__init__(
            regexes=regexes, ignore_directories=True,
        )

    def on_drf_moved(self, event):
        """Called when a file or a directory is moved or renamed.

        :param event:
            Event representing file/directory movement.
        :type event:
            :class:`DirMovedEvent` or :class:`FileMovedEvent`
        """

    # override on_moved so
    #  1) src and dest match regex -> on_drf_moved
    #  2) src only matches regex -> on_deleted
    #  3) dest only matches regex -> on_created
    def on_moved(self, event):
        """Translate filesystem move events. DO NOT OVERRIDE."""
        src_path = unicode_paths.decode(event.src_path)
        dest_path = unicode_paths.decode(event.dest_path)

        src_match = any(r.match(src_path) for r in self.regexes)
        dest_match = any(r.match(dest_path) for r in self.regexes)

        if src_match and dest_match:
            self.on_drf_moved(event)
        elif src_match:
            new_event = FileDeletedEvent(event.src_path)
            self.on_deleted(new_event)
        else:
            new_event = FileCreatedEvent(event.dest_path)
            self.on_created(new_event)


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
    files.sort()
    print('\n'.join(files))


def _build_watch_parser(Parser, *args):
    desc = 'Print Digital RF file events occurring in a directory.'
    parser = Parser(*args, description=desc)

    parser.add_argument('dir', nargs='?', default='.',
                        help='''Data directory to monitor.
                               (default: %(default)s)''')

    parser.set_defaults(func=_run_watch)

    return parser


def _run_watch(args):
    args.dir = os.path.abspath(args.dir)

    # subclass DigitalRFEventHandler to just print events
    class DigitalRFPrint(DigitalRFEventHandler):

        def on_drf_moved(self, event):
            print('Moved {0} to {1}'.format(event.src_path, event.dest_path))

        def on_created(self, event):
            print('Created {0}'.format(event.src_path))

        def on_deleted(self, event):
            print('Deleted {0}'.format(event.src_path))

        def on_modified(self, event):
            print('Modified {0}'.format(event.src_path))

    event_handler = DigitalRFPrint()
    observer = Observer()
    observer.schedule(event_handler, args.dir, recursive=True)
    print('Monitoring {0}:'.format(args.dir))
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_watch_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
