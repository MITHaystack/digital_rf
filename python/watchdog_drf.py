# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for montoring creation/modification/deletion of Digital RF files."""

import os
import time

from watchdog.events import (FileCreatedEvent, FileDeletedEvent,
                             RegexMatchingEventHandler)
from watchdog.observers import Observer
from watchdog.utils import unicode_paths

from .list_drf import (RE_DMD, RE_DMDPROP, RE_DRF, RE_DRFDMD, RE_DRFDMDPROP,
                       RE_DRFPROP)

__all__ = (
    'DigitalRFEventHandler',
)


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
        self, include_drf=True, include_dmd=True, include_properties=True,
    ):
        """Create event handler, optionally ignoring all properties."""
        regexes = []
        if include_drf and include_dmd:
            regexes.append(RE_DRFDMD)
            if include_properties:
                regexes.append(RE_DRFDMDPROP)
        elif include_drf:
            regexes.append(RE_DRF)
            if include_properties:
                regexes.append(RE_DRFPROP)
        elif include_dmd:
            regexes.append(RE_DMD)
            if include_properties:
                regexes.append(RE_DMDPROP)
        elif include_properties:
            regexes.append(RE_DRFDMDPROP)
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
