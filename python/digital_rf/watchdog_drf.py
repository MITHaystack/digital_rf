# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for montoring creation/modification/deletion of Digital RF files."""
from __future__ import absolute_import, division, print_function

import datetime
import os
import re
import sys
import time

from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    RegexMatchingEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver, ObservedWatch
from watchdog.observers.polling import PollingObserver

try:
    from os import fsdecode
except ImportError:
    # must be using Python 2, which implies a watchdog version with unicode_paths
    from watchdog.utils.unicode_paths import decode as fsdecode

from . import list_drf, util
from .list_drf import RE_DMD, RE_DMDPROP, RE_DRF, RE_DRFDMD, RE_DRFDMDPROP, RE_DRFPROP

__all__ = ("DigitalRFEventHandler", "DirWatcher")


class DigitalRFEventHandler(RegexMatchingEventHandler):
    """Event handler for Digital RF and Digital Metadata files.

    Events to files that match the Digital RF or Digital Metadata format and
    directory structure will trigger the corresponding method. Override these
    methods in a subclass to define their actions.

    The triggered methods are: `on_created`, `on_deleted`, `on_modified`, and
    `on_moved`.

    """

    def __init__(
        self,
        starttime=None,
        endtime=None,
        include_drf=True,
        include_dmd=True,
        include_drf_properties=None,
        include_dmd_properties=None,
        ignore_regexes=None,
    ):
        """Create Digital RF event handler for given time range and file types.

        Parameters
        ----------
        starttime : datetime.datetime
            Data covering this time or after will be included. This has no
            effect on property files.

        endtime : datetime.datetime
            Data covering this time or earlier will be included. This has no
            effect on property files.

        include_drf : bool
            If True, include Digital RF files.

        include_dmd : bool
            If True, include Digital Metadata files.

        include_drf_properties : bool | None
            If True, include the Digital RF properties file.
            If None, use `include_drf` value.

        include_dmd_properties : bool | None
            If True, include the Digital Metadata properties file.
            If None, use `include_dmd` value.

        ignore_regexes : list
            List of regexes for ignoring matching event paths.

        """
        # convert starttime and endtime to timedeltas for comparison
        if starttime is not None:
            if starttime.tzinfo is None:
                starttime = starttime.replace(tzinfo=datetime.timezone.utc)
            starttime = starttime - util.epoch
        self.starttime = starttime
        if endtime is not None:
            if endtime.tzinfo is None:
                endtime = endtime.replace(tzinfo=datetime.timezone.utc)
            endtime = endtime - util.epoch
        self.endtime = endtime

        if include_drf_properties is None:
            include_drf_properties = include_drf
        if include_dmd_properties is None:
            include_dmd_properties = include_dmd

        if ignore_regexes is None:
            ignore_regexes = []

        regexes = []
        if include_drf and include_dmd:
            regexes.append(RE_DRFDMD)
        elif include_drf:
            regexes.append(RE_DRF)
        elif include_dmd:
            regexes.append(RE_DMD)
        if include_drf_properties and include_dmd_properties:
            regexes.append(RE_DRFDMDPROP)
        elif include_drf_properties:
            regexes.append(RE_DRFPROP)
        elif include_dmd_properties:
            regexes.append(RE_DMDPROP)

        if not regexes:
            raise ValueError("Must include at least one file type.")

        super(DigitalRFEventHandler, self).__init__(
            regexes=regexes, ignore_regexes=ignore_regexes, ignore_directories=True
        )

    def dispatch(self, event, match_time=True):
        """Dispatch events to the appropriate methods.

        Parameters
        ----------
        event : FileSystemEvent
            Event object representing the file system event.

        match_time : bool
            If False, do not check the matched file's time against the
            handler's starttime and endtime.

        """
        if self.ignore_directories and event.is_directory:
            return

        src_match = False
        if event.src_path:
            src_path = fsdecode(event.src_path)
            if not any(r.match(src_path) for r in self.ignore_regexes):
                for r in self.regexes:
                    m = r.match(src_path)
                    if m:
                        src_match = True
                        match = m

        dest_match = False
        if getattr(event, "dest_path", None):
            dest_path = fsdecode(event.dest_path)
            if not any(r.match(dest_path) for r in self.ignore_regexes):
                for r in self.regexes:
                    m = r.match(dest_path)
                    if m:
                        dest_match = True
                        match = m

            # change move event to deleted/created if both regexes didn't match
            if src_match and not dest_match:
                event = FileDeletedEvent(event.src_path)
            elif dest_match and not src_match:
                event = FileCreatedEvent(event.dest_path)

        if not src_match and not dest_match:
            return

        # regexes matched, now check the time
        if match_time:
            try:
                secs = int(match.group("secs"))
            except (IndexError, TypeError):
                # no time, don't need to check it
                pass
            else:
                try:
                    msecs = int(match.group("frac"))
                except (IndexError, TypeError):
                    msecs = 0
                time = datetime.timedelta(seconds=secs, milliseconds=msecs)
                if self.starttime is not None and time < self.starttime:
                    return
                elif self.endtime is not None and time > self.endtime:
                    return

        # the event matched, including time if applicable, dispatch
        # we've handled matching, so jump to the regex handler's parent for dispatching
        super(RegexMatchingEventHandler, self).dispatch(event)


class DirWatcher(BaseObserver, RegexMatchingEventHandler):
    """Watchdog observer for monitoring a particular directory.

    This observer has a sub-observer and handler for noticing when the
    specified directory is created or deleted so that observations of that
    directory can be preserved. In this way observations can be scheduled
    for a directory that doesn't yet exist, and they will be enacted once it is
    created.

    Notes
    -----
    Watchdog uses the following nomenclature:

    watch :
        A directory to be monitored, paired with a flag for whether it should be
        monitored recursively.

    event :
        A filesystem event (created/deleted/modified/moved) for a file or directory.

    emitter :
        A thread that adds (event, watch) pairs to a shared queue for processing.

    handler :
        A class whose dispatch method is called to process events from the queue.

    observer :
        A thread that owns the event queue and manages emitters and handlers for a
        collection of watches. Watches are returned from the `schedule` method, which
        registers a handler with the observer and creates an emitter if necessary.
        Calling `unschedule` on a watch stops the emitter and deletes the associated
        handlers so that no further events are processed.

    """

    def __init__(self, path, force_polling=False, **kwargs):
        """Create observer for the directory at `path`."""
        if force_polling:
            observer_class = PollingObserver
        else:
            observer_class = Observer

        self.root_observer = observer_class(**kwargs)

        # get proper emitter class from root_observer instance of observer_class
        BaseObserver.__init__(
            self, emitter_class=self.root_observer._emitter_class, **kwargs
        )
        # initialize as an event handler as well for handling the root observer events
        RegexMatchingEventHandler.__init__(self)

        self.path = os.path.abspath(path)
        self.root_watch = None
        self._stopped_handlers = dict()

    def _get_next_dir_in_path(self, start):
        """From `start`, find the next directory from self.path."""
        relpath = os.path.relpath(self.path, start)
        nextdirname = relpath.split(os.path.sep)[0]
        nextdir = os.path.abspath(os.path.join(start, nextdirname))
        return nextdir

    def _set_root(self, root):
        """Set up watching `root` or closest existing parent."""
        if self.root_watch is not None and self.root_watch.path == root:
            # already watching the specified root, return early
            return

        # schedule new root watch
        while True:
            try:
                watch = self.root_observer.schedule(
                    event_handler=self, path=root, recursive=False
                )
            except OSError as sched_err:
                # root doesn't exist, move up one directory and try again
                try:
                    # clean up from failed scheduling
                    self.root_observer.unschedule(ObservedWatch(root, False))
                except KeyError:
                    pass
                newroot = os.path.dirname(root)
                if newroot == root:
                    # we've gotten to system root and still failed
                    raise sched_err
                else:
                    root = newroot
            else:
                # watch was set, break out of while loop
                break

        # add regex for root and next subdirectory
        regexes = []
        regexes.append("^" + re.escape(root) + "$")
        nextdir = self._get_next_dir_in_path(root)
        if nextdir != root:
            regexes.append("^" + re.escape(nextdir) + "$")

        # update handler (self) with regexes
        RegexMatchingEventHandler.__init__(
            self, regexes=regexes, ignore_directories=False
        )

        # unschedule old root watch
        if self.root_watch is not None:
            try:
                self.root_observer.unschedule(self.root_watch)
            except KeyError:
                # emitter already stopped
                pass

        self.root_watch = watch

    def _start_watching_path(self):
        """Schedule handlers for self.path now that it exists."""
        # re-schedule the saved watches/handlers immediately so we miss as few events
        # as possible
        for watch, handlers in self._stopped_handlers.items():
            for handler in handlers:
                self.schedule(
                    event_handler=handler, path=watch.path, recursive=watch.is_recursive
                )

        # generate any events for files/dirs in self.path that were created before the
        # watch started (these events might come out of order relative to)
        with self._lock:
            for emitter in self.emitters:
                dirs = [emitter.watch.path]
                for root in dirs:
                    names = os.listdir(root)
                    paths = [os.path.join(root, name) for name in names]
                    paths.sort(key=os.path.getmtime)
                    for path in paths:
                        if os.path.isdir(path):
                            if emitter.watch.is_recursive:
                                dirs.append(path)
                            event = DirCreatedEvent(path)
                        else:
                            event = FileCreatedEvent(path)
                        emitter.queue_event(event)

    def _stop_watching_path(self):
        """Save handlers for self.path so they can be reinstated, clean up."""
        with self._lock:
            # copy all watches/handlers so we can restart them
            self._stopped_handlers.update(self._handlers.copy())
        for watch in list(self._watches):
            try:
                self.unschedule(watch)
            except KeyError:
                # emitter already stopped
                pass
        self._watches.clear()

    def on_created(self, event):
        """Adjust watches for just-created directory in self.path hierarchy."""
        if event.is_directory:
            if event.src_path == self.path:
                # if event is our path, start watching it immediately
                self._start_watching_path()
                # then update the root to be our path to catch when it is deleted
                self._set_root(event.src_path)
            else:
                # just update the root to the new leaf level
                self._set_root(event.src_path)
                # check to see if the next directory in the path already exists
                # and generate an event (that may have been missed) if it does
                nextdir = self._get_next_dir_in_path(event.src_path)
                if os.path.isdir(nextdir):
                    emitter = self.root_observer._emitter_for_watch[self.root_watch]
                    event = DirCreatedEvent(nextdir)
                    emitter.queue_event(event)

    def on_deleted(self, event):
        """Adjust watches for deleted directory in self.path hierarchy."""
        if event.is_directory:
            # root directory in self.path tree has been deleted
            # stop watching if it is self.path
            if event.src_path == self.path:
                self._stop_watching_path()
            # set root to (closest) parent of deleted directory
            self._set_root(os.path.dirname(event.src_path))

    def schedule(self, event_handler, path=None, recursive=False):
        """Schedule watching a path with an event handler."""
        if path is None:
            path = self.path
        try:
            super(DirWatcher, self).schedule(
                event_handler=event_handler, path=path, recursive=recursive
            )
        except OSError:
            # path doesn't exist and we're already running, but watch was added
            # to _handlers, so re-do _stop_watching_path
            self._stop_watching_path()

    def start(self):
        """Start watching and enable handlers."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
        msg = "{0} | Starting watchdog observer(s)."
        print(msg.format(now))
        sys.stdout.flush()

        # start observing directories
        self.root_observer.start()
        # start observer before adding path so it can find closest root
        self._set_root(self.path)
        # start self observer
        if not os.path.isdir(self.path):
            # save scheduled watches so we can start without error
            self._stop_watching_path()
        super(DirWatcher, self).start()

    def stop(self):
        """Stop watching and shut down threads."""
        # stop watching the path
        self._stop_watching_path()

        # stop the root observer
        self.root_observer.stop()
        self.root_watch = None

        super(DirWatcher, self).stop()

    def all_alive(self):
        """Check if all observer and emitter threads are running."""
        # check if self thread is running
        if not self.is_alive():
            return False
        # check if self emitters are running
        if not all(
            emitter.is_alive()
            and (
                not hasattr(emitter, "_inotify")
                or not emitter._inotify
                or emitter._inotify.is_alive()
            )
            for emitter in self.emitters
        ):
            return False
        # check if root observer thread is running
        if not self.root_observer.is_alive():
            return False
        # check if all root observer emitters are running
        if not all(
            emitter.is_alive()
            and (
                not hasattr(emitter, "_inotify")
                or not emitter._inotify
                or emitter._inotify.is_alive()
            )
            for emitter in self.root_observer.emitters
        ):
            return False

        return True


def _add_watchdog_group(parser):
    watchdoggroup = parser.add_argument_group(title="watchdog")
    watchdoggroup.add_argument(
        "--force_polling",
        action="store_true",
        help="""Force watchdog to use polling instead of the default observer.""",
    )
    return parser


def _build_watch_parser(Parser, *args):
    desc = "Print Digital RF file events occurring in a directory."
    parser = Parser(*args, description=desc)

    parser.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="""Data directory to monitor.
                               (default: %(default)s)""",
    )

    parser = _add_watchdog_group(parser)
    parser = list_drf._add_time_group(parser)
    parser = list_drf._add_include_group(parser)

    parser.set_defaults(func=_run_watch)

    return parser


def _run_watch(args):
    args.dir = os.path.abspath(args.dir)

    if args.starttime is not None:
        args.starttime = util.parse_identifier_to_time(args.starttime)
    if args.endtime is not None:
        args.endtime = util.parse_identifier_to_time(
            args.endtime, ref_datetime=args.starttime
        )

    # subclass DigitalRFEventHandler to just print events
    class DigitalRFPrint(DigitalRFEventHandler):
        def __init__(self, dir, force_polling=None, **kwargs):
            self.root_dir = dir
            super(DigitalRFPrint, self).__init__(**kwargs)

        def on_created(self, event):
            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            path = os.path.relpath(event.src_path, self.root_dir)
            print(f"{now} | Created {path}")

        def on_deleted(self, event):
            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            path = os.path.relpath(event.src_path, self.root_dir)
            print(f"{now} | Deleted {path}")

        def on_modified(self, event):
            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            path = os.path.relpath(event.src_path, self.root_dir)
            print(f"{now} | Modified {path}")

        def on_moved(self, event):
            msg = "{0} | Moved {1} to {2}"
            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            src_path = os.path.relpath(event.src_path, self.root_dir)
            dest_path = os.path.relpath(event.dest_path, self.root_dir)
            print(msg.format(now, src_path, dest_path))

    kwargs = vars(args).copy()
    del kwargs["func"]
    event_handler = DigitalRFPrint(**kwargs)
    observer = DirWatcher(args.dir, force_polling=args.force_polling)
    observer.schedule(event_handler, args.dir, recursive=True)
    print("Type Ctrl-C to quit.")
    observer.start()
    now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
    print(f"{now} | Monitoring {args.dir}:")
    sys.stdout.flush()
    try:
        while True:
            if not observer.all_alive():
                observer.stop()
                # catch any dead threads and create a new observer
                observer = DirWatcher(args.dir, force_polling=args.force_polling)
                observer.schedule(event_handler, args.dir, recursive=True)
                observer.start()
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        observer.stop()
        sys.stdout.write("\n")
        sys.stdout.flush()
    observer.join()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_watch_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
