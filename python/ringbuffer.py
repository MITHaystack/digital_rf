# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for watching a directory and deleting the oldest Digital RF files."""

import os
import re
import sys
import time
import uuid
from collections import OrderedDict, defaultdict, deque, namedtuple

from watchdog.observers import Observer

from .list_drf import lsdrf, sortkey_drf
from .watchdog_drf import DigitalRFEventHandler

__all__ = (
    'DigitalRFRingbufferHandler', 'DigitalRFSizeRingbufferHandler',
    'DigitalRFTimeRingbufferHandler', 'DigitalRFRingbuffer',
)


class DigitalRFRingbufferHandler(DigitalRFEventHandler):
    """Event handler for implementing a ringbuffer of Digital RF files.

    This handler tracks the number of files in each channel. When the count
    threshold of a channel is exceeded, the oldest files in that channel are
    deleted until the count constraint is met.

    """

    FileRecord = namedtuple('FileRecord', ('key', 'size', 'path', 'group'))

    def __init__(
        self, threshold, verbose=False, dryrun=False,
        include_drf=True, include_dmd=True,
    ):
        """Create ringbuffer handler given a max file count for each group."""
        self.threshold = threshold
        self.verbose = verbose
        self.dryrun = dryrun
        # separately track file groups (ch path, name) with different queues
        self.queues = defaultdict(deque)
        self.record_ids = {}
        self.records = {}
        super(DigitalRFRingbufferHandler, self).__init__(
            include_drf=include_drf, include_dmd=include_dmd,
            include_properties=False,
        )

    def _get_file_record(self, path):
        """Return self.FileRecord tuple for file at path."""
        # get time key (seconds) from file path
        key = None
        for r in self.regexes:
            m = r.match(path)
            try:
                secs = int(m.group('secs'))
            except (AttributeError, IndexError, TypeError):
                # no match, or regex matched but there is no 'secs' in regex
                continue
            else:
                try:
                    frac = int(m.group('frac'))
                except (IndexError, TypeError):
                    frac = 0
                # key is time in milliseconds
                key = secs*1000 + frac
                break
        if key is None:
            return

        # ringbuffer by file groups, which are a channel path and name
        group = (m.group('chpath'), m.group('name'))

        try:
            stat = os.stat(path)
        except OSError:
            size = None
        else:
            size = stat.st_size

        return self.FileRecord(key=key, size=size, path=path, group=group)

    def _add_to_queue(self, rec, rid):
        # find insertion index for record (queue sorted in ascending order)
        # we expect new records to go near the end (most recent)
        queue = self.queues[rec.group]
        k = 0  # in case files queue is empty
        for k, (kkey, krid) in enumerate(reversed(queue)):
            if rec.key > kkey:
                # we've found the insertion point at index k from end
                break
            elif rid == krid:
                # already in ringbuffer, so simply return
                return
        # insert record at index k
        queue.rotate(k)
        queue.append((rec.key, rid))
        queue.rotate(-k)
        return len(queue)

    def _remove_from_queue(self, rec, rid):
        queue = self.queues[rec.group]
        queue.remove((rec.key, rid))
        return len(queue)

    def _expire_oldest(self, group):
        # oldest file is at end of sorted records deque
        # (don't just popleft on the queue because we want to call
        #  _remove_from_queue, which is overridden by subclasses)
        key, rid = self.queues[group][0]
        rec = self.records.pop(rid)
        del self.record_ids[rec.path]
        active_amount = self._remove_from_queue(rec, rid)

        # delete file
        if not self.dryrun:
            os.remove(rec.path)
        # try to clean up directory in case it is empty
        head, tail = os.path.split(rec.path)
        try:
            os.rmdir(head)
        except OSError:
            # directory not empty, just move on
            pass

        if self.verbose:
            print('{0}%: Expired {1} ({2} bytes).'.format(
                int(float(active_amount)/self.threshold*100),
                rec.path, rec.size,
            ))
        else:
            sys.stdout.write('-')
            sys.stdout.flush()

        return active_amount

    def _add_record(self, rec):
        # create unique id for record
        rid = uuid.uuid4()
        # add id to dict so it can be looked up by path
        self.record_ids[rec.path] = rid
        # add record to dict so record information can be looked up by id
        self.records[rid] = rec
        # add record to expiration queue
        active_amount = self._add_to_queue(rec, rid)

        if self.verbose:
            print('{0}%: Added {1} ({2} bytes).'.format(
                int(float(active_amount)/self.threshold*100),
                rec.path, rec.size,
            ))
        else:
            sys.stdout.write('+')
            sys.stdout.flush()
        # expire oldest files until size constraint is met
        while active_amount > self.threshold:
            active_amount = self._expire_oldest(rec.group)

    def _remove_record(self, path):
        # get and remove record id if path is in the ringbuffer, return if not
        try:
            rid = self.record_ids.pop(path)
        except KeyError:
            return
        # remove record from ringbuffer
        rec = self.records.pop(rid)
        active_amount = self._remove_from_queue(rec, rid)

        if self.verbose:
            print('{0}%: Removed {1} ({2} bytes).'.format(
                int(float(active_amount)/self.threshold*100),
                rec.path, rec.size,
            ))
        else:
            sys.stdout.write('-')
            sys.stdout.flush()

    def add_files(self, paths):
        """Create file records from paths and add to ringbuffer."""
        # get records and add from oldest to newest by key (time)
        record_list = [self._get_file_record(p) for p in paths]
        # filter out invalid paths (can't extract a time from the regex)
        record_list = [r for r in record_list if r is not None]
        for rec in sorted(record_list):
            self._add_record(rec)

    def remove_files(self, paths):
        """Retrieve file records from paths and remove from ringbuffer."""
        for p in paths:
            self._remove_record(p)

    def on_created(self, event):
        """Add new file to ringbuffer."""
        self.add_files([event.src_path])

    def on_deleted(self, event):
        """Remove file from ringbuffer if it was deleted externally."""
        self.remove_files([event.src_path])

    def on_drf_moved(self, event):
        """Track moved file in ringbuffer."""
        self.remove_files([event.src_path])
        self.add_files([event.dest_path])

    def on_modified(self, event):
        """Update modified file in ringbuffer."""
        pass


class DigitalRFSizeRingbufferHandler(DigitalRFRingbufferHandler):
    """Event handler for implementing a ringbuffer of Digital RF files.

    This handler tracks the amount of space that new or modified files consume
    for all channels together. When the space threshold is exceeded, the oldest
    files in the channel with the new file are deleted until the size
    constraint is met.

    """

    def __init__(self, size, **kwargs):
        """Create ringbuffer handler given a size in bytes."""
        self.active_size = 0
        super(DigitalRFSizeRingbufferHandler, self).__init__(
            threshold=size, **kwargs
        )

    def _add_to_queue(self, rec, rid):
        super(DigitalRFSizeRingbufferHandler, self)._add_to_queue(
            rec, rid,
        )
        self.active_size += rec.size
        return self.active_size

    def _remove_from_queue(self, rec, rid):
        super(DigitalRFSizeRingbufferHandler, self)._remove_from_queue(
            rec, rid,
        )
        self.active_size -= rec.size
        return self.active_size

    def on_modified(self, event):
        """Update modified file in ringbuffer."""
        rid = self.record_ids[event.src_path]
        rec = self.records.pop(rid)
        newrec = self._get_file_record(event.src_path)
        self.records[rid] = newrec
        self.active_size -= rec.size
        self.active_size += newrec.size
        if self.verbose:
            print('{0}%: Updated {1} ({2} to {3} bytes).'.format(
                int(float(self.active_size)/self.threshold*100),
                rec.path, rec.size, newrec.size,
            ))


class DigitalRFTimeRingbufferHandler(DigitalRFRingbufferHandler):
    """Event handler for implementing a ringbuffer of Digital RF files.

    This handler tracks the sample timestamp of files in each channel. When the
    duration threshold of a channel is exceeded (newest timestamp minus
    oldest), the oldest files in the channel are deleted until the duration
    constraint is met.

    """

    def __init__(self, duration, **kwargs):
        """Create ringbuffer handler given a duration in milliseconds."""
        super(DigitalRFTimeRingbufferHandler, self).__init__(
            threshold=duration, **kwargs
        )

    def _add_to_queue(self, rec, rid):
        super(DigitalRFTimeRingbufferHandler, self)._add_to_queue(
            rec, rid,
        )
        queue = self.queues[rec.group]
        oldkey, _ = queue[0]
        newkey, _ = queue[-1]
        return (newkey - oldkey)

    def _remove_from_queue(self, rec, rid):
        super(DigitalRFTimeRingbufferHandler, self)._remove_from_queue(
            rec, rid,
        )
        queue = self.queues[rec.group]
        try:
            oldkey, _ = queue[0]
            newkey, _ = queue[-1]
        except IndexError:
            return -1
        return (newkey - oldkey)


class DigitalRFRingbuffer(object):
    """Monitor a directory and delete old Digital RF files when space is full.

    This class combines an event handler and a file system observer. It
    monitors a directory and its subdirectories for new Digital RF and Digital
    Metadata files. When the ringbuffer threshold in size, count, or duration
    is exceeded, the oldest files are deleted until the constraint is met.

    """

    def __init__(
        self, path, size=-200e6, count=None, duration=None,
        verbose=False, dryrun=False, include_drf=True, include_dmd=True,
    ):
        """Create Digital RF ringbuffer object. Use start/run method to begin.

        Parameters
        ----------

        path : str
            Directory in which the ringbuffer is enforced.

        size : float | int | None
            Size of the ringbuffer in bytes. Negative values are used to
            indicate all available space except the given amount. If None, no
            size constraint is used.

        count : int | None
            Maximum number of files *for each channel*. If None, no count
            constraint is used.

        duration : int | float | None
            Maximum time span *for each channel* in milliseconds. If None, no
            duration constraint is used.


        Other Parameters
        ----------------

        verbose : bool
            If True, print debugging info about the files that are created and
            deleted and how much space they consume.

        dryrun : bool
            If True, do not actually delete files when expiring them from the
            ringbuffer. Use for testing only!

        include_drf : bool
            If True, include Digital RF files. If False, ignore Digital RF
            files.

        include_dmd : bool
            If True, include Digital Metadata files. If False, ignore Digital
            Metadata files.

        """
        self.path = os.path.abspath(path)
        self.size = size
        self.count = count
        self.duration = duration
        self.verbose = verbose
        self.dryrun = dryrun
        self.include_drf = include_drf
        self.include_dmd = include_dmd

        if self.size is None and self.count is None and self.duration is None:
            errstr = 'One of `size`, `count`, or `duration` must not be None.'
            raise ValueError(errstr)

        if not self.include_drf and not self.include_dmd:
            errstr = 'One of `include_drf` or `include_dmd` must be True.'
            raise ValueError(errstr)

        self.event_handlers = []

        if self.count is not None:
            handler = DigitalRFRingbufferHandler(
                threshold=self.count, verbose=self.verbose, dryrun=self.dryrun,
                include_drf=self.include_drf, include_dmd=self.include_dmd,
            )
            self.event_handlers.append(handler)

        if self.duration is not None:
            handler = DigitalRFTimeRingbufferHandler(
                duration=self.duration, verbose=self.verbose,
                dryrun=self.dryrun,
                include_drf=self.include_drf, include_dmd=self.include_dmd,
            )
            self.event_handlers.append(handler)

        if self.size is not None:
            if self.size < 0:
                # get available space and reduce it by the (negative) size
                # value to get the actual size to use
                statvfs = os.statvfs(self.path)
                bytes_available = statvfs.f_frsize*statvfs.f_bavail
                existing = lsdrf(
                    self.path, include_drf=self.include_drf,
                    include_dmd=self.include_dmd, include_properties=False,
                )
                bytes_available += sum([os.stat(p).st_size for p in existing])
                self.size = max(bytes_available + self.size, 0)

            handler = DigitalRFSizeRingbufferHandler(
                size=self.size, verbose=self.verbose, dryrun=self.dryrun,
                include_drf=self.include_drf, include_dmd=self.include_dmd,
            )
            self.event_handlers.append(handler)

        self.observer = Observer()
        for handler in self.event_handlers:
            self.observer.schedule(handler, self.path, recursive=True)

    def start(self):
        """Start ringbuffer process."""
        # start observer to add new files
        self.observer.start()

        # add existing files to ringbuffer handler
        existing = lsdrf(
            self.path, include_drf=self.include_drf,
            include_dmd=self.include_dmd, include_properties=False,
        )
        existing.sort(key=sortkey_drf)
        for handler in self.event_handlers:
            handler.add_files(existing)
            # since handler might delete some existing files, update existing
            # file list using handler's list of files
            existing = sorted(handler.record_ids.keys(), key=sortkey_drf)

    def join(self):
        """Wait until a KeyboardInterrupt is received to stop ringbuffer."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            sys.stdout.write('\n')
            sys.stdout.flush()
        self.observer.join()

    def run(self):
        """Start ringbuffer and wait for a KeyboardInterrupt to stop."""
        self.start()
        self.join()

    def stop(self):
        """Stop ringbuffer process."""
        self.observer.stop()


def _build_ringbuffer_parser(Parser, *args):
    desc = (
        'Enforce ringbuffer of Digital RF and Digital Metadata files. When'
        ' the space threshold is exceeded, the oldest files are deleted until'
        ' the size constraint is met.'
    )
    parser = Parser(*args, description=desc)

    parser.add_argument(
        'path',
        help='Directory in which to enforce ringbuffer.',
    )
    parser.add_argument(
        '-z', '--size', default=None,
        help='''Size of ringbuffer, in bytes or using unit symbols (e.g 100GB).
                Negative values are used to indicate all available space except
                the given amount. (default: -200MB if no count or duration)''',
    )
    parser.add_argument(
        '-c', '--count', type=int, default=None,
        help='''Max file count for each channel. (default: %(default)s)''',
    )
    parser.add_argument(
        '-l', '--duration', default=None,
        help='''Max duration for each channel in seconds.
                (default: %(default)s)''',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print the name new/deleted files and the space consumed.',
    )
    parser.add_argument(
        '-n', '--dryrun', action='store_true',
        help='Do not delete files when expiring them from the ringbuffer.',
    )

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

    parser.set_defaults(func=_run_ringbuffer)

    return parser


def _run_ringbuffer(args):
    # parse size string into number of bytes
    if args.size == '':
        args.size = None
    if args.size is not None:
        suffixes = OrderedDict([
            ('B', 1),
            ('KB', 1000**1),
            ('KiB', 1024**1),
            ('MB', 1000**2),
            ('MiB', 1024**2),
            ('GB', 1000**3),
            ('GiB', 1024**3),
            ('TB', 1000**4),
            ('TiB', 1024**4),
            ('PB', 1000**5),
            ('PiB', 1024**5),
        ])
        m = re.match('(?P<num>\-?\d+\.?\d*)(?P<suf>\D*)', args.size)
        if not m:
            raise ValueError('Size string not recognized. '
                             'Use number followed by suffix.')
        sizenum = eval(m.group('num'))
        suf = m.group('suf').strip()
        if not suf:
            args.size = sizenum
        elif suf in suffixes:
            args.size = sizenum*suffixes[suf]
        else:
            raise ValueError('Size suffix not recognized. Use one of:\n'
                             '{0}'.format(suffixes.keys()))
    elif args.count is None and args.duration is None:
        args.size = -200e6

    # evaluate duration to float, from seconds to milliseconds
    if args.duration is not None:
        args.duration = float(eval(args.duration))*1e3

    dargs = vars(args)
    del dargs['func']
    ringbuffer = DigitalRFRingbuffer(**dargs)
    if args.dryrun:
        print('DRY RUN (files will not be deleted):')
    print((
        'Enforcing ringbuffer of ({0} bytes, {1} files, {2} s) in {3}.'
    ).format(
        ringbuffer.size, ringbuffer.count, ringbuffer.duration/1e3,
        ringbuffer.path,
    ))
    print('Type Ctrl-C to quit.')
    ringbuffer.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ringbuffer_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
