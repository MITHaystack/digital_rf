"""Module for watching a directory and deleting the oldest Digital RF files."""

import os
import re
import sys
import time
import uuid
from collections import OrderedDict, deque, namedtuple

from watchdog.observers import Observer

from .list_drf import lsdrf, sortkey_drf
from .watchdog_drf import DigitalRFEventHandler

__all__ = (
    'DigitalRFRingbufferHandler', 'DigitalRFRingbuffer',
)


class DigitalRFRingbufferHandler(DigitalRFEventHandler):
    """Event handler for implementing a ringbuffer of Digital RF files.

    This handler tracks the amount of space that new or modified files consume.
    When the space threshold is exceeded, the oldest files are deleted until
    the size constraint is met.

    """

    FileRecord = namedtuple('FileRecord', ('key', 'size', 'path'))

    def __init__(
        self, size, verbose=False, dryrun=False,
        include_drf=True, include_dmd=False, include_metadata=False,
    ):
        """Create ringbuffer handler given a size in bytes."""
        self.size = size
        self.verbose = verbose
        self.dryrun = dryrun
        self.queue = deque()
        self.record_ids = {}
        self.records = {}
        self.active_size = 0
        super(DigitalRFRingbufferHandler, self).__init__(
            include_drf=include_drf, include_dmd=include_dmd,
            include_metadata=include_metadata,
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

        try:
            stat = os.stat(path)
        except OSError:
            size = None
        else:
            size = stat.st_size

        return self.FileRecord(key=key, size=size, path=path)

    def _expire_oldest(self):
        # oldest file is at end of sorted records deque
        key, rid = self.queue.pop()
        rec = self.records.pop(rid)
        del self.record_ids[rec.path]
        self.active_size -= rec.size
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
                int(float(self.active_size)/self.size*100), rec.path, rec.size,
            ))
        else:
            sys.stdout.write('-')
            sys.stdout.flush()

    def _add_to_queue(self, key, rid):
        # find insertion index for record (queue sorted in descending order)
        # we expect new records to go near the beginning (most recent)
        k = 0  # in case files queue is empty
        for k, (kkey, krid) in enumerate(self.queue):
            if key > kkey:
                # we've found the insertion point at index k
                break
            elif rid == krid:
                # already in ringbuffer, so simply return
                return
        # insert record at index k
        self.queue.rotate(-k)
        self.queue.appendleft((key, rid))
        self.queue.rotate(k)

    def _add_record(self, rec):
        # create unique id for record
        rid = uuid.uuid4()
        # add id to dict so it can be looked up by path
        self.record_ids[rec.path] = rid
        # add record to dict so record information can be looked up by id
        self.records[rid] = rec
        # add record to expiration queue
        self._add_to_queue(rec.key, rid)
        # account for size of added record
        self.active_size += rec.size
        if self.verbose:
            print('{0}%: Added {1} ({2} bytes).'.format(
                int(float(self.active_size)/self.size*100), rec.path, rec.size,
            ))
        else:
            sys.stdout.write('+')
            sys.stdout.flush()
        # expire oldest files until size constraint is met
        while self.active_size > self.size:
            self._expire_oldest()

    def _remove_record(self, path):
        # get and remove record id if path is in the ringbuffer, return if not
        try:
            rid = self.record_ids.pop(path)
        except KeyError:
            return
        # remove record from ringbuffer, accounting for lost size
        rec = self.records.pop(rid)
        self.queue.remove((rec.key, rid))
        self.active_size -= rec.size
        if self.verbose:
            print('{0}%: Removed {1} ({2} bytes).'.format(
                int(float(self.active_size)/self.size*100), rec.path, rec.size,
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

    def on_drf_moved(self, event):
        """Track moved file in ringbuffer."""
        # we assume that a DRF file's time is immutable regardless of rename
        # so all we need to do is update the path in the record
        rid = self.record_ids.pop(event.src_path)
        self.record_ids[event.dest_path] = rid
        rec = self.records.pop(rid)
        newrec = self.FileRecord(
            key=rec.key, size=rec.size, path=event.dest_path,
        )
        self.records[rid] = newrec
        if self.verbose:
            print('{0}%: Tracked {1} move to {2}.'.format(
                int(float(self.active_size)/self.size*100), rec.path,
                newrec.path,
            ))
        else:
            sys.stdout.write('>')
            sys.stdout.flush()

    def on_created(self, event):
        """Add new file to ringbuffer."""
        self.add_files([event.src_path])

    def on_deleted(self, event):
        """Remove file from ringbuffer if it was deleted externally."""
        self.remove_files([event.src_path])

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
                int(float(self.active_size)/self.size*100), rec.path, rec.size,
                newrec.size,
            ))
        else:
            sys.stdout.write('.')
            sys.stdout.flush()


class DigitalRFRingbuffer(object):
    """Monitor a directory and delete old Digital RF files when space is full.

    This class combines an event handler and a file system observer. It
    monitors a directory and its subdirectories for new Digital RF and Digital
    Metadata files. When the total space consumed by all of the files exceeds
    the ringbuffer size, the oldest files are deleted until the size constraint
    is met.

    """

    def __init__(self, path, size=-200e6, verbose=False, dryrun=False):
        """Create Digital RF ringbuffer object. Use start/run method to begin.

        Parameters
        ----------

        path : str
            Directory in which the ringbuffer is enforced.

        size : float | int
            Size of the ringbuffer in bytes. Negative values are used to
            indicate all available space except the given amount.


        Other Parameters
        ----------------

        verbose : bool
            If True, print debugging info about the files that are created and
            deleted and how much space they consume.

        dryrun : bool
            If True, do not actually delete files when expiring them from the
            ringbuffer. Use for testing only!

        """
        self.path = os.path.abspath(path)
        self.size = size
        self.verbose = verbose
        self.dryrun = dryrun

        if self.size < 0:
            # get available space and reduce it by the (negative) size value
            # to get the actual size to use
            statvfs = os.statvfs(self.path)
            bytes_available = statvfs.f_frsize*statvfs.f_bavail
            existing = lsdrf(
                self.path, include_dmd=False, include_metadata=False,
            )
            bytes_available += sum([os.stat(p).st_size for p in existing])
            self.size = max(bytes_available + self.size, 0)

        self.event_handler = DigitalRFRingbufferHandler(
            size=self.size, verbose=verbose, dryrun=dryrun,
        )
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.path, recursive=True)

    def start(self):
        """Start ringbuffer process."""
        # start observer to add new files
        self.observer.start()

        # add existing files to ringbuffer handler
        existing = lsdrf(self.path, include_dmd=False, include_metadata=False)
        existing.sort(key=sortkey_drf)
        self.event_handler.add_files(existing)

    def join(self):
        """Wait until a KeyboardInterrupt is received to stop ringbuffer."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
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
        '-s', '--size', default='-200MB',
        help='''Size of ringbuffer, in bytes or using unit symbols (e.g 100GB).
                Negative values are used to indicate all available space except
                the given amount. (default: %(default)s)''',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print the name new/deleted files and the space consumed.',
    )
    parser.add_argument(
        '-n', '--dryrun', action='store_true',
        help='Do not delete files when expiring them from the ringbuffer.',
    )

    parser.set_defaults(func=_run_ringbuffer)

    return parser


def _run_ringbuffer(args):
    # parse size string into number of bytes
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

    ringbuffer = DigitalRFRingbuffer(
        args.path, args.size, args.verbose, args.dryrun,
    )
    if args.dryrun:
        print('DRY RUN (files will not be deleted):')
    print('Enforcing ringbuffer of {0} bytes in {1}.'.format(
        ringbuffer.size, ringbuffer.path,
    ))
    print('Type Ctrl-C to quit.')
    ringbuffer.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ringbuffer_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
