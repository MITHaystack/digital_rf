# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module for watching a directory and deleting the oldest Digital RF files."""
from __future__ import absolute_import, division, print_function

import datetime
import errno
import os
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict, defaultdict, deque, namedtuple

from . import list_drf, util, watchdog_drf

__all__ = ("DigitalRFRingbufferHandler", "DigitalRFRingbuffer")


class DigitalRFRingbufferHandlerBase(watchdog_drf.DigitalRFEventHandler):
    """Base event handler for implementing a ringbuffer of Digital RF files.

    This handler tracks files but does nothing to expire them. At least one
    expirer mixin must be used with this class in order to create a complete
    ringbuffer.

    """

    FileRecord = namedtuple("FileRecord", ("key", "size", "path", "group"))

    def __init__(
        self,
        verbose=False,
        dryrun=False,
        starttime=None,
        endtime=None,
        include_drf=True,
        include_dmd=True,
    ):
        """Create a ringbuffer handler.

        Other Parameters
        ----------------
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

        """
        self.verbose = verbose
        self.dryrun = dryrun
        # separately track file groups (ch path, name) with different queues
        self.queues = defaultdict(deque)
        self.records = {}
        # acquire the record lock to modify the queue or record dicts
        self._record_lock = threading.RLock()
        super(DigitalRFRingbufferHandlerBase, self).__init__(
            starttime=starttime,
            endtime=endtime,
            include_drf=include_drf,
            include_dmd=include_dmd,
            include_drf_properties=False,
            include_dmd_properties=False,
        )

    def status(self):
        """Return status string about state of the ringbuffer."""
        nfiles = sum(len(q) for q in self.queues.values())
        return "{0} files".format(nfiles)

    def _get_file_record(self, path):
        """Return self.FileRecord tuple for file at path."""
        # get time key (seconds) from file path
        key = None
        for r in self.regexes:
            m = r.match(path)
            try:
                secs = int(m.group("secs"))
            except (AttributeError, IndexError, TypeError):
                # no match, or regex matched but there is no 'secs' in regex
                continue
            else:
                try:
                    frac = int(m.group("frac"))
                except (IndexError, TypeError):
                    frac = 0
                # key is time in milliseconds
                key = secs * 1000 + frac
                break
        if key is None:
            return

        # ringbuffer by file groups, which are a channel path and name
        group = (m.group("chpath"), m.group("name"))

        try:
            stat = os.stat(path)
        except OSError:
            if self.verbose:
                traceback.print_exc()
            return
        else:
            size = stat.st_size

        return self.FileRecord(key=key, size=size, path=path, group=group)

    def _add_to_queue(self, rec):
        """Add record to queue."""
        # find insertion index for record (queue sorted in ascending order)
        # we expect new records to go near the end (most recent)
        queue = self.queues[rec.group]
        with self._record_lock:
            for k, (kkey, kpath) in enumerate(reversed(queue)):
                if rec.key > kkey:
                    # we've found the insertion point at index k from end
                    break
                elif rec.path == kpath:
                    # already in ringbuffer, so simply return
                    return
            else:
                # new key is oldest (or queue is empty),
                # needs to be put at beginning
                queue.appendleft((rec.key, rec.path))
                k = None
            if k is not None:
                # insert record at index k
                queue.rotate(k)
                queue.append((rec.key, rec.path))
                queue.rotate(-k)

    def _remove_from_queue(self, rec):
        """Remove record from queue."""
        queue = self.queues[rec.group]
        with self._record_lock:
            queue.remove((rec.key, rec.path))

    def _expire_oldest_from_group(self, group):
        """Expire oldest record from group and delete corresponding file."""
        # oldest file is at start of sorted records deque
        # (don't just popleft on the queue because we want to call
        #  _remove_from_queue, which is overridden by subclasses)
        with self._record_lock:
            key, path = self.queues[group][0]
            rec = self.records.pop(path)
            self._remove_from_queue(rec)

        if self.verbose:
            now = datetime.datetime.utcnow().replace(microsecond=0)
            print("{0} | Expired {1}".format(now, rec.path))

        # delete file
        if not self.dryrun:
            try:
                os.remove(rec.path)
            except EnvironmentError as e:
                # Python 2 and 3 compatible substitute for FileNotFoundError
                if e.errno == errno.ENOENT:
                    # path doesn't exist like we thought it did, oh well
                    if self.verbose:
                        traceback.print_exc()
                else:
                    # try again if path still exists, otherwise it's ok to move on
                    if os.path.exists(rec.path):
                        os.remove(rec.path)
        # try to clean up directory in case it is empty
        head, tail = os.path.split(rec.path)
        try:
            os.rmdir(head)
        except OSError:
            # directory not empty, just move on
            pass

    def _expire(self, group):
        """Expire records until ringbuffer constraint is met."""
        # must override with mixins for any expiration to occur
        pass

    def _add_record(self, rec):
        """Add a record to the ringbuffer and expire old ones if necesssary."""
        with self._record_lock:
            # make sure record does not already exist, remove if it does
            if rec.path in self.records:
                if self.verbose:
                    msg = (
                        "Adding record for {0} but it already exists in"
                        " ringbuffer, modify instead."
                    ).format(rec.path)
                    print(msg)
                self._modify_record(rec)
            # add record to dict so record information can be looked up by path
            self.records[rec.path] = rec
            # add record to expiration queue
            self._add_to_queue(rec)

            if self.verbose:
                now = datetime.datetime.utcnow().replace(microsecond=0)
                print("{0} | Added {1}".format(now, rec.path))
            # expire oldest files until size constraint is met
            self._expire(rec.group)

    def _modify_record(self, rec):
        """Modify a record in the ringbuffer, return whether it was done."""
        with self._record_lock:
            if rec.path not in self.records:
                # don't have record in ringbuffer when we should, add instead
                if self.verbose:
                    msg = (
                        "Missing modified file {0} from ringbuffer, adding" " instead."
                    ).format(rec.path)
                    print(msg)
                self._add_record(rec)
                return True
            # nothing to do otherwise
            return False

    def _remove_record(self, path):
        """Remove a record from the ringbuffer."""
        # get and remove record id if path is in the ringbuffer, return if not
        with self._record_lock:
            try:
                rec = self.records.pop(path)
            except KeyError:
                # we probably got here from a FileDeletedEvent after expiring
                # an old record, but in any case, no harm to just ignore
                return
            # remove record from ringbuffer
            self._remove_from_queue(rec)

        if self.verbose:
            now = datetime.datetime.utcnow().replace(microsecond=0)
            print("{0} | Removed {1}".format(now, rec.path))

    def add_files(self, paths, sort=True):
        """Create file records from paths and add to ringbuffer."""
        # get records and add from oldest to newest by key (time)
        records = (self._get_file_record(p) for p in paths)
        # filter out invalid paths (can't extract a time, doesn't exist)
        records = (r for r in records if r is not None)
        if sort:
            records = sorted(records)
        for rec in records:
            self._add_record(rec)

    def modify_files(self, paths, sort=True):
        """Create file records from paths and update in ringbuffer."""
        # get records and add from oldest to newest by key (time)
        records = (self._get_file_record(p) for p in paths)
        # filter out invalid paths (can't extract a time, doesn't exist)
        records = (r for r in records if r is not None)
        if sort:
            records = sorted(records)
        for rec in records:
            self._modify_record(rec)

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

    def on_modified(self, event):
        """Update modified file in ringbuffer."""
        self.modify_files([event.src_path])

    def on_moved(self, event):
        """Track moved file in ringbuffer."""
        self.remove_files([event.src_path])
        self.add_files([event.dest_path])


class CountExpirer(object):
    """Ringbuffer handler mixin to track the number of files in each channel.

    When the count threshold of a channel is exceeded, the oldest files in that
    channel are deleted until the count constraint is met.

    """

    def __init__(self, *args, **kwargs):
        """Create a ringbuffer handler."""
        self.count = kwargs.pop("count")
        super(CountExpirer, self).__init__(*args, **kwargs)

    def status(self):
        """Return status string about state of the ringbuffer."""
        status = super(CountExpirer, self).status()
        try:
            max_count = max(len(q) for q in self.queues.values())
        except ValueError:
            max_count = 0
        pct_full = int(float(max_count) / self.count * 100)
        return ", ".join((status, "{0}% count".format(pct_full)))

    def _expire(self, group):
        """Expire records until file count constraint is met."""
        with self._record_lock:
            queue = self.queues[group]
            while len(queue) > self.count:
                self._expire_oldest_from_group(group)
        super(CountExpirer, self)._expire(group)


class SizeExpirer(object):
    """Ringbuffer handler mixin to track the space used by all channels.

    This expirer tracks the amount of space that new or modified files consume
    for all channels together. When the space threshold is exceeded, the oldest
    file of any channel is deleted (unless it would empty the channel) until
    the size constraint is met.

    """

    def __init__(self, *args, **kwargs):
        """Create a ringbuffer handler."""
        self.size = kwargs.pop("size")
        self.active_size = 0
        super(SizeExpirer, self).__init__(*args, **kwargs)

    def status(self):
        """Return status string about state of the ringbuffer."""
        status = super(SizeExpirer, self).status()
        pct_full = int(float(self.active_size) / self.size * 100)
        return ", ".join((status, "{0}% size".format(pct_full)))

    def _add_to_queue(self, rec):
        """Add record to queue, tracking file size."""
        with self._record_lock:
            super(SizeExpirer, self)._add_to_queue(rec)
            self.active_size += rec.size

    def _remove_from_queue(self, rec):
        """Remove record from queue, tracking file size."""
        with self._record_lock:
            super(SizeExpirer, self)._remove_from_queue(rec)
            self.active_size -= rec.size

    def _expire_oldest(self, group):
        """Expire oldest record overall, preferring group if tied."""
        with self._record_lock:
            # remove oldest regardless of group unless it would empty group,
            # but prefer `group` if tie
            removal_group = group
            # oldest file is at start of sorted records deque
            try:
                oldest_key, oldest_path = self.queues[group][0]
            except IndexError:
                oldest_key = float("inf")
            for grp in self.queues.keys():
                if grp != group:
                    queue = self.queues[grp]
                    if len(queue) > 1:
                        key, path = queue[0]
                        if key < oldest_key:
                            oldest_key = key
                            removal_group = grp
            self._expire_oldest_from_group(removal_group)

    def _expire(self, group):
        """Expire records until overall file size constraint is met."""
        with self._record_lock:
            while self.active_size > self.size:
                self._expire_oldest(group)
        super(SizeExpirer, self)._expire(group)

    def _modify_record(self, rec):
        """Modify a record in the ringbuffer."""
        with self._record_lock:
            # have parent handle cases where we actually need to add or delete
            handled = super(SizeExpirer, self)._modify_record(rec)
            if not handled:
                # if we're here, we know that the record exists and needs to
                # be modified (in this case, update the size)
                oldrec = self.records[rec.path]

                self.records[rec.path] = rec
                self.active_size -= oldrec.size
                self.active_size += rec.size

                if self.verbose:
                    now = datetime.datetime.utcnow().replace(microsecond=0)
                    print("{0} | Updated {1}".format(now, rec.path))


class TimeExpirer(object):
    """Ringbuffer handler mixin to track the time span of each channel.

    This handler tracks the sample timestamp of files in each channel. When the
    duration threshold of a channel is exceeded (newest timestamp minus
    oldest), the oldest files in the channel are deleted until the duration
    constraint is met.

    """

    def __init__(self, *args, **kwargs):
        """Create a ringbuffer handler."""
        # duration is time span in milliseconds
        self.duration = kwargs.pop("duration")
        super(TimeExpirer, self).__init__(*args, **kwargs)

    @staticmethod
    def _queue_duration(queue):
        """Get time span in milliseconds of files in a queue."""
        try:
            oldkey, _ = queue[0]
            newkey, _ = queue[-1]
        except IndexError:
            return 0
        return newkey - oldkey

    def status(self):
        """Return status string about state of the ringbuffer."""
        status = super(TimeExpirer, self).status()
        try:
            max_duration = max(self._queue_duration(q) for q in self.queues.values())
        except ValueError:
            max_duration = 0
        pct_full = int(float(max_duration) / self.duration * 100)
        return ", ".join((status, "{0}% duration".format(pct_full)))

    def _expire(self, group):
        """Expire records until time span constraint is met."""
        with self._record_lock:
            queue = self.queues[group]
            while self._queue_duration(queue) > self.duration:
                self._expire_oldest_from_group(group)
        super(TimeExpirer, self)._expire(group)


def DigitalRFRingbufferHandler(size=None, count=None, duration=None, **kwargs):
    """Create ringbuffer handler given constraints.

    Parameters
    ----------
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

    starttime : datetime.datetime
        Data covering this time or after will be included. This has no
        effect on property files.

    endtime : datetime.datetime
        Data covering this time or earlier will be included. This has no
        effect on property files.

    include_drf : bool
        If True, include Digital RF files. If False, ignore Digital RF
        files.

    include_dmd : bool
        If True, include Digital Metadata files. If False, ignore Digital
        Metadata files.

    """
    if size is None and count is None and duration is None:
        errstr = "One of `size`, `count`, or `duration` must not be None."
        raise ValueError(errstr)

    bases = (DigitalRFRingbufferHandlerBase,)
    # add mixins in this particular order for expected results
    if size is not None:
        bases = (SizeExpirer,) + bases
        kwargs["size"] = size
    if duration is not None:
        bases = (TimeExpirer,) + bases
        kwargs["duration"] = duration
    if count is not None:
        bases = (CountExpirer,) + bases
        kwargs["count"] = count

    # now create the class with the desired mixins
    docstring = """Event handler for implementing a ringbuffer of Digital RF files.

        This class inherits from a base class (DigitalRFRingbufferHandlerBase)
        and some expirer mixins determined from the class factor arguments.
        The expirers determine when a file needs to be expired from the
        ringbuffer based on size, count, or duration constraints.

        """
    cls = type("DigitalRFRingbufferHandler", bases, {"__doc__": docstring})

    return cls(**kwargs)


class DigitalRFRingbuffer(object):
    """Monitor a directory and delete old Digital RF files when space is full.

    This class combines an event handler and a file system observer. It
    monitors a directory and its subdirectories for new Digital RF and Digital
    Metadata files. When the ringbuffer threshold in size, count, or duration
    is exceeded, the oldest files are deleted until the constraint is met.

    """

    def __init__(
        self,
        path,
        size=-200e6,
        count=None,
        duration=None,
        verbose=False,
        status_interval=10,
        dryrun=False,
        starttime=None,
        endtime=None,
        include_drf=True,
        include_dmd=True,
        force_polling=False,
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

        status_interval : None | int
            Interval in seconds between printing of status updates. If None,
            do not print status updates.

        dryrun : bool
            If True, do not actually delete files when expiring them from the
            ringbuffer. Use for testing only!

        starttime : datetime.datetime
            Data covering this time or after will be included. This has no
            effect on property files.

        endtime : datetime.datetime
            Data covering this time or earlier will be included. This has no
            effect on property files.

        include_drf : bool
            If True, include Digital RF files. If False, ignore Digital RF
            files.

        include_dmd : bool
            If True, include Digital Metadata files. If False, ignore Digital
            Metadata files.

        force_polling : bool
            If True, force the watchdog to use polling instead of the default
            observer.

        """
        self.path = os.path.abspath(path)
        self.size = size
        self.count = count
        self.duration = duration
        self.verbose = verbose
        self.status_interval = status_interval
        self.dryrun = dryrun
        self.starttime = starttime
        self.endtime = endtime
        self.include_drf = include_drf
        self.include_dmd = include_dmd
        self.force_polling = force_polling
        self._start_time = None
        self._task_threads = []

        if self.size is None and self.count is None and self.duration is None:
            errstr = "One of `size`, `count`, or `duration` must not be None."
            raise ValueError(errstr)

        if not self.include_drf and not self.include_dmd:
            errstr = "One of `include_drf` or `include_dmd` must be True."
            raise ValueError(errstr)

        if self.status_interval is None:
            self.status_interval = float("inf")

        if self.size is not None:
            if self.size < 0:
                # get available space and reduce it by the (negative) size
                # value to get the actual size to use
                root = self.path
                while not os.path.isdir(root):
                    root = os.path.dirname(root)
                statvfs = os.statvfs(root)
                bytes_available = statvfs.f_frsize * statvfs.f_bavail
                if os.path.isdir(self.path):
                    existing = list_drf.ilsdrf(
                        self.path,
                        starttime=self.starttime,
                        endtime=self.endtime,
                        include_drf=self.include_drf,
                        include_dmd=self.include_dmd,
                        include_drf_properties=False,
                        include_dmd_properties=False,
                    )
                    for p in existing:
                        try:
                            bytes_available += os.path.getsize(p)
                        except OSError:
                            # catch instances where file no longer exists
                            if self.verbose:
                                traceback.print_exc()
                            return
                self.size = max(bytes_available + self.size, 0)

        self.event_handler = DigitalRFRingbufferHandler(
            size=self.size,
            count=self.count,
            duration=self.duration,
            verbose=self.verbose,
            dryrun=self.dryrun,
            starttime=self.starttime,
            endtime=self.endtime,
            include_drf=self.include_drf,
            include_dmd=self.include_dmd,
        )

        self._init_observer()

    def _init_observer(self):
        self.observer = watchdog_drf.DirWatcher(
            self.path, force_polling=self.force_polling
        )
        self.observer.schedule(self.event_handler, self.path, recursive=True)

    def _add_existing_files(self):
        """Add existing files on disk to ringbuffer."""
        # pause dispatching while we add existing files so files are added
        # to the ringbuffer in the correct order
        with self.observer.paused_dispatching():
            # add existing files to ringbuffer handler
            existing = list_drf.ilsdrf(
                self.path,
                starttime=self.starttime,
                endtime=self.endtime,
                include_drf=self.include_drf,
                include_dmd=self.include_dmd,
                include_drf_properties=False,
                include_dmd_properties=False,
            )
            # do not sort because existing will already be sorted and we
            # don't want to convert to a list
            self.event_handler.add_files(existing, sort=False)

    def start(self):
        """Start ringbuffer process."""
        self._start_time = datetime.datetime.utcnow().replace(microsecond=0)

        # start observer to add new files
        self.observer.start()

        if self.dryrun:
            print("DRY RUN (files will not be deleted):")
        now = datetime.datetime.utcnow().replace(microsecond=0)
        print("{0} | Starting {1}:".format(now, self))
        sys.stdout.flush()

        # add files that already existed before the observer started
        # (do it in another thread so we can get to join())
        thread = threading.Thread(target=self._add_existing_files)
        thread.daemon = True
        thread.start()
        self._task_threads.append(thread)

    def _verify_ringbuffer_files(self, inbuffer):
        """Verify ringbuffer's `inbuffer` set of files with files on disk."""
        # get set of all files that should be in the ringbuffer right away
        # so we duplicate as few files from new events as possible
        # events that happen while we build this file set can be duplicated
        # when we verify the ringbuffer state below, but that's ok
        ondisk = set(
            list_drf.ilsdrf(
                self.path,
                starttime=self.starttime,
                endtime=self.endtime,
                include_drf=self.include_drf,
                include_dmd=self.include_dmd,
                include_drf_properties=False,
                include_dmd_properties=False,
            )
        )

        # now any file in inbuffer that is not in ondisk is a missed or
        # duplicate deletion event, so remove those files
        deletions = inbuffer - ondisk
        self.event_handler.remove_files(deletions)

        # any file in ondisk that is not in inbuffer is a missed or duplicate
        # creation event, so add those files
        creations = ondisk - deletions
        self.event_handler.add_files(creations, sort=True)

        # any file in both ondisk and inbuffer could have a missed modify
        # event, so trigger a modify event for those files
        possibly_modified = inbuffer & ondisk
        self.event_handler.modify_files(possibly_modified, sort=True)

    def _restart(self):
        """Restart observer using existing event handlers."""
        # get list of files currently in ringbuffer before we modify it
        # so we can detect missed events from after crash until ondisk
        # file list is complete
        inbuffer = set(self.event_handler.records.keys())

        # make a new observer and start it ASAP
        self._init_observer()
        self.observer.start()

        # verify existing state of ringbuffer
        # (do it in another thread so we can get to join())
        thread = threading.Thread(
            target=self._verify_ringbuffer_files, kwargs=dict(inbuffer=inbuffer)
        )
        thread.daemon = True
        thread.start()
        self._task_threads.append(thread)

    def join(self):
        """Wait until a KeyboardInterrupt is received to stop ringbuffer."""
        try:
            while True:
                now = datetime.datetime.utcnow().replace(microsecond=0)
                interval = int((now - self._start_time).total_seconds())
                if (interval % self.status_interval) == 0:
                    status = self.event_handler.status()
                    print("{0} | ({1})".format(now, status))
                    sys.stdout.flush()

                if not self.observer.all_alive():
                    # if not all threads of the observer are alive,
                    # reinitialize and restart
                    print("Found stopped thread, reinitializing and restarting.")
                    sys.stdout.flush()
                    # first make sure all task threads have stopped
                    for thread in self._task_threads:
                        while thread.is_alive():
                            print("Waiting for task thread to finish.")
                            sys.stdout.flush()
                            thread.join(1)
                    del self._task_threads[:]
                    self._restart()

                time.sleep(1)
        except KeyboardInterrupt:
            # catch keyboard interrupt and simply exit
            pass
        finally:
            self.stop()
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.observer.join()

    def run(self):
        """Start ringbuffer and wait for a KeyboardInterrupt to stop."""
        self.start()
        self.join()

    def stop(self):
        """Stop ringbuffer process."""
        self.observer.stop()

    def __str__(self):
        """Return string describing ringbuffer."""
        amounts = []
        if self.size is not None:
            amounts.append("{0} bytes".format(self.size))
        if self.count is not None:
            amounts.append("{0} files".format(self.count))
        if self.duration is not None:
            amounts.append("{0} s".format(self.duration / 1e3))
        s = "DigitalRFRingbuffer of ({0}) in {1}".format(", ".join(amounts), self.path)
        return s


def _build_ringbuffer_parser(Parser, *args):
    desc = (
        "Enforce ringbuffer of Digital RF and Digital Metadata files. When"
        " the space threshold is exceeded, the oldest files are deleted until"
        " the size constraint is met."
    )
    parser = Parser(*args, description=desc)

    parser.add_argument("path", help="Directory in which to enforce ringbuffer.")
    parser.add_argument(
        "-z",
        "--size",
        default=None,
        help="""Size of ringbuffer, in bytes or using unit symbols (e.g 100GB).
                Negative values are used to indicate all available space except
                the given amount. (default: -200MB if no count or duration)""",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="""Max file count for each channel. (default: %(default)s)""",
    )
    parser.add_argument(
        "-l",
        "--duration",
        default=None,
        help="""Max duration for each channel in seconds.
                (default: %(default)s)""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the name new/deleted files and the space consumed.",
    )
    parser.add_argument(
        "-p",
        "--status_interval",
        type=int,
        default=10,
        help="""Interval in seconds between printing of status updates.
                (default: %(default)s)""",
    )
    parser.add_argument(
        "-n",
        "--dryrun",
        action="store_true",
        help="Do not delete files when expiring them from the ringbuffer.",
    )

    parser = list_drf._add_time_group(parser)

    includegroup = parser.add_argument_group(title="include")
    includegroup.add_argument(
        "--nodrf",
        dest="include_drf",
        action="store_false",
        help="""Do not ringbuffer Digital RF HDF5 files.
                (default: False)""",
    )
    includegroup.add_argument(
        "--nodmd",
        dest="include_dmd",
        action="store_false",
        help="""Do not ringbuffer Digital Metadata HDF5 files.
                (default: False)""",
    )

    parser = watchdog_drf._add_watchdog_group(parser)

    parser.set_defaults(func=_run_ringbuffer)

    return parser


def _run_ringbuffer(args):
    import signal

    # parse size string into number of bytes
    if args.size == "":
        args.size = None
    if args.size is not None:
        suffixes = OrderedDict(
            [
                ("B", 1),
                ("KB", 1000 ** 1),
                ("KiB", 1024 ** 1),
                ("MB", 1000 ** 2),
                ("MiB", 1024 ** 2),
                ("GB", 1000 ** 3),
                ("GiB", 1024 ** 3),
                ("TB", 1000 ** 4),
                ("TiB", 1024 ** 4),
                ("PB", 1000 ** 5),
                ("PiB", 1024 ** 5),
            ]
        )
        m = re.match(r"(?P<num>\-?\d+\.?\d*)(?P<suf>\D*)", args.size)
        if not m:
            raise ValueError(
                "Size string not recognized. " "Use number followed by suffix."
            )
        sizenum = eval(m.group("num"))
        suf = m.group("suf").strip()
        if not suf:
            args.size = sizenum
        elif suf in suffixes:
            args.size = sizenum * suffixes[suf]
        else:
            raise ValueError(
                "Size suffix not recognized. Use one of:\n"
                "{0}".format(list(suffixes.keys()))
            )
    elif args.count is None and args.duration is None:
        args.size = -200e6

    # evaluate duration to float, from seconds to milliseconds
    if args.duration is not None:
        args.duration = float(eval(args.duration)) * 1e3

    if args.starttime is not None:
        args.starttime = util.parse_identifier_to_time(args.starttime)
    if args.endtime is not None:
        args.endtime = util.parse_identifier_to_time(
            args.endtime, ref_datetime=args.starttime
        )

    kwargs = vars(args).copy()
    del kwargs["func"]

    # handle SIGTERM (getting killed) gracefully by calling sys.exit
    def sigterm_handler(signal, frame):
        print("Killed")
        sys.stdout.flush()
        sys.exit(128 + signal)

    signal.signal(signal.SIGTERM, sigterm_handler)

    ringbuffer = DigitalRFRingbuffer(**kwargs)
    print("Type Ctrl-C to quit.")
    ringbuffer.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = _build_ringbuffer_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
