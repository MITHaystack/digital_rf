#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""digital_rf_archive.py is a tool for archiving Digital RF data.

$Id$
"""
from __future__ import absolute_import, division, print_function

import argparse
import datetime
import glob
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback

from six.moves import urllib


def archive_subdirectory_local_local(args):
    """archive_subdirectory_local_local is the method called by a pool of multiprocessing process to archive
    one subdirectory from a local machine to a local machine.

    Inputs: args, a list of:
        source - path to high level source directory
        channel - channel name
        subdir - basename of subdirectory to archive
        dt - datetime of subdirectory name
        nextDT - datetime of next subdirectory.  None if no next subdirectory.
        is_metadata - True if metadata, False if RF data
        channel_dest - path to where channel will be backed up
        startDT - start datetime of archive
        endDT - end datetime of archive
        gzip - gzip (0-9) 0 is no compression
        verbose - True for verbose output, False otherwise

    Returns: the number of files backed up
    """
    # constants
    _rf_file_glob = "rf@[0-9]*.[0-9][0-9][0-9].h5"
    _metadata_file_glob = "*@[0-9]*.h5"

    file_count = 0

    (
        source,
        channel,
        subdir,
        dt,
        nextDT,
        is_metadata,
        channel_dest,
        startDT,
        endDT,
        gzip,
        verbose,
    ) = args

    # determine if this channel is compressed
    if is_metadata:
        archive_files = glob.glob(
            os.path.join(source, channel, subdir, _metadata_file_glob)
        )
    else:
        archive_files = glob.glob(os.path.join(source, channel, subdir, _rf_file_glob))
    if len(archive_files) == 0:
        if verbose:
            print("skipping because no files found in %s" % (subdir))
        return 0

    # decide if this is a full copy, partial copy, or skip
    if dt > endDT:
        if verbose:
            print(
                "skipping %s because after period %s - %s"
                % (subdir, str(startDT), str(endDT))
            )
        return 0

    if dt < startDT and nextDT != None:
        if nextDT < startDT:
            if verbose:
                print(
                    "skipping %s because entirely before period %s - %s"
                    % (subdir, str(startDT), str(endDT))
                )
            return 0

    if gzip > 0:
        is_compressed = _is_h5_file_compressed(archive_files[0])
    else:
        is_compressed = False

    # the only options left now are complete or partial copies
    # check if complete copy needed
    if startDT < dt and nextDT != None:
        if nextDT < endDT:
            if (is_compressed or gzip == 0) or is_metadata:
                if verbose:
                    print("%s - complete copy" % (subdir))
                shutil.copytree(
                    os.path.join(source, channel, subdir),
                    os.path.join(channel_dest, subdir),
                )
                file_count += len(glob.glob(os.path.join(source, channel, subdir, "*")))
                return file_count
    if verbose:
        print("%s partial copy or copy with added compression" % (subdir))
    target_dir = os.path.join(channel_dest, subdir)
    os.mkdir(target_dir)
    if is_metadata:
        archive_files = glob.glob(
            os.path.join(source, channel, subdir, _metadata_file_glob)
        )
    else:
        archive_files = glob.glob(os.path.join(source, channel, subdir, _rf_file_glob))
    archive_files.sort()  # allows us to stop checking when a file moves out of time range
    for archive_file in archive_files:
        basename = os.path.basename(archive_file)
        ts = datetime.datetime.utcfromtimestamp(
            float(basename[basename.find("@") + 1 : -3])
        )
        if ts >= startDT and ts <= endDT:
            # check if we need to compress it
            if not is_compressed and gzip > 0 and not is_metadata:
                src_file = os.path.join(target_dir, basename)
                cmd = "h5repack -i %s -o %s -f rf_data:GZIP=%i" % (
                    archive_file,
                    src_file,
                    gzip,
                )
                try:
                    subprocess.check_call(cmd.split())
                    file_count += 1
                except:
                    traceback.print_exc()
                    print("cmd <%s> failed" % (cmd))
            else:
                shutil.copyfile(
                    archive_file,
                    os.path.join(target_dir, os.path.basename(archive_file)),
                )
                file_count += 1
        elif ts > endDT:
            return file_count

    return file_count


def archive_subdirectory_local_remote(args):
    """archive_subdirectory_local_remote is the method called by a pool of multiprocessing process to archive
    one subdirectory from a local machine to a remote machine.

    Inputs: args, a list of:
        source - path to high level source directory
        channel - channel name
        subdir - basename of subdirectory to archive
        dt - datetime of subdirectory name
        nextDT - datetime of next subdirectory.  None if no next subdirectory.
        is_metadata - True if metadata, False if RF data
        channel_dest - path to where channel will be backed up
        startDT - start datetime of archive
        endDT - end datetime of archive
        gzip - gzip (0-9) ) is no compression
        verbose - True for verbose output, False otherwise

    Returns: the number of files backed up
    """
    # constants
    _rf_file_glob = "rf@[0-9]*.[0-9][0-9][0-9].h5"
    _metadata_file_glob = "*@[0-9]*.h5"

    file_count = 0

    (
        source,
        channel,
        subdir,
        dt,
        nextDT,
        is_metadata,
        channel_dest,
        startDT,
        endDT,
        gzip,
        verbose,
    ) = args

    # determine if this channel is compressed
    if is_metadata:
        archive_files = glob.glob(
            os.path.join(source, channel, subdir, _metadata_file_glob)
        )
    else:
        archive_files = glob.glob(os.path.join(source, channel, subdir, _rf_file_glob))
    if len(archive_files) == 0:
        if verbose:
            print("skipping because no files found in %s" % (subdir))
        return 0

    # decide if this is a full copy, partial copy, or skip
    if dt > endDT:
        if verbose:
            print(
                "skipping %s because after period %s - %s"
                % (subdir, str(startDT), str(endDT))
            )
        return 0

    if dt < startDT and nextDT != None:
        if nextDT < startDT:
            if verbose:
                print(
                    "skipping %s because entirely before period %s - %s"
                    % (subdir, str(startDT), str(endDT))
                )
            return 0

    if gzip > 0:
        is_compressed = _is_h5_file_compressed(archive_files[0])
    else:
        is_compressed = False

    # the only options left now are complete or partial copies
    # check if complete copy needed
    if startDT < dt and nextDT != None:
        if nextDT < endDT:
            if (is_compressed or gzip == 0) or is_metadata:
                if verbose:
                    print("%s - complete copy" % (subdir))
                cmd = "scp -q -r %s %s" % (
                    os.path.join(source, channel, subdir),
                    os.path.join(channel_dest, subdir),
                )
                try:
                    subprocess.check_call(cmd.split())
                    file_count += len(
                        glob.glob(os.path.join(source, channel, subdir, "*"))
                    )
                except subprocess.CalledProcessError:
                    raise IOError(
                        "Failed to copy entire subdirectory with cmd <%s>" % (cmd)
                    )
                return file_count

    if verbose:
        print("%s partial copy or copy with added compression" % (subdir))
    target_dir = os.path.join(channel_dest, subdir)
    # make channel dir locally, then copy
    subdir_local = os.path.join("/tmp", subdir)
    try:
        os.system("rm -rf %s" % (subdir_local))
    except:
        pass
    os.mkdir(subdir_local)
    cmd = "scp -q -r %s %s" % (subdir_local, channel_dest)
    try:
        subprocess.check_call(cmd.split())
    except subprocess.CalledProcessError:
        raise IOError("Failed to create remote subdirectory dir with cmd <%s>" % (cmd))
    try:
        os.system("rm -rf %s" % (subdir_local))
    except:
        pass

    if is_metadata:
        archive_files = glob.glob(
            os.path.join(source, channel, subdir, _metadata_file_glob)
        )
    else:
        archive_files = glob.glob(os.path.join(source, channel, subdir, _rf_file_glob))
    archive_files.sort()  # allows us to stop checking when a file moves out of time range
    for archive_file in archive_files:
        basename = os.path.basename(archive_file)
        ts = datetime.datetime.utcfromtimestamp(
            float(basename[basename.find("@") + 1 : -3])
        )
        if ts >= startDT and ts <= endDT:
            # if we need to compress it, make a copy first
            if not is_compressed and gzip > 0 and not is_metadata:
                src_file = os.path.join("/tmp", basename)
                try:
                    os.remove(src_file)
                except:
                    pass
                cmd = "h5repack -i %s -o %s -f rf_data:GZIP=%i" % (
                    archive_file,
                    src_file,
                    gzip,
                )
                try:
                    subprocess.check_call(cmd.split())
                except:
                    traceback.print_exc()
                    print("cmd <%s> failed" % (cmd))
                    continue
            else:
                src_file = archive_file
            cmd = "scp -q %s %s" % (src_file, target_dir)
            try:
                subprocess.check_output(cmd.split())
                file_count += 0
            except subprocess.CalledProcessError:
                raise IOError("Failed to copy archive_file with cmd <%s>" % (cmd))
            if src_file != archive_file:
                os.remove(src_file)
        elif ts > endDT:
            break

    return file_count


def _is_h5_file_compressed(filename):
    """_is_h5_file_compressed returns True if Hdf5 file gzip compresses, False otherwise
    """
    cmd = "h5stat %s" % (filename)
    output = subprocess.check_output(cmd.split())
    for line in output.split("\n"):
        if line.find("GZIP") != -1:
            items = line.split()
            level = int(items[-1])
            if level > 0 and level < 10:
                return True
    return False


class archive(object):
    """archive is a class to archive a digital rf data set
    """

    def __init__(
        self,
        startDT,
        endDT,
        source,
        dest,
        pool=1,
        channels=None,
        is_metadata=False,
        gzip=1,
        verbose=False,
        check_only=False,
    ):
        """
        __init__ will create a Digital RF archive

        Inputs:
            startDT - start datetime for the data archive
            endDT end datetime for the data archive
            source - string defining data source. Either a path, or a url.
                Empty string is defined as $PWD
            dest - where archive is to be created.  May be a path, or a scp
                type path (user@host:/path) If scp-type, keypair must allow
                scp without passwords
            pool - number of multiprocessing processes to use. Default is 1
            channels - a list of channels to archive.  If None (the default),
                archive all channels
            is_metadata - False if standard rf data, True if metadata with modified file
                naming convention.  That convention is *@<timestamp>*h5.  Metadata is stored
                under metadata subdirectory, rf_data under rf_data subdirectory
            gzip - integer specifying level of gzip compression to apply if file not already compressed.
                Default is 1.  0 for no compression.
            verbose - if True, print one line for each subdirectory.  If False (the default),
                no output except for errors.
            check_only - if True, simply check if enough space. Raises error if not local source and local dest.

        Attributes:
            self._source_type - local if local input data, remote if remote
            self._chan_dist - dictionary with keys = channel found, values = tuple of
                1: ordered list of tuples of
                    1. paths or urls to subdirectories,
                    2. datetime of that subdirectory name. and
                2: list of metadata files in channel directory
        """
        # constants
        self._sub_directory_glob = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        self._rf_file_glob = "rf@[0-9]*.[0-9][0-9][0-9].h5"
        self._properties_glob = "*_properties.h5"
        self._metadata_file_glob = "*@[0-9]*.h5"
        self._metadata_dir = "metadata"
        self._rf_dir = "rf_data"

        # verify arguments
        if endDT <= startDT:
            print("endDT <%s> must be after startDT <%s>" % (str(endDT), str(startDT)))
            sys.exit(-1)

        # save args
        self.startDT = startDT
        self.endDT = endDT
        self.source = source
        self.dest = dest
        self.pool_count = pool
        self.channels = channels
        self.is_metadata = bool(is_metadata)
        self.gzip = int(gzip)
        self.verbose = bool(verbose)
        self.check_only = bool(check_only)

        self._source_type = "local"
        if len(self.source) > 5:
            if self.source[0:5] == "http:":
                self._source_type = "remote"

        if self._source_type == "local":
            # make sure source is full path
            if self.source[0] != "/":
                self.source = os.path.join(os.getcwd(), self.source)

        self._dest_type = "local"
        hyphen = self.dest.find(":")
        slash = self.dest.find("/")
        if hyphen != -1:
            if slash == -1:
                self._dest_type = "remote"
            elif hyphen < slash:
                self._dest_type = "remote"

        if self.check_only:
            if self._dest_type != "local" or self._source_type != "local":
                raise ValueError(
                    "check_only flag only valid for local source and destination"
                )

        if self.verbose:
            # set up a timer
            t = time.time()

        if self._source_type == "local":
            self._chan_dict = self.get_chan_dict_local()
        else:
            self._chan_dict = self.get_chan_dict_remote()

        # verify all channels exist if specified
        if self.channels:
            for channel in self.channels:
                if channel not in self._chan_dict:
                    raise IOError("Requested channel %s not found" % (channel))
        else:
            self.channels = list(self._chan_dict.keys())  # archive all channels

        self._hostname = self._get_hostname()
        self._top_level_dest_dir = self.get_top_level_dest_dir()

        # set _next_level_dir to rf_data or metadata
        if self.is_metadata:
            self._next_level_dir = self._metadata_dir
        else:
            self._next_level_dir = self._rf_dir

        if self._dest_type == "local" and self._source_type == "local":
            # verify enough disk space
            needed_GB, avail_GB = self.estimate_disk_space()
            if self.verbose or self.check_only:
                print(
                    "Estimated GB to use = %f GB, available = %f GB"
                    % (needed_GB, avail_GB)
                )
                if self.check_only:
                    sys.exit(-1)
            if needed_GB > avail_GB:
                raise IOError(
                    "aborting because needed GB %f greater than available GB %f"
                    % (needed_GB, avail_GB)
                )

        self._create_top_level_dest_dirs()

        # create a pool of workers
        self._pool = multiprocessing.Pool(pool)

        for channel in self.channels:
            if self.verbose:
                print("archiving channel %s" % (channel))
            result = self._archive_channel(channel)

        if self.verbose:
            print("digital_rf_archive took %f seconds" % (time.time() - t))

    def _archive_channel(self, channel):
        """_archive_channel will archive the channel.  Returns True if any data archived, False otherwise
        """
        # to simplify the code, break into for subroutines depending on
        # self._source_type self._dest_type
        if self._source_type == "local":
            if self._dest_type == "local":
                return self._archive_channel_local_local(channel)
            else:
                return self._archive_channel_local_remote(channel)

        else:
            raise ValueError("nyi")

    def _archive_channel_local_local(self, channel):
        """_archive_channel_local_local archives local input data to a local destination
        """
        is_compressed = None  # not yet known
        subdir_list = self._chan_dict[channel]
        # make channel dir in self._top_level_dest_dir/self._next_level_dir
        channel_dest = os.path.join(
            self.dest, self._top_level_dest_dir, self._next_level_dir, channel
        )
        os.mkdir(channel_dest)

        # first copy all the properties
        for properties_file in subdir_list[1]:
            if self.verbose:
                print("archiving properties file %s" % (properties_file))
            shutil.copyfile(
                os.path.join(self.source, channel, properties_file),
                os.path.join(channel_dest, os.path.basename(properties_file)),
            )

        # build iterable of arguments
        args_list = []
        for i, items in enumerate(subdir_list[0]):
            subdir, dt = items
            if i < len(subdir_list[0]) - 1:
                nextDT = subdir_list[0][i + 1][1]
            else:
                nextDT = None
            # distribute the subdirectories to the pool
            args_list.append(
                [
                    self.source,
                    channel,
                    subdir,
                    dt,
                    nextDT,
                    self.is_metadata,
                    channel_dest,
                    self.startDT,
                    self.endDT,
                    self.gzip,
                    self.verbose,
                ]
            )

        file_count_list = self._pool.map(archive_subdirectory_local_local, args_list)

        if self.verbose:
            print("%i files backed up in channel %s" % (sum(file_count_list), channel))
        if sum(file_count_list) == 0:
            print("WARNING: No files backed up in channel %s" % (channel))

    def _archive_channel_local_remote(self, channel):
        """_archive_channel_local_remote archives local input data to a remote destination
        """
        is_compressed = None  # not yet known
        subdir_list = self._chan_dict[channel]
        file_count = 0

        # make channel dir in self._top_level_dest_dir
        top_level_dest = os.path.join(
            self.dest, self._top_level_dest_dir, self._next_level_dir
        )
        channel_dest = os.path.join(
            self.dest, self._top_level_dest_dir, self._next_level_dir, channel
        )
        # make channel dir locally, then copy
        channel_local = os.path.join("/tmp", channel)
        try:
            os.system("rm -rf %s" % (channel_local))
        except:
            pass
        os.mkdir(channel_local)
        cmd = "scp -q -r %s %s" % (channel_local, top_level_dest)
        try:
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError:
            raise IOError("Failed to create remote channel dir with cmd <%s>" % (cmd))
        try:
            os.system("rm -rf %s" % (channel_local))
        except:
            pass

        # first copy all the properties
        for properties_file in subdir_list[1]:
            if self.verbose:
                print("archiving properties file %s" % (properties_file))
            cmd = "scp -q %s %s" % (
                os.path.join(self.source, channel, properties_file),
                channel_dest,
            )
            try:
                subprocess.check_call(cmd.split())
            except subprocess.CalledProcessError:
                raise IOError("Failed to copy properties file with cmd <%s>" % (cmd))

        # build iterable of arguments
        args_list = []
        for i, items in enumerate(subdir_list[0]):
            subdir, dt = items
            if i < len(subdir_list[0]) - 1:
                nextDT = subdir_list[0][i + 1][1]
            else:
                nextDT = None
            # distribute the subdirectories to the pool
            args_list.append(
                [
                    self.source,
                    channel,
                    subdir,
                    dt,
                    nextDT,
                    self.is_metadata,
                    channel_dest,
                    self.startDT,
                    self.endDT,
                    self.gzip,
                    self.verbose,
                ]
            )

        file_count_list = self._pool.map(archive_subdirectory_local_remote, args_list)

        if self.verbose:
            print("%i files backed up in channel %s" % (sum(file_count_list), channel))
        if sum(file_count_list) == 0:
            print("WARNING: No files backed up in channel %s" % (channel))

    def _get_hostname(self):
        """_get_hostname returns the hostname of the local system or the remote system of the source data
        """
        if self._source_type == "local":
            return socket.gethostname()
        else:
            o = urllib.parse.urlparse(self.source)
            return o.netloc

    def _create_top_level_dest_dirs(self):
        """_create_top_level_dest_dirs will create the dest top level directory if needed. Raise IOError
        if it can't be created. Also creates subdirectories <metadata> and <rf_data>
        under that directory.
        """
        full_path = os.path.join(self.dest, self._top_level_dest_dir)
        if self._dest_type == "local":
            if not os.path.isdir(self.dest):
                raise IOError("%s does not exist or is not a directory" % (self.dest))
            if not os.access(full_path, os.W_OK):
                os.mkdir(full_path)
                os.mkdir(os.path.join(full_path, self._rf_dir))
                os.mkdir(os.path.join(full_path, self._metadata_dir))
        else:
            # try to create the three remote directories - even if they already
            # exist, should succeed
            for i, this_dir in enumerate(
                (self._top_level_dest_dir, self._metadata_dir, self._rf_dir)
            ):
                localDir = os.path.join("/tmp", this_dir)
                try:
                    os.system("rm -rf %s" % (localDir))
                except:
                    pass
                os.mkdir(localDir)
                # now scp better succeed
                if i == 0:
                    cmd = "scp -q -r %s %s" % (localDir, self.dest)
                else:
                    cmd = "scp -q -r %s %s" % (
                        localDir,
                        os.path.join(self.dest, self._top_level_dest_dir),
                    )
                try:
                    subprocess.check_call(cmd.split())
                except subprocess.CalledProcessError:
                    raise IOError(
                        "Failed to create remote top level dir with cmd <%s>" % (cmd)
                    )
                try:
                    os.system("rm -rf %s" % (localDir))
                except:
                    pass

    def _is_h5_file_compressed(self, filename):
        cmd = "h5stat %s" % (filename)
        output = subprocess.check_output(cmd.split())
        for line in output.split("\n"):
            if line.find("GZIP") != -1:
                items = line.split()
                level = int(items[-1])
                if level > 0 and level < 10:
                    return True
        return False

    def estimate_disk_space(self):
        """estimate_disk_space returns a tuple of (needed_GB, avail_GB) as floats for local dest"""
        # constants
        _rf_file_glob = "rf@[0-9]*.[0-9][0-9][0-9].h5"
        _metadata_file_glob = "*@[0-9]*.h5"

        # estimate available disk space
        result = os.statvfs(self.dest)
        avail_GB = (result.f_bavail * result.f_frsize) / (1024 * 1024 * 1024.0)

        # estimate needed space
        needed_GB = 0.0
        compress = None  # will be estimated with test file
        for channel in self.channels:
            subdir_list = self._chan_dict[channel]

            # properties
            for properties_file in subdir_list[1]:
                needed_GB += os.path.getsize(
                    os.path.join(self.source, channel, properties_file)
                )

            # rf data
            for i, items in enumerate(subdir_list[0]):
                subdir, dt = items
                if i < len(subdir_list[0]) - 1:
                    nextDT = subdir_list[0][i + 1][1]
                else:
                    nextDT = None

                if dt > self.endDT:
                    break
                if nextDT != None:
                    if nextDT <= self.startDT:
                        # too early
                        continue

                if self.is_metadata:
                    archive_files = glob.glob(
                        os.path.join(self.source, channel, subdir, _metadata_file_glob)
                    )
                    compress = 1.0
                else:
                    archive_files = glob.glob(
                        os.path.join(self.source, channel, subdir, _rf_file_glob)
                    )
                    if compress == None and len(archive_files) > 0:
                        compress = self._test_compression(archive_files[0])

                for archive_file in archive_files:
                    needed_GB += compress * os.path.getsize(archive_file)

        return (needed_GB / (1024 * 1024 * 1024.0), avail_GB)

    def _test_compression(self, testHdf5File):
        """_test_compression returns the gzip compression ratio for test file testHdf5File
        """
        if self.gzip == 0:
            return 1.0
        resultFile = "/tmp/test.h5"
        try:
            os.remove(resultFile)
        except:
            pass
        cmd = "h5repack -i %s -o %s -f rf_data:GZIP=%i" % (
            testHdf5File,
            resultFile,
            self.gzip,
        )
        subprocess.check_call(cmd.split())
        result = float(os.path.getsize(resultFile)) / os.path.getsize(testHdf5File)
        os.remove(resultFile)
        return result

    def get_top_level_dest_dir(self):
        """returns the top level dest directory to write to in format
        <hostname>_YYYY-MM-DDTHH:MM:SS_YYYY-MM-DDTHH:MM:SS
        """
        startDTStr = self.startDT.strftime("%Y%m%dT%H%M%S")
        endDTStr = self.endDT.strftime("%Y%m%dT%H%M%S")
        return "%s_%s_%s" % (self._hostname, startDTStr, endDTStr)

    def get_chan_dict_local(self):
        """get_chan_dict_local returns a dictionary with keys = channels found,
            values = list of two lists - 1: list of
            subdirectory basenames, 2: list of properties files in channel directory
        """
        ret_dict = {}
        subdirectories = glob.glob(
            os.path.join(self.source, "*", self._sub_directory_glob)
        )
        subdirectories.sort()
        for subdirectory in subdirectories:
            basename = os.path.basename(subdirectory)
            dt = datetime.datetime.strptime(basename, "%Y-%m-%dT%H-%M-%S")
            channel_path = os.path.dirname(subdirectory)
            channel = os.path.basename(channel_path)
            if channel not in ret_dict:
                ret_dict[channel] = [[], []]
            ret_dict[channel][0].append((basename, dt))

        # now get properties files
        properties_files = glob.glob(
            os.path.join(self.source, "*", self._properties_glob)
        )
        for properties_file in properties_files:
            basename = os.path.basename(properties_file)
            channel_path = os.path.dirname(properties_file)
            channel = os.path.basename(channel_path)
            if channel in ret_dict:
                ret_dict[channel][1].append(basename)

        return ret_dict

    def get_chan_dict_remote(self):
        """get_chan_dict_local returns a dictionary with keys = channels found,
            values = list of two lists - 1: list of
            subdirectory basenames, 2: list of metadata files in channel directory
        """
        return "nyi"


### main begins here ###
if __name__ == "__main__":

    # command line interface
    parser = argparse.ArgumentParser(
        description="digital_rf_archive.py is a tool for archiving Digital RF data.."
    )
    parser.add_argument(
        "--startDT",
        metavar="start datetime string",
        help="Start UT datetime for archiving in format YYYY-MM-DDTHH:MM:SS",
        required=True,
    )
    parser.add_argument(
        "--endDT",
        metavar="end datetime string",
        help="End UT datetime for archiving in format YYYY-MM-DDTHH:MM:SS",
        required=True,
    )
    parser.add_argument(
        "--source",
        metavar="Top level directory or url",
        help="Top level directory or url containing the Digital RF channels or metadata channels to archive",
        required=True,
    )
    parser.add_argument(
        "--dest",
        metavar="Full path to destination",
        help="Full path to destination directory.  May be local, or remote in scp form (user@host:)",
        required=True,
    )
    parser.add_argument(
        "--pool",
        metavar="Pool multiprocessing count.",
        help="Number of multiprocesses processes to use in the pool.  Default=1.",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--channel",
        metavar="Optional channel name",
        action="append",
        help="Optional channel name to backup.  May be given more than once.  Default is all channels.",
    )
    parser.add_argument(
        "--is_metadata",
        action="store_true",
        default=False,
        help="Set this flag if metadata, as opposed to rf_data, being archived.",
    )
    parser.add_argument(
        "--gzip",
        metavar="GZIP compression 1-9.",
        help="Level of GZIP compression 1-9.  Default=1. 0 for no compression",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Get one line of output for each subdirectory archived.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        default=False,
        help="Set this flag to simply check if enough space. Only possible for local source and local dest.",
    )
    args = parser.parse_args()

    # create datetimes
    try:
        startDT = datetime.datetime.strptime(args.startDT, "%Y-%m-%dT%H:%M:%S")
    except:
        print(
            "startDT <%s> not in expected YYYY-MM-DDTHH:MM:SS format" % (args.startDT)
        )
        sys.exit(-1)
    try:
        endDT = datetime.datetime.strptime(args.endDT, "%Y-%m-%dT%H:%M:%S")
    except:
        print("endDT <%s> not in expected YYYY-MM-DDTHH:MM:SS format" % (args.endDT))
        sys.exit(-1)

    # verify gzip in range
    if args.gzip < 0 or args.gzip > 9:
        raise ValueError(
            "gzip must be 0 (no compression), or 1-9, not %i" % (args.gzip)
        )

    # call main class
    archive(
        startDT,
        endDT,
        args.source,
        args.dest,
        args.pool,
        args.channel,
        args.is_metadata,
        args.gzip,
        args.verbose,
        args.check_only,
    )
