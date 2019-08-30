#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Record beacon signals of specified satellites in Digital RF.

Satellite and recording parameters are specified in .ini configuration files.
Example configurations are included along with this script.

"""
from __future__ import absolute_import, division, print_function

import datetime
import math
import optparse
import os
import string
import subprocess
import sys
import time
import traceback

import dateutil.parser
import ephem
import numpy as np
import pytz
from digital_rf import DigitalMetadataWriter

from six.moves import configparser


class ExceptionString(Exception):
    """ Simple exception handling string """

    def __str__(self):
        return repr(self.args[0])


def doppler_shift(frequency, relativeVelocity):
    """
    DESCRIPTION:
        This function calculates the doppler shift of a given frequency when actual
        frequency and the relative velocity is passed.
        The function for the doppler shift is f' = f - f*(v/c).
    INPUTS:
        frequency (float)        = satlitte's beacon frequency in Hz
        relativeVelocity (float) = Velocity at which the satellite is moving
                                   towards or away from observer in m/s
    RETURNS:
        Param1 (float)           = The frequency experienced due to doppler shift in Hz
    AFFECTS:
        None
    EXCEPTIONS:
        None
    DEPENDENCIES:
        ephem.Observer(...), ephem.readtle(...)
    Note: relativeVelocity is positive when moving away from the observer
          and negative when moving towards
    """
    return frequency - frequency * (relativeVelocity / 3e8)


def satellite_rise_and_set(
    opt, obsLat, obsLong, obsElev, objName, tle1, tle2, startDate
):
    """
    DESCRIPTION:
        This function take in the observers latitude, longitude, elevation as well
        as the object's name and TLE line 1 and 2 to calculate the next closest rise
        and set times from the given start date. Returns an array of values.
    INPUTS:
        obsLat (string)                  = Latitude of Observer in degrees represented as strings
        obsLong (string)                 = Longitude of Observer in degrees represented as strings
        obsElev (float)                  = Elevation of Observer in meters
        objName (string)                 = Name of the satellite
        tle1 (string)                    = First line of TLE
        tle2 (string)                    = Second line of TLE
        startDate (string or ephem.date) = The date from which next closest rise and set are to be
                                           found in radians that print as degrees

    RETURNS:
        Param1 (ephem.date)              = Rise time of satellite in 'yyyy/mm/dd hh:mm:ss'
        Param2 (ephem.date)              = Half time between rise and set in 'yyyy/mm/dd hh:mm:ss'
        Param3 (ephem.date)              = Set time of satellite in 'yyyy/mm/dd hh:mm:ss'
    AFFECTS:
        None
    EXCEPTIONS:
        None
    DEPENDENCIES:
        ephem.Observer(...), ephem.readtle(...)
    """
    obsLoc = ephem.Observer()
    obsLoc.lat = obsLat
    obsLoc.long = obsLong
    obsLoc.elev = obsElev
    obsLoc.date = startDate

    if opt.debug:
        print("dbg location: ", obsLoc)

        print("dbg tle1: ", tle1)
        print("dbg tle2: ", tle2)

    satObj = ephem.readtle(objName, tle1, tle2)

    if opt.debug:
        print("dbg object: ", satObj)

    satObj.compute(obsLoc)  # computes closest next rise time to given date

    pinfo = obsLoc.next_pass(satObj)

    return (pinfo[0], pinfo[2], pinfo[4])


def satellite_values_at_time(opt, obsLat, obsLong, obsElev, objName, tle1, tle2, date):
    """
    DESCRIPTION:
        This function take in the observers latitude, longitude, elevation as well
        as the object's name and TLE line 1 and 2 to calculate various values
        at a given date. It returns an array of values
    INPUTS:
        obsLat (string)                  = Latitude of Observer in degrees represented as strings
        obsLong (string)                 = Longitude of Observer in degrees represented as strings
        obsElev (float)                  = Elevation of Observer in meters
        objName (string)                 = Name of the satellite
        tle1 (string)                    = First line of TLE
        tle2 (string)                    = Second line of TLE
        date (string or ephem.date)      = The date from which next closest rise and set are to be
                                           found in radians that print as degrees

    RETURNS:
        Param1 (ephem.angle)             = satellite's latitude in radians that print as degrees
        Param2 (ephem.angle)             = satellite's longitude in radians that print as degrees
        Param3 (float)                   = satellite's current range from the observer in meters
        Param4 (float)                   = satellite's current range_velocity from the observer in m/s
        Param5 (ephem.angle)             = satellite's current azimuth in radians that print as degrees
        Param6 (ephem.angle)             = satellite's current altitude in radians that print as degrees
        Param7 (ephem.angle)             = satellite's right ascention in radians that print as hours of arc
        Param8 (ephem.angle)             = satellite's declination in radians that print as degrees
        Param9 (float)                   = satellite's elevation in meters from sea level
    AFFECTS:
        None
    EXCEPTIONS:
        None
    DEPENDENCIES:
        ephem.Observer(...), ephem.readtle(...)
    """
    obsLoc = ephem.Observer()
    obsLoc.lat = obsLat
    obsLoc.long = obsLong
    obsLoc.elev = obsElev
    obsLoc.date = date

    satObj = ephem.readtle(objName, tle1, tle2)
    satObj.compute(obsLoc)

    if opt.debug:
        print(
            "\tLatitude: %s, Longitude %s, Range: %gm, Range Velocity: %gm/s"
            % (satObj.sublat, satObj.sublong, satObj.range, satObj.range_velocity)
        )
        print(
            "\tAzimuth: %s, Altitude: %s, Elevation: %gm"
            % (satObj.az, satObj.alt, satObj.elevation)
        )
        print("\tRight Ascention: %s, Declination: %s" % (satObj.ra, satObj.dec))

    return (
        satObj.sublat,
        satObj.sublong,
        satObj.range,
        satObj.range_velocity,
        satObj.az,
        satObj.alt,
        satObj.ra,
        satObj.dec,
        satObj.elevation,
    )


def max_satellite_bandwidth(
    opt,
    obsLat,
    obsLong,
    obsElev,
    objName,
    tle1,
    tle2,
    startDate,
    endDate,
    interval,
    beaconFreq,
):
    """
    DESCRIPTION:
        The function calls the satellite_bandwidth function over and over for however many rises and sets occur
        during the [startDate, endDate]. The max bandwidth is then returned.
    INPUTS:
        obsLat (string)                  = Latitude of Observer in degrees represented as strings
        obsLong (string)                 = Longitude of Observer in degrees represented as strings
        obsElev (float)                  = Elevation of Observer in meters
        objName (string)                 = Name of the satellite
        tle1 (string)                    = First line of TLE
        tle2 (string)                    = Second line of TLE
        startDate (string or ephem.date) = The date/time at which to find the first cycle
        endDate (string or ephem.date)   = The date/time at which to stop looking for a cycle
        interval (float)                = The rate at which to sample during one rise/set cycle same format
                                           as time
        beaconFreq (float)               = The frequency of the beacon

    RETURNS:
        Param1 (float)                   = The max bandwidth of the satellite in the given range
                                           of start and end dates
    AFFECTS:
        None
    EXCEPTIONS:
        None
    DEPENDENCIES:
        None
    """
    maxBandwidth = 0
    (satRise, satTransit, satSet) = satellite_rise_and_set(
        opt, obsLat, obsLong, obsElev, objName, tle1, tle2, startDate
    )
    if satRise == satTransit == satSet:
        return 0

    while satRise < endDate:
        (objBandwidth, shiftedFrequencies) = satellite_bandwidth(
            opt,
            obsLat,
            obsLong,
            obsElev,
            objName,
            tle1,
            tle2,
            satRise,
            satSet,
            interval,
            beaconFreq,
        )
        if objBandwidth > maxBandwidth:
            maxBandwidth = objBandwidth
        (satRise, satTransit, satSet) = satellite_rise_and_set(
            opt,
            obsLat,
            obsLong,
            obsElev,
            objName,
            tle1,
            tle2,
            satSet + ephem.minute * 5.0,
        )
        # print "Name: %s, Rise Time: %s, Transit Time: %s, Set Time: %s" % (objName, ephem.date(satRise-ephem.hour*4.0), ephem.date(satTransit-ephem.hour*4.0), ephem.date(satSet-ephem.hour*4.0))

    return maxBandwidth


def satellite_bandwidth(
    opt,
    obsLat,
    obsLong,
    obsElev,
    objName,
    tle1,
    tle2,
    satRise,
    satSet,
    interval,
    beaconFreq,
):
    """
    DESCRIPTION:
        The function finds the bandwidth of a satellite pass
    INPUTS:
        obsLat (string)                  = Latitude of Observer in degrees represented as strings
        obsLong (string)                 = Longitude of Observer in degrees represented as strings
        obsElev (float)                  = Elevation of Observer in meters
        objName (string)                 = Name of the satellite
        tle1 (string)                    = First line of TLE
        tle2 (string)                    = Second line of TLE
        satRise (string or ephem.date)  = The time at which the satellite rises above horizon
        satSet (string or ephem.date)   = The time at which the satellite sets
        interval (float)                = The rate at which to sample during one rise/set cycle in seconds
        beaconFreq (float)               = The frequency of the beacon

    RETURNS:
        Param1 (float)                   = The bandwidth of the satellite during the rise/set cycle
        Param2 (list)                    = All the frequencies during the rise/set cycle sampled
                                           by given interval
    AFFECTS:
        None
    EXCEPTIONS:
        None
    DEPENDENCIES:
        ephem.date(...), ephem.hour, ephem.minute, ephem.second
    """
    currTime = satRise
    dopplerFrequencies = []
    dopplerBandwidth = []

    if opt.debug:
        print("satellite_bandwidth ", currTime, satSet, interval)

    while (currTime.triple())[2] < (satSet.triple())[
        2
    ]:  # the 2nd index of the returned tuple has the fraction of the day
        try:
            (
                sublat,
                sublong,
                range_val,
                range_velocity,
                az,
                alt,
                ra,
                dec,
                elevation,
            ) = satellite_values_at_time(
                opt, obsLat, obsLong, obsElev, objName, tle1, tle2, currTime
            )
            (dopplerFreq) = doppler_shift(beaconFreq, range_velocity)
            dopplerFrequencies.append(dopplerFreq)
            dopplerBandwidth.append(dopplerFreq - beaconFreq)
            currTime = currTime + ephem.second * interval
            currTime = ephem.date(currTime)
        except Exception as eobj:
            exp_str = str(ExceptionString(eobj))
            print("exception: %s." % (exp_str))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print(lines)

    if opt.debug:
        print("# DF:", np.array(dopplerFrequencies) / 1e6, " MHz")
        print("# OB:", np.array(dopplerBandwidth) / 1e3, " kHz")

    return (np.array(dopplerBandwidth), np.array(dopplerFrequencies))


def __read_config__(inifile):
    """
    DESCRIPTION:
        The function parses the given file and returns a dictionary with the values.
    INPUTS:
        inifile (string) = the name of the file to be read including path

    RETURNS:
        For an object config :

        Dictionary with name given by [Section] each of which contains:
            ObsLat              Decimal observer latitude
            ObsLong             Decimal observer longitude
            ObsElev             Decimal observer elevation
            ObjName             String object name
            TLE1                String of TLE line 1 in standard format.
            TLE2                String of TLE line 2 in standard format.
            BeaconFreq          Array of floating point beacon frequencies in Hz.

        For a radio config :
            TBD

    AFFECTS:
        None
    EXCEPTIONS:
        None
    DEPENDENCIES:
        Use module re for the simple regex used in parsing the file.

    Note:

        Example object format

        [PROPCUBE_MERRY]
        ObsLat=42.623108
        ObsLong=-71.489069
        ObsElev=150.0
        ObjName="PROPCUBE_MERRY"
        TLE1="1 90735U          16074.41055570 +.00001915 +00000-0 +22522-3 0 00790"
        TLE2="2 90735 064.7823 174.3149 0209894 234.3073 123.8339 14.73463371009286"
        BeaconFreq=[380.0e6,2.38e9]
        StartDate="2016/03/21 14:00:00"
        Interval="00:00:10"

        Example radio format :

        TBD

    """

    objects = {}

    print("# loading config ", inifile)
    cparse = configparser.ConfigParser()
    cparse.read(inifile)

    for s in cparse.sections():
        vals = cparse.options(s)
        cfg = {}
        for v in vals:
            cfg[v] = cparse.get(s, v)
        objects[s] = cfg

    return objects


def get_next_object(opt, site, objects, ctime):
    """ Not too efficent but works for now. """
    rise_list = {}

    for obj in objects:
        obj_id = obj
        obj_info = objects[obj]

        if opt.debug:
            print("# object ", obj_id, " @ ", ctime)
            print("# obj_info", obj_info)

        site_name = site["site"]["name"]
        site_tag = site["site"]["tag"]
        obs_lat = site["site"]["latitude"]
        obs_long = site["site"]["longitude"]
        obs_elev = float(site["site"]["elevation"])
        obj_name = obj_info["name"]
        obj_tle1 = obj_info["tle1"][1:-1]
        obj_tle2 = obj_info["tle2"][1:-1]
        obj_freqs = np.array(string.split(obj_info["frequencies"], ","), np.float32)
        c_dtime = datetime.datetime.utcfromtimestamp(ctime)
        c_ephem_time = ephem.Date(c_dtime)

        (sat_rise, sat_transit, sat_set) = satellite_rise_and_set(
            opt, obs_lat, obs_long, obs_elev, obj_name, obj_tle1, obj_tle2, c_ephem_time
        )

        if sat_set <= sat_rise or sat_transit <= sat_rise or sat_set <= sat_transit:
            continue

        rise_list[sat_rise] = obj

    if opt.debug:
        print(" rise list : ", rise_list)

    keys = list(rise_list.keys())

    if opt.debug:
        print(" rise keys : ", keys)

    keys.sort()

    if opt.debug:
        print(" sorted : ", keys)
        print(" selected : ", rise_list[keys[0]])

    return rise_list[keys[0]]


def ephemeris_passes(opt, st0, et0):
    """
    DESCRIPTION:
        Finds passes from the start time to the end time given the options. Will
        implement a bash script or execute on the command line.
    USAGE:
        ephemeris_passes(opt, st0, et0)
    INPUTS:
        opt             command line arguments
        st0             unix time start time
        et0             unix time end time

    RETURNS:
        None
    AFFECTS:
        Prints all the passes
    EXCEPTIONS:
        None
    DEPENDENCIES:
        ephem
    """

    passes = {}

    objects = __read_config__(opt.config)

    site = __read_config__(opt.site)

    if opt.verbose:
        print("# got objects ", objects)
        print("# got radio site ", site)
        print("\n")

    ctime = st0
    etime = et0

    last_sat_rise = ctime

    while ctime < etime:

        obj = get_next_object(opt, site, objects, ctime)

        obj_id = obj
        obj_info = objects[obj]

        if opt.debug:
            print("# object ", obj_id, " @ ", ctime)
            print("# obj_info", obj_info)

        site_name = site["site"]["name"]
        site_tag = site["site"]["tag"]
        obs_lat = site["site"]["latitude"]
        obs_long = site["site"]["longitude"]
        obs_elev = float(site["site"]["elevation"])
        obj_name = obj_info["name"]
        obj_tle1 = obj_info["tle1"][1:-1]
        obj_tle2 = obj_info["tle2"][1:-1]
        obj_freqs = np.array(string.split(obj_info["frequencies"], ","), np.float32)
        c_dtime = datetime.datetime.utcfromtimestamp(ctime)
        c_ephem_time = ephem.Date(c_dtime)

        try:
            (sat_rise, sat_transit, sat_set) = satellite_rise_and_set(
                opt,
                obs_lat,
                obs_long,
                obs_elev,
                obj_name,
                obj_tle1,
                obj_tle2,
                c_ephem_time,
            )

            if sat_set <= sat_rise or sat_transit <= sat_rise or sat_set <= sat_transit:
                continue

            if not last_sat_rise == sat_rise:
                (
                    sub_lat,
                    sub_long,
                    sat_range,
                    sat_velocity,
                    az,
                    el,
                    ra,
                    dec,
                    alt,
                ) = satellite_values_at_time(
                    opt,
                    obs_lat,
                    obs_long,
                    obs_elev,
                    obj_name,
                    obj_tle1,
                    obj_tle2,
                    sat_transit,
                )
                (obj_bandwidth, obj_doppler) = satellite_bandwidth(
                    opt,
                    obs_lat,
                    obs_long,
                    obs_elev,
                    obj_name,
                    obj_tle1,
                    obj_tle2,
                    sat_rise,
                    sat_set,
                    op.interval,
                    obj_freqs,
                )
                last_sat_rise = sat_rise
                if opt.debug:
                    print(
                        "time : ",
                        c_ephem_time,
                        sat_set,
                        (sat_set - c_ephem_time) * 60 * 60 * 24,
                    )
                ctime = ctime + (sat_set - c_ephem_time) * 60 * 60 * 24

                if opt.el_mask:
                    el_val = np.rad2deg(el)
                    el_mask = np.float(opt.el_mask)

                    if opt.debug:
                        print("# el_val ", el_val, " el_mask ", el_mask)

                    if el_val < el_mask:  # check mask here!
                        continue

                # This should really go out as digital metadata into the recording location

                print("# Site : %s " % (site_name))
                print("# Site tag : %s " % (site_tag))
                print("# Object Name: %s" % (obj_name))
                print(
                    "# observer @ latitude : %s, longitude : %s, elevation : %s m"
                    % (obs_lat, obs_long, obs_elev)
                )

                print(
                    "# GMT -- Rise Time: %s, Transit Time: %s, Set Time: %s"
                    % (sat_rise, sat_transit, sat_set)
                )
                print(
                    "# Azimuth: %f deg, Elevation: %f deg, Altitude: %g km"
                    % (np.rad2deg(az), np.rad2deg(el), alt / 1000.0)
                )
                print(
                    "# Frequencies: %s MHz, Bandwidth: %s kHz"
                    % (
                        obj_freqs / 1.0e6,
                        obj_bandwidth[np.argmax(obj_bandwidth)] / 1.0e3,
                    )
                )

                pass_md = {
                    "obj_id": obj_id,
                    "rise_time": sat_rise,
                    "transit_time": sat_transit,
                    "set_time": sat_set,
                    "azimuth": np.rad2deg(az),
                    "elevation": np.rad2deg(el),
                    "altitude": alt,
                    "doppler_frequency": obj_doppler,
                    "doppler_bandwidth": obj_bandwidth,
                }

                if opt.schedule:
                    d = sat_rise.tuple()
                    rise_time = "%04d%02d%02d_%02d%02d" % (d[0], d[1], d[2], d[3], d[4])

                    offset_rise = ephem.date(sat_rise - ephem.minute)
                    d = offset_rise.tuple()
                    offset_rise_time = "%04d-%02d-%02dT%02d:%02d:%02dZ" % (
                        d[0],
                        d[1],
                        d[2],
                        d[3],
                        d[4],
                        int(d[5]),
                    )

                    offset_set = ephem.date(sat_set + ephem.minute)
                    d = offset_set.tuple()
                    offset_set_time = "%04d-%02d-%02dT%02d:%02d:%02dZ" % (
                        d[0],
                        d[1],
                        d[2],
                        d[3],
                        d[4],
                        int(d[5]),
                    )

                    cmd_lines = []
                    radio_channel = string.split(site["radio"]["channel"][1:-1], ",")
                    radio_gain = string.split(site["radio"]["gain"][1:-1], ",")
                    radio_address = string.split(site["radio"]["address"][1:-1], ",")
                    recorder_channels = string.split(
                        site["recorder"]["channels"][1:-1], ","
                    )
                    radio_sample_rate = site["radio"]["sample_rate"]

                    cmd_line0 = "%s " % (site["recorder"]["command"])

                    if site["radio"]["type"] == "b210":

                        # just record a fixed frequency, needs a dual radio Thor3 script. This can be done!
                        idx = 0
                        freq = obj_freqs[1]

                        cmd_line1 = '-r %s -d "%s" -s %s -e %s -c %s -f %4.3f ' % (
                            radio_sample_rate,
                            radio_channel[idx],
                            offset_rise_time,
                            offset_set_time,
                            recorder_channels[idx],
                            freq,
                        )

                        log_file_name = "%s_%s_%s_%dMHz.log" % (
                            site_tag,
                            obj_id,
                            offset_rise_time,
                            int(freq / 1.0e6),
                        )
                        cmd_fname = "%s_%s_%s_%dMHz" % (
                            site_tag,
                            obj_id,
                            rise_time,
                            int(freq / 1.0e6),
                        )

                        cmd_line2 = (
                            " -g %s -m %s --devargs num_recv_frames=1024 --devargs master_clock_rate=24.0e6 -o %s/%s"
                            % (
                                radio_gain[idx],
                                radio_address[idx],
                                site["recorder"]["data_path"],
                                cmd_fname,
                            )
                        )
                        cmd_line2 += " {0}".format(
                            site["radio"].get("extra_args", "")
                        ).rstrip()

                        if not opt.foreground:
                            cmd_line0 = "nohup " + cmd_line0
                            cmd_line2 = cmd_line2 + " 2>&1 &"
                        else:
                            cmd_line2 = cmd_line2

                        if opt.debug:
                            print(cmd_line0, cmd_line1, cmd_line2, cmd_fname)

                        cmd_lines.append(
                            (
                                cmd_line0 + cmd_line1 + cmd_line2,
                                cmd_fname,
                                pass_md,
                                obj_info,
                            )
                        )

                        print("\n")

                    elif site["radio"]["type"] == "n200_tvrx2":

                        cmd_line1 = (
                            ' -r %s -d "%s %s" -s %s -e %s -c %s,%s -f %4.3f,%4.3f '
                            % (
                                radio_sample_rate,
                                radio_channel[0],
                                radio_channel[1],
                                offset_rise_time,
                                offset_set_time,
                                recorder_channels[0],
                                recorder_channels[1],
                                obj_freqs[0],
                                obj_freqs[1],
                            )
                        )

                        log_file_name = "%s_%s_%s_combined.log" % (
                            site_tag,
                            obj_id,
                            offset_rise_time,
                        )
                        cmd_fname = "%s_%s_%s_combined" % (site_tag, obj_id, rise_time)

                        cmd_line2 = " -g %s,%s -m %s -o %s/%s" % (
                            radio_gain[0],
                            radio_gain[1],
                            radio_address[0],
                            site["recorder"]["data_path"],
                            cmd_fname,
                        )
                        cmd_line2 += " {0}".format(
                            site["radio"].get("extra_args", "")
                        ).rstrip()

                        if not opt.foreground:
                            cmd_line0 = "nohup " + cmd_line0
                            cmd_line2 = cmd_line2 + " 2>&1 &"
                        else:
                            cmd_line2 = cmd_line2

                        if opt.debug:
                            print(cmd_line0, cmd_line1, cmd_line2, cmd_fname)

                        cmd_lines.append(
                            (
                                cmd_line0 + cmd_line1 + cmd_line2,
                                cmd_fname,
                                pass_md,
                                obj_info,
                            )
                        )

                print("\n")

                if opt.foreground:
                    dtstart0 = dateutil.parser.parse(offset_rise_time)
                    dtstop0 = dateutil.parser.parse(offset_set_time)
                    start0 = int(
                        (
                            dtstart0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
                        ).total_seconds()
                    )
                    stop0 = int(
                        (
                            dtstop0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
                        ).total_seconds()
                    )

                    if opt.verbose:
                        print("# waiting for %s @ %s " % (obj_id, offset_rise_time))

                    while time.time() < start0 - 30:
                        time.sleep(op.interval)
                        if opt.verbose:
                            print("#    %d sec" % (start0 - time.time()))

                    for cmd_tuple in cmd_lines:

                        cmd, cmd_fname, pass_md, info_md = cmd_tuple

                        print("# Executing command %s " % (cmd))

                        # write the digital metadata
                        start_idx = int(start0)
                        mdata_dir = (
                            site["recorder"]["metadata_path"]
                            + "/"
                            + cmd_fname
                            + "/metadata"
                        )

                        # site metadata
                        # note we use directory structure for the dictionary here
                        # eventually we will add this feature to digital metadata

                        for k in site:

                            try:
                                os.makedirs(mdata_dir + "/config/%s" % (k))
                            except:
                                pass

                            md_site_obj = DigitalMetadataWriter(
                                mdata_dir + "/config/%s" % (k), 3600, 60, 1, 1, k
                            )

                            if opt.debug:
                                print(site[k])

                            if opt.verbose:
                                print("# writing metadata config / %s " % (k))

                            md_site_obj.write(start_idx, site[k])

                        # info metadata
                        try:
                            os.makedirs(mdata_dir + "/info")
                        except:
                            pass

                        md_info_obj = DigitalMetadataWriter(
                            mdata_dir + "/info", 3600, 60, 1, 1, "info"
                        )

                        if opt.verbose:
                            print("# writing metadata info")

                        if opt.debug:
                            print(info_md)

                        md_info_obj.write(start_idx, info_md)

                        # pass metadata
                        try:
                            os.makedirs(mdata_dir + "/pass")
                        except:
                            pass

                        md_pass_obj = DigitalMetadataWriter(
                            mdata_dir + "/pass", 3600, 60, 1, 1, "pass"
                        )

                        if opt.verbose:
                            print("# writing metadata pass")

                        if opt.debug:
                            print(pass_md)

                        md_pass_obj.write(start_idx, pass_md)

                        # sys.exit(1)

                        # call the command
                        try:
                            subprocess.call(cmd, shell=True)
                        except Exception as eobj:
                            exp_str = str(ExceptionString(eobj))
                            print("exception: %s." % (exp_str))
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            lines = traceback.format_exception(
                                exc_type, exc_value, exc_traceback
                            )
                            print(lines)

                    print("# wait...")
                    while time.time() < stop0 + 1:
                        time.sleep(op.interval)
                        if opt.verbose:
                            print("# complete in %d sec" % (stop0 - time.time()))

        except Exception as eobj:
            exp_str = str(ExceptionString(eobj))
            print("exception: %s." % (exp_str))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print(lines)
            # sys.exit(1)
            # advance 10 minutes
            ctime = ctime + 60 * op.interval

            continue


def parse_command_line():
    parser = optparse.OptionParser()
    parser.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="prints debug output and additional detail.",
    )
    parser.add_option(
        "-d",
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="run in debug mode and not service context.",
    )
    parser.add_option(
        "-b",
        "--bash",
        action="store_true",
        dest="schedule",
        default=False,
        help="create schedule file for bash shell based command / control.",
    )
    parser.add_option(
        "-m",
        "--mask",
        dest="el_mask",
        type=float,
        default=0.0,
        help="mask all passes below the provided elevation.",
    )
    parser.add_option(
        "-c",
        "--config",
        dest="config",
        default="config/beacons.ini",
        help="Use configuration file <config>.",
    )
    parser.add_option(
        "-f",
        "--foreground",
        action="store_true",
        dest="foreground",
        help="Execute schedule in foreground.",
    )
    parser.add_option(
        "-s",
        "--starttime",
        dest="starttime",
        help="Start time in ISO8601 format, e.g. 2016-01-01T15:24:00Z",
    )
    parser.add_option(
        "-e",
        "--endtime",
        dest="endtime",
        help="End time in ISO8601 format, e.g. 2016-01-01T16:24:00Z",
    )
    parser.add_option(
        "-i",
        "--interval",
        dest="interval",
        type=float,
        default=10.0,
        help="Sampling interval for ephemeris predictions, default is 10 seconds.",
    )
    parser.add_option(
        "-r",
        "--radio",
        dest="site",
        default="config/site.ini",
        help="Radio site configuration file.",
    )

    (options, args) = parser.parse_args()

    return (options, args)


if __name__ == "__main__":
    # parse command line options
    op, args = parse_command_line()

    if op.starttime is None:
        st0 = int(math.ceil(time.time())) + 10
    else:
        dtst0 = dateutil.parser.parse(op.starttime)
        st0 = int(
            (dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
        )

        print("Start time: %s (%d)" % (dtst0.strftime("%a %b %d %H:%M:%S %Y"), st0))

    if op.endtime is None:
        # default to the next 24 hours
        et0 = st0 + 60 * 60 * 24.0
    else:
        dtet0 = dateutil.parser.parse(op.endtime)
        et0 = int(
            (dtet0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
        )

        print("End time: %s (%d)" % (dtet0.strftime("%a %b %d %H:%M:%S %Y"), et0))

    ephemeris_passes(op, st0, et0)
