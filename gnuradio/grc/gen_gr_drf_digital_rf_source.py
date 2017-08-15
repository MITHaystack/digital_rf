# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
MAIN_TMPL = """\
<?xml version="1.0"?>
<!--
  Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
  All rights reserved.

  Distributed under the terms of the BSD 3-clause license.

  The full license is in the LICENSE file, distributed with this software.
-->
<block>
    <name>Digital RF Source</name>
    <key>gr_drf_digital_rf_source</key>
    <category>Digital RF</category>
    <flags>\#if \$throttle() then 'throttle' else ''\#</flags>
    <import>import gr_drf</import>
    <make>\\
    \#set \$allchannels = #echo '[{0}]'.format(', '.join(['$channel{0}()'.format(n) for n in range($max_num_channels)]))
    \#set \$channels = \$allchannels[:\$nchan()]
    \#set \$allstart = #echo '[{0}]'.format(', '.join(['$start{0}()'.format(n) for n in range($max_num_channels)]))
    \#set \$start = \$allstart[:\$nchan()]
    \#set \$allend = #echo '[{0}]'.format(', '.join(['$end{0}()'.format(n) for n in range($max_num_channels)]))
    \#set \$end = \$allend[:\$nchan()]
    gr_drf.digital_rf_source(
        \$top_level_dir,
        channels=\$channels,
        start=\$start,
        end=\$end,
        repeat=\$repeat,
        throttle=\$throttle,
        min_chunksize=None if \$min_chunksize==0 else \$min_chunksize,
    )
    </make>

    <callback>set_repeat(\$repeat)</callback>

    <param>
        <name>Directory</name>
        <key>top_level_dir</key>
        <type>string</type>
    </param>
    <param>
        <name>Repeat</name>
        <key>repeat</key>
        <value>False</value>
        <type>bool</type>
        <hide>\#if \$repeat() then 'none' else 'part'\#</hide>
        <option>
            <name>Yes</name>
            <key>True</key>
        </option>
        <option>
            <name>No</name>
            <key>False</key>
        </option>
    </param>
    <param>
        <name>Throttle</name>
        <key>throttle</key>
        <value>False</value>
        <type>bool</type>
        <hide>\#if \$throttle() then 'none' else 'part'\#</hide>
        <option>
            <name>Yes</name>
            <key>True</key>
        </option>
        <option>
            <name>No</name>
            <key>False</key>
        </option>
    </param>
    <param>
        <name>Number of channels</name>
        <key>nchan</key>
        <value>1</value>
        <type>int</type>
        <hide>part</hide>
        #for $n in range(1, $max_num_channels+1)
        <option>
            <name>$n</name>
            <key>$n</key>
        </option>
        #end for
    </param>
    $channel_params
    <param>
        <name>Minimum Chunk Size</name>
        <key>min_chunksize</key>
        <value>0</value>
        <type>int</type>
        <hide>part</hide>
        <tab>Advanced</tab>
    </param>
    <param>
        <name>Show Message Port</name>
        <key>hide_msg_port</key>
        <value>True</value>
        <type>enum</type>
        <hide>part</hide>
        <option>
            <name>Yes</name>
            <key>False</key>
        </option>
        <option>
            <name>No</name>
            <key>True</key>
        </option>
        <tab>Advanced</tab>
    </param>

    <check>\$nchan > 0</check>
    <check>\$min_chunksize >= 0</check>

    <source>
        <name>out</name>
        <type></type>
        <vlen></vlen>
        <nports>\$nchan</nports>
    </source>
    <source>
        <name>properties</name>
        <type>message</type>
        <optional>True</optional>
        <hide>\$hide_msg_port</hide>
    </source>

    <doc>
Read data in Digital RF format.

In addition to outputting samples from Digital RF format data, this
block also emits a 'properties' message containing inherent channel
properties and adds stream tags using the channel's accompanying
Digital Metadata. See the Notes section for details on what the
messages and stream tags contain.


Parameters
---------------

Directory : string
    A top-level directory containing Digital RF channel directories.

Repeat : bool
    If True, loop the data continuously from the start after the end
    is reached. If False, stop after the data is read once.

Throttle : bool
    If True, playback the samples at their recorded sample rate.
    If False, read samples as quickly as possible.

Number of channels : int
    Number of channels to output.

ChN : string
    Identifier for channel number N. This can be the channel name or
    an integer giving the index from the available channels sorted
    alphabetically. An empty string uses the next available channel
    alphabetically.

ChN start : string
    A value giving the start of the channel's playback.
    If None or '', the start of the channel's available data is used.
    If an integer, it is interpreted as a sample index giving the
    number of samples since the epoch (t_since_epoch*sample_rate).
    If a float, it is interpreted as a timestamp (seconds since epoch).
    If a string, three forms are permitted:
        1) a string which can be evaluated to an integer/float and
            interpreted as above,
        2) a string beginning with '+' and followed by an integer
            (float) expression, interpreted as samples (seconds) from
            the start of the data, and
        3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

ChN end : string
    A value giving the end of the channel's playback.
    If None or '', the end of the channel's available data is used.
    Otherwise, this is interpreted in the same way as the start value.


Other Parameters
-----------------------

Minimum Chunk Size : int, optional
    Minimum number of samples to output at once. This value can be
    used to adjust the source's performance to reduce underruns and
    processing time. If 0, a sensible default will be used.


Notes
-----

A top-level directory must contain files in the format:
    [channel]/[YYYY-MM-DDTHH-MM-SS]/rf@[seconds].[%03i milliseconds].h5

If more than one top level directory contains the same channel_name
subdirectory, this is considered the same channel. An error is raised
if their sample rates differ, or if their time periods overlap.

Upon start, this block sends 'properties' messages on its output
message port that contains a dictionaries with one key, the channel's
name, and a value which is a dictionary of properties found in the
channel's 'drf_properties.h5' file.

This block emits the following stream tags at the appropriate sample
for each of the channel's accompanying Digital Metadata samples:

    rx_time : (int secs, float frac) tuple
        Time since epoch of the sample.

    rx_rate : float
        Sample rate in Hz.

    rx_freq : float | 1-D array of floats
        Center frequency or frequencies of the subchannels based on
        the 'center_frequencies' metadata field.

    metadata : dict
        Any additional Digital Metadata fields are added to this
        dictionary tag of metadata.

    </doc>
</block>
"""


CHANNEL_PARAMS_TMPL = """\
    <param>
        <name>Ch${n}</name>
        <key>channel${n}</key>
        <value></value>
        <type>string</type>
        <hide>\#if \$nchan() > $n then 'none' else 'all'#</hide>
    </param>
    <param>
        <name>Ch${n}: start</name>
        <key>start${n}</key>
        <value></value>
        <type>string</type>
        <hide>\#if \$nchan() > $n then 'part' else 'all'#</hide>
    </param>
    <param>
        <name>Ch${n}: end</name>
        <key>end${n}</key>
        <value></value>
        <type>string</type>
        <hide>\#if \$nchan() > $n then 'part' else 'all'#</hide>
    </param>
"""


def parse_tmpl(_tmpl, **kwargs):
    from Cheetah import Template
    return str(Template.Template(_tmpl, kwargs))


max_num_channels = 10


if __name__ == '__main__':
    import sys
    for fname in sys.argv[1:]:
        channel_params = ''.join(
            [parse_tmpl(CHANNEL_PARAMS_TMPL, n=n)
                for n in range(max_num_channels)]
        )
        open(fname, 'w').write(
            parse_tmpl(
                MAIN_TMPL,
                channel_params=channel_params,
                max_num_channels=max_num_channels,
            )
        )
