% DigitalRFReader  Read data in the Digital RF format.
%   See testDigitalRFReader.m for usage, or run <doc DigitalRFReader>
%

% ----------------------------------------------------------------------------
% Copyright (c) 2017, 2019 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------

classdef DigitalRFReader
    properties
        % char array of one or more top level directories
        topLevelDirectories
        % Map object with key = channel_name, value = drf_channel object
        channel_map

    end

    methods
        function reader = DigitalRFReader(topLevelDirectories)
            % DigitalRFReader  Initialize Digital RF reader.
            %   reader = DigitalRFReader(topLevelDirectories)
            %
            %   topLevelDirectories : char array
            %     Path to a top level directory (or multiple directories)
            %     that contains channel directories with Digital RF data.
            %     The top level direcotry is the *parent* directory of
            %     channel directories containing a drf_properties.h5 file.
            %     Pass multiple directories as char('PATH1', 'PATH2', ...).
            %
            if (~(ischar(topLevelDirectories)))
                ME = MException('DigitalRFReader:invalidArg', ...
                  'topLevelDirectories arg not a string or char array');
                throw(ME)
            end
            dims = size(topLevelDirectories);
            if length(dims) == 1
                reader.topLevelDirectories = char(topLevelDirectories);
            else
                reader.topLevelDirectories = topLevelDirectories;
            end

            % make sure all exist
            dims = size(reader.topLevelDirectories);
            for i = 1:dims(1)
                if exist(strtrim(reader.topLevelDirectories(i,:)), 'dir') == 0
                    ME = MException('DigitalRFReader:invalidArg', ...
                        'topLevelDirectory %s not found', ...
                        reader.topLevelDirectories(i,:));
                    throw(ME)
                end
            end

            % the rest of this constructor fills out this map
            reader.channel_map = containers.Map();

            % fill out temp structure dirArr with fields: 1) top_level_dir,
            %   2) channel.
            dirFlag = 0;
            for i = 1:dims(1)
                topGlobPath = fullfile(strtrim(reader.topLevelDirectories(i,:)),'*', 'drf_properties.h5');
                result = glob(topGlobPath);
                resultDims = size(result);
                for j = 1:resultDims(1)
                    data = char(result(j));
                    path_components = strsplit(data, filesep);
                    % last component is drf_properties.h5, second to last
                    % is channel name
                    ch = path_components{end-1};
                    if dirFlag == 0
                        dirArr = struct('top_level_dir', strtrim(reader.topLevelDirectories(i,:)), ...
                            'channel', ch);
                        dirFlag = 1;
                    else
                        newArr = struct('top_level_dir', strtrim(reader.topLevelDirectories(i,:)), ...
                            'channel', ch);
                        dirArr(end+1) = newArr;
                    end
                end

            end

            if (dirFlag == 0)
                ME = MException('DigitalRFReader:invalidArg', ...
                    'no valid channels found in top_level_dir %s', ...
                    reader.topLevelDirectories(i,:));
                throw(ME)
            end


            % now loop through each unique channel name and get all
            % topLevelDirectories and associated metadata
            channels = char(dirArr.channel);
            chanDims = size(dirArr);
            unique_channels = unique(channels, 'rows');
            uniqueDims = size(unique_channels);
            top_levels = char(dirArr.top_level_dir);

            for i = 1:uniqueDims(1)
                thisChannel = strtrim(char(unique_channels(i,:)));
                % properties to read and/or verify consistency
                subdir_cadence_secs = 0;
                file_cadence_millisecs = 0;
                sample_rate_numerator = 0;
                sample_rate_denominator = 0;
                is_complex = 0;
                num_subchannels = 0;
                top_dir_list = {}; % cell array of top level dirs in this channel
                for j = 1:chanDims(2)
                    thisItem = strtrim(char(dirArr(j).channel));
                    if strcmp(thisItem, thisChannel)
                        this_top_level = strtrim(top_levels(j,:));
                        top_dir_list{end+1} = this_top_level;
                        % read properties from drf_properties.h5
                        propFile = fullfile(this_top_level, thisChannel, 'drf_properties.h5');
                        this_subdir_cadence_secs = h5readatt(propFile, '/', 'subdir_cadence_secs');
                        this_file_cadence_millisecs = h5readatt(propFile, '/', 'file_cadence_millisecs');
                        this_sample_rate_numerator = h5readatt(propFile, '/', 'sample_rate_numerator');
                        this_sample_rate_denominator = h5readatt(propFile, '/', 'sample_rate_denominator');
                        this_is_complex = h5readatt(propFile, '/', 'is_complex');
                        this_num_subchannels = h5readatt(propFile, '/', 'num_subchannels');
                        if (subdir_cadence_secs == 0)
                            subdir_cadence_secs = this_subdir_cadence_secs;
                            file_cadence_millisecs = this_file_cadence_millisecs;
                            sample_rate_numerator = this_sample_rate_numerator;
                            sample_rate_denominator = this_sample_rate_denominator;
                            is_complex = this_is_complex;
                            num_subchannels = this_num_subchannels;
                        else
                            if (subdir_cadence_secs ~= this_subdir_cadence_secs)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched subdir_cadence_secs found');
                                throw(ME)
                            end
                            if (file_cadence_millisecs ~= this_file_cadence_millisecs)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched file_cadence_millisecs found');
                                throw(ME)
                            end
                            if (sample_rate_numerator ~= this_sample_rate_numerator)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched sample_rate_numerator found');
                                throw(ME)
                            end
                            if (sample_rate_denominator ~= this_sample_rate_denominator)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched sample_rate_denominator found');
                                throw(ME)
                            end
                            if (is_complex ~= this_is_complex)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched is_complex found');
                                throw(ME)
                            end
                            if (num_subchannels ~= this_num_subchannels)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched num_subchannels found');
                                throw(ME)
                            end
                        end
                    end
                end
                % found all top level dirs for this channel - add it
                new_drf_channel = drf_channel(thisChannel, top_dir_list, subdir_cadence_secs, ...
                    file_cadence_millisecs, sample_rate_numerator, sample_rate_denominator, is_complex, num_subchannels);
                reader.channel_map(thisChannel) = new_drf_channel;
            end % end loop through unique channel names

        end % end DigitalRFReader constructor



        function channels = get_channels(obj)
            % get_channels  Return a cell array of channel names.
            %   channels = get_channels()
            %
            %   channels : cell array
            %     Cell array containing the names of all channel
            %     directories found within the reader's top level
            %     directory.
            %
            channels = keys(obj.channel_map);
        end % end get_channels



        function [start_sample, end_sample] = get_bounds(obj, channel)
            % get_bounds  Return the index of the first and last samples.
            %   [start_sample, end_sample] = get_bounds(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   start_sample : integer
            %     Sample index (number of samples since the Unix epoch,
            %     i.e. unix_time * sample_rate) for the start of the data.
            %
            %   end_sample : integer
            %     Sample index for the end of the data (inclusive).
            %
            drf_chan = obj.channel_map(channel);
            [start_sample, end_sample] = drf_chan.get_bounds();
        end


        function reader = get_digital_metadata(obj, channel)
            % get_digital_metadata  Return a reader for the metadata.
            %   md_reader = get_digital_metadata(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   md_reader : DigitalMetadataReader object
            %     Reader object for the metadata associated with the
            %     data channel (found in the 'metadata' directory within
            %     the channel directory).
            %
            drf_chan = obj.channel_map(channel);
            reader = drf_chan.get_digital_metadata();
        end


        function data_map = read(obj, channel, start_sample, end_sample, subchannel)
            % read  Return a Map object of contiguous data blocks.
            %   read(channel, start_sample, end_sample, subchannel)
            %
            %   channel : string
            %     Name of the channel to read.
            %   start_sample : integer
            %     Sample index for start of read.
            %   end_sample : integer
            %     Sample index for end of read (inclusive).
            %   subchannel : integer
            %     Index of subchannel to read. If 0, read all subchannels.
            %     If -1, return only the data block lengths and not the
            %     data itself.
            %
            %   data_map : containers.Map object
            %     Map of data block start samples to data arrays (or
            %     lengths) for samples within the window of
            %     [start_sample:end_sample]. The keys are a sample index
            %     indicating the start of a contiguous data block, while
            %     the values are the corresponding data array for the
            %     block or (when subchannel is -1) the number of samples
            %     in the block.
            %
            drf_chan = obj.channel_map(channel);
            data_map = drf_chan.read(start_sample, end_sample, subchannel);
        end


        function subdir_cadence_secs = get_subdir_cadence_secs(obj, channel)
            % get_subdir_cadence_secs  Return the subdirectory cadence.
            %   subdir_cadence = get_subdir_cadence_secs(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   subdir_cadence : uint64
            %     Subdirectory cadence in seconds.
            %
            drf_chan = obj.channel_map(channel);
            subdir_cadence_secs = drf_chan.subdir_cadence_secs;
        end


        function file_cadence_millisecs = get_file_cadence_millisecs(obj, channel)
            % get_file_cadence_millisecs  Return the file cadence.
            %   file_cadence = get_file_cadence_millisecs(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   file_cadence : uint64
            %     File cadence in milliseconds.
            %
            drf_chan = obj.channel_map(channel);
            file_cadence_millisecs = drf_chan.file_cadence_millisecs;
        end



        function sample_rate_numerator = get_sample_rate_numerator(obj, channel)
            % get_sample_rate_numerator  Return sample rate numerator.
            %   num = get_sample_rate_numerator(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   num : uint64
            %     Numerator of the sample rate in Hz.
            %
            drf_chan = obj.channel_map(channel);
            sample_rate_numerator = drf_chan.sample_rate_numerator;
        end



        function sample_rate_denominator = get_sample_rate_denominator(obj, channel)
            % get_sample_rate_denominator  Return sample rate denominator.
            %   den = get_sample_rate_denominator(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   den : uint64
            %     Denominator of the sample rate in Hz.
            %
            drf_chan = obj.channel_map(channel);
            sample_rate_denominator = drf_chan.sample_rate_denominator;
        end



        function samples_per_second = get_samples_per_second(obj, channel)
            % get_samples_per_second  Return the samples per second.
            %   sps = get_samples_per_second(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   sps : double
            %     Sample rate in Hz.
            %
            drf_chan = obj.channel_map(channel);
            samples_per_second = drf_chan.samples_per_second;
        end



        function is_complex = get_is_complex(obj, channel)
            % get_is_complex  Return whether the channel is complex.
            %   is_complex = get_is_complex(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   is_complex : 0 or 1
            %     1 if complex, 0 if not
            %
            drf_chan = obj.channel_map(channel);
            is_complex = drf_chan.is_complex;
        end



        function num_subchannels = get_num_subchannels(obj, channel)
            % get_num_subchannels  Return the number of subchannels.
            %   nsc = get_num_subchannels(channel)
            %
            %   channel : string
            %     Name of the channel to query.
            %
            %   nsc : integer
            %     Number of subchannels.
            %
            drf_chan = obj.channel_map(channel);
            num_subchannels = drf_chan.num_subchannels;
        end



        function vector = read_vector(obj, channel, start_sample, vector_length)
            % read_vector  Read a contiguous vector of data.
            %   data = read_vector(channel, start_sample, vector_length)
            %
            %   channel : string
            %     Name of the channel to query.
            %   start_sample : integer
            %     Sample index for start of read.
            %   vector_length : integer
            %     Number of samples to return.
            %
            %   data : array of size [sample_length, num_subchannels]
            %     Data vector.
            %
            %   The returned data type will be the same as stored in
            %   the HDF5 file.
            %
            %   An error is raised if a data gap is found. (This just
            %   calls the read method for all channels and throws an
            %   error if more than one block is returned.)
            %

            end_sample = start_sample + (vector_length - 1);
            data_map = obj.read(channel, start_sample, end_sample, 0);
            if (isempty(data_map.keys()))
                ME = MException('DigitalRFReader:invalidArg', ...
                  'no data found between %i and %i', ...
                    start_sample, end_sample);
                throw(ME)
            elseif (length(data_map.keys()) > 1)
                 ME = MException('DigitalRFReader:invalidArg', ...
                  'data gap found between %i and %i', ...
                    start_sample, end_sample);
                throw(ME)
            end
            keys = data_map.keys();
            vector = data_map(keys{1});
        end


    end % end methods

end % end DigitalRFReader class
