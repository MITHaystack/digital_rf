classdef DigitalRFReader
    % class DigitalRFReader allows easy read access to static Digital RF data
    %   See testDigitalRFReader.m for usage, or run <doc DigitalRFReader>
    %
    % $Id$

    properties
        topLevelDirectories % a char array of one or more top level directories
        channel_map % a Map object with key=channel_name, value = drf_channel object

    end

    methods
        function reader = DigitalRFReader(topLevelDirectories)
            % DigitalRFReader is the contructor for this class.
            % Inputs - topLevelDirectories - a char array of one or more
            % top level directories, where a top level directory holds
            % channel directories


            % topLevelDirectories - a char array of one or more top level
            %   directories.
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
                top_level_dir_len = length(strtrim(reader.topLevelDirectories(i,:)));
                result = glob(topGlobPath);
                resultDims = size(result);
                for j = 1:resultDims(1)
                    data = char(result(j));
                    remainder = data(top_level_dir_len + 2:end-1);
                    [pathstr,name,ext] = fileparts(remainder);
                    if dirFlag == 0
                        dirArr = struct('top_level_dir', strtrim(reader.topLevelDirectories(i,:)), ...
                            'channel', pathstr);
                        dirFlag = 1;
                    else
                        newArr = struct('top_level_dir', strtrim(reader.topLevelDirectories(i,:)), ...
                            'channel', pathstr);
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
                samples_per_second_numerator = 0;
                samples_per_second_denominator = 0;
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
                        this_samples_per_second_numerator = h5readatt(propFile, '/', 'sample_rate_numerator');
                        this_samples_per_second_denominator = h5readatt(propFile, '/', 'sample_rate_denominator');
                        this_is_complex = h5readatt(propFile, '/', 'is_complex');
                        this_num_subchannels = h5readatt(propFile, '/', 'num_subchannels');
                        if (subdir_cadence_secs == 0)
                            subdir_cadence_secs = this_subdir_cadence_secs;
                            file_cadence_millisecs = this_file_cadence_millisecs;
                            samples_per_second_numerator = this_samples_per_second_numerator;
                            samples_per_second_denominator = this_samples_per_second_denominator;
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
                            if (samples_per_second_numerator ~= this_samples_per_second_numerator)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched samples_per_second_numerator found');
                                throw(ME)
                            end
                            if (samples_per_second_denominator ~= this_samples_per_second_denominator)
                                ME = MException('DigitalRFReader:metadataError', ...
                                    'mismatched samples_per_second_denominator found');
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
                    file_cadence_millisecs, samples_per_second_numerator, samples_per_second_denominator, is_complex, num_subchannels);
                reader.channel_map(thisChannel) = new_drf_channel;
            end % end loop through unique channel names

        end % end DigitalRFReader constructor



        function channels = get_channels(obj)
            % get_channels returns a cell array of channel names found
            % Inputs: None
            channels = keys(obj.channel_map);
        end % end get_channels



        function [lower_sample, upper_sample] = get_bounds(obj, channel)
            % get_bounds returns the first and last sample in channel.
            % sample bounds are in samples since 0 seconds unix time
            % (that is, unix time * sample_rate)
            drf_chan = obj.channel_map(channel);
            [lower_sample, upper_sample] = drf_chan.get_bounds();
        end


        function [data_map] = read(obj, channel, start_sample, end_sample, subchannel)
            % read returns a containers.Map() object containing key= all
            % first samples of continuous block of data found between
            % start_sample and end_sample (inclusive).  Value is an array
            % of the type stored in /rf_data. If subchannel is 0, all
            % channels returned.  If subchannel == -1, length of continuous
            % data is returned instead of data, Else, only subchannel set
            % by subchannel argument returned.
            drf_chan = obj.channel_map(channel);
            data_map = drf_chan.read(start_sample, end_sample, subchannel);
        end


        function subdir_cadence_secs = get_subdir_cadence_secs(obj, channel)
            % get_subdir_cadence_secs returns subdir_cadence_secs for given channel
            drf_chan = obj.channel_map(channel);
            subdir_cadence_secs = drf_chan.subdir_cadence_secs;
        end %


        function file_cadence_millisecs = get_file_cadence_millisecs(obj, channel)
            % get_file_cadence_millisecs returns file_cadence_millisecs for given channel
            drf_chan = obj.channel_map(channel);
            file_cadence_millisecs = drf_chan.file_cadence_millisecs;
        end %



        function samples_per_second = get_samples_per_second(obj, channel)
            % get_samples_per_second returns samples_per_second for given channel
            drf_chan = obj.channel_map(channel);
            samples_per_second = drf_chan.samples_per_second;
        end % end get_channels



        function is_complex = get_is_complex(obj, channel)
            % get_is_complex returns is_complex (1 or 0) for given channel
            drf_chan = obj.channel_map(channel);
            is_complex = drf_chan.is_complex;
        end % end get_is_complex



        function num_subchannels = get_num_subchannels(obj, channel)
            % get_num_subchannels returns num_subchannels (1 or greater) for given channel
            drf_chan = obj.channel_map(channel);
            num_subchannels = drf_chan.num_subchannels;
        end % end get_num_subchannels



        function vector = read_vector(obj, channel, start_sample, sample_length)
            % read_vector returns a data vector sample_length x num_subchannels.
            % Data type will be complex if data was complex, otherwise data type
            % as stored in same format as in Hdf5 file. Raises error if
            % data gap found.  Simply calls read for all channels, and
            % throws error if more than one block returned.

            end_sample = start_sample + (sample_length - 1);
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
