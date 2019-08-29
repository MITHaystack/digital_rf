% ----------------------------------------------------------------------------
% Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------
classdef drf_channel
    % class drf_channel is a private class to describe a single
    % DigitalRFReader channel - do not create directly
    %
    % $Id$

    properties
        channel_name % channel name (string)
        top_level_dirs % char array of one or more top level dirs
        subdir_cadence_secs % seconds per subdirectory (int)
        file_cadence_millisecs % number of millseconds per file (int)
        sample_rate_numerator % sample rate numerator in Hz in this drf_channel (int)
        sample_rate_denominator % sample rate denomerator in Hz in this drf_channel (int)
        samples_per_second % sample rate in Hz in this drf_channel (numerator/denominator) (double)
        is_complex % 1 if channel has real and imag data, 0 if real only
        num_subchannels % number of subchannels - 1 or greater
        sub_directory_glob % glob string for subdirectories
        rf_file_glob % glob string for rf files
    end

    methods
        function channel = drf_channel(channel_name, top_level_dirs, subdir_cadence_secs, file_cadence_millisecs, ...
                sample_rate_numerator, sample_rate_denominator, is_complex, num_subchannels)
            % drf_channel constructor
            % Inputs:
            %   channel_name - channel name
            %   top_level_dirs - char array of one or more top level dirs
            %   subdir_cadence_secs - seconds per subdirectory (int)
            %   file_cadence_millisecs - number of millseconds per file (int)
            %   sample_rate_numerator - sample rate numerator in Hz in this drf_channel (int)
            %   sample_rate_denominator - sample rate denominator in Hz in this drf_channel (int)
            %   is_complex - 1 if channel has real and imag data, 0 if real only
            %   num_subchannels - number of subchannels - 1 or greater


            % constants
            % define glob string for sub_directories in form YYYY-MM-DDTHH-MM-SS
            channel.sub_directory_glob = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]';
            channel.rf_file_glob = 'rf@[0-9]*.[0-9][0-9][0-9].h5';

            channel.channel_name = channel_name;
            channel.top_level_dirs = top_level_dirs;
            channel.subdir_cadence_secs = uint64(subdir_cadence_secs);
            channel.file_cadence_millisecs = uint64(file_cadence_millisecs);
            channel.sample_rate_numerator = uint64(sample_rate_numerator);
            channel.sample_rate_denominator = uint64(sample_rate_denominator);
            channel.samples_per_second = double(sample_rate_numerator)/double(sample_rate_denominator);
            channel.is_complex = is_complex;
            channel.num_subchannels = num_subchannels;
        end


        function [lower_sample, upper_sample] = get_bounds(obj)
            % get bounds returns upper and lower bounds in channel in
            % samples
            lower_sample = 0;
            upper_sample = 0;
            for i = 1:length(obj.top_level_dirs)
                this_glob = char(fullfile(obj.top_level_dirs(i), obj.channel_name, obj.sub_directory_glob));
                subdirs = glob(this_glob);
                subdirs = sort(subdirs);
                if (isempty(subdirs))
                    continue
                end

                % first find first sample
                first_sub = subdirs(1);
                this_glob = char(fullfile(first_sub, obj.rf_file_glob));
                rf_files = glob(this_glob);
                rf_files = sort(rf_files);
                if (isempty(rf_files))
                    continue
                end
                first_file = char(rf_files(1));
                index_data = h5read(first_file, '/rf_data_index');
                start_sample = index_data(1,1);
                if (lower_sample == 0)
                    lower_sample = start_sample;
                elseif (lower_sample > start_sample)
                    lower_sample = start_sample;
                end

                % next find last sample
                last_sub = subdirs(end);
                this_glob = char(fullfile(last_sub, obj.rf_file_glob));
                rf_files = glob(this_glob);
                rf_files = sort(rf_files);
                if (isempty(rf_files))
                    continue
                end
                last_file = char(rf_files(end));
                index_data = h5read(last_file, '/rf_data_index');
                rf_data = h5read(last_file, '/rf_data');
                if (obj.is_complex)
                    this_size = size(rf_data.r);
                else
                    this_size = size(rf_data);
                end
                data_len = this_size(2);
                last_index = index_data(2, end);
                last_sample = index_data(1, end) + ((data_len - last_index) - 1);
                if (upper_sample == 0)
                    upper_sample = last_sample;
                elseif (upper_sample < last_sample)
                    upper_sample = last_sample;
                end

            end

        end

        function [reader] = get_digital_metadata(obj)
            % get Digital Metadata reader for the channel
            for i = 1:length(obj.top_level_dirs)
                metadata_dir = fullfile(obj.top_level_dirs{i}, obj.channel_name, 'metadata');
                if (exist(metadata_dir, 'dir'))
                    reader = DigitalMetadataReader(metadata_dir);
                    return
                else
                    error('Associated metadata directory does not exist!')
                end
            end
        end

        function [data_map] = read(obj, start_sample, end_sample, subchannel)
            % read returns a containers.Map() object containing key= all
            % first samples of continuous block of data found between
            % start_sample and end_sample (inclusive).  Value is an array
            % of the type sorted in /rf_data. If subchannel is 0, all
            % channels returned.  If subchannel == -1, only returns length
            % of continuous data. Else, only subchannel set by subchannel
            % argument returned.
            first_data_map = containers.Map('KeyType','uint64','ValueType','any');
            file_list = obj.get_file_list(start_sample, end_sample);
            for i = 1:length(obj.top_level_dirs)
                for j = 1:length(file_list)
                   datafile = char(fullfile(obj.top_level_dirs(i), obj.channel_name, file_list(j)));
                   if (exist(datafile, 'file'))
                        data = h5read(datafile, '/rf_data');
                        if (obj.is_complex)
                            this_vector = complex(data.r, data.i);
                        else
                            this_vector = data;
                        end
                        this_vector = transpose(this_vector);
                        vector_size = size(this_vector);
                        if (subchannel > vector_size(2))
                            ME = MException('DigitalRFReader:invalidArg', ...
                                'subchannel %i not found', ...
                                subchannel);
                            throw(ME)
                        end
                        if (subchannel > 0)
                            this_vector = this_vector(:,subchannel);
                        end
                        data_len = size(this_vector);
                        data_len = data_len(1);
                        index_data = h5read(datafile, '/rf_data_index');
                        index_data_size = size(index_data);
                        % loop through each row in index_data
                        for k = 1:index_data_size(2)
                            this_sample = index_data(1,k);
                            this_index = index_data(2,k);
                            if (k == index_data_size(2))
                                last_index = data_len - 1;
                            else
                                last_index = index_data(2, k+1) - 1;
                            end
                            last_sample = this_sample + (last_index - this_index);
                            if (start_sample <= this_sample)
                                read_start_index = this_index;
                                read_start_sample = this_sample;
                            elseif (start_sample <= last_sample)
                                read_start_index = this_index + (start_sample - this_sample);
                                read_start_sample = this_sample + (start_sample - this_sample);
                            else
                                % no data in this block to read
                                continue
                            end

                            if (end_sample >= last_sample)
                                read_end_index = last_index;
                            else
                                read_end_index = last_index - (last_sample - end_sample);
                            end

                            % skip if no data found
                            if (read_start_index > read_end_index)
                                continue
                            end

                            % add this block of data - 1 added because
                            % logic was based on python indexing
                            if (subchannel == -1)
                                first_data_map(read_start_sample) = (read_end_index+1) - read_start_index;
                            else
                                first_data_map(read_start_sample) = this_vector(read_start_index+1:read_end_index+1,:);
                            end
                        end
                   end
                end
                % temp only
                data_map = obj.combine_blocks(first_data_map, subchannel);
            end
        end



        function [new_data_map] = combine_blocks(obj, data_map, combine_flag)
            % combine_blocks takes as a input data_map which is a
            % containers.Map with key = start_sample, value = array being
            % returned. Returned new_data_map is the same, except that all
            % possible continuous blocks have been stiched.  If
            % combine_flag is -1, then value is simply the length of the
            % array rather than the array itself.
            new_data_map = containers.Map('KeyType','uint64','ValueType','any');
            keys = data_map.keys();
            keys = cell2mat(keys);
            keys = sort(keys);
            last_start_sample = 0;
            last_end_sample = 0;
            last_data = [];
            for i=1:length(keys)
                key = keys(i);
                data = data_map(key);
                if (combine_flag ~= -1)
                    data_len = size(data);
                    data_len = data_len(1);
                else
                    data_len = data;
                end
                if (isempty(last_data))
                    % new block
                    last_start_sample = key;
                    last_end_sample = key + (data_len-1);
                    last_data = data;
                else
                    % see if this block is continuous
                    if (key - 1 == last_end_sample)
                        % continuous data - append vector
                        if (combine_flag ~= -1)
                            last_data = cat(1, last_data, data);
                        else
                            last_data = last_data + data;
                        end
                        last_end_sample = last_end_sample + data_len;
                    elseif (key - 1 < last_end_sample)
                        % overlapping data found - abort!!!
                        ME = MException('DigitalRFReader:badData', ...
                                'overlapping data found at sample %i', ...
                                last_end_sample);
                        throw(ME)
                    else
                        % non-continuous data found - append last data
                        new_data_map(last_start_sample) = last_data;
                        last_start_sample = key;
                        last_end_sample = key + (data_len-1);
                        last_data = data;
                    end
                end
            end

            % append last block
            new_data_map(last_start_sample) = last_data;
        end




        function [file_list] = get_file_list(obj, sample0, sample1)
            % _get_file_list returns an ordered list of full file names, starting at subdir,
            % of data files that could contain data.
            % Inputs:
            %   sample0 - the first sample to read
            %   sample1 - the last sample (inclusive) to read

            file_list = {};

            sps_n = obj.sample_rate_numerator;
            sps_d = obj.sample_rate_denominator;
            sample0 = uint64(sample0);
            sample1 = uint64(sample1);
            % get the start and end time in seconds to get the subdirectory
            start_ts = idivide(sample0, sps_n)*sps_d + idivide(mod(sample0, sps_n)*sps_d, sps_n);
            end_ts = idivide(sample1, sps_n)*sps_d + idivide(mod(sample1, sps_n)*sps_d, sps_n) + 1;
            % Get the start time and end time in microsecs to get file
            start_msts = idivide(sample0, sps_n)*sps_d*1000 + idivide(mod(sample0, sps_n)*sps_d*1000, sps_n);
            end_msts = idivide(sample1, sps_n)*sps_d*1000 + idivide(mod(sample1, sps_n)*sps_d*1000, sps_n);

            % get subdirectory start and end ts
            start_sub_ts = idivide(start_ts, obj.subdir_cadence_secs)*obj.subdir_cadence_secs;
            end_sub_ts = idivide(end_ts, obj.subdir_cadence_secs)*obj.subdir_cadence_secs;

            sub_ts_arr = start_sub_ts:obj.subdir_cadence_secs:end_sub_ts;
            for i = 1:length(sub_ts_arr)
                sub_ts = sub_ts_arr(i);
                sub_datetime = datetime(sub_ts, 'ConvertFrom', 'posixtime');
                subdir = datestr(sub_datetime, 'YYYY-mm-ddTHH-MM-SS');
                start_point = sub_ts*1000;
                end_point = (sub_ts + obj.subdir_cadence_secs)*1000 - 1;
                file_msts_in_subdir = start_point:obj.file_cadence_millisecs:end_point;
                % file has valid samples if last time in file is after start time
                % and first time in file is before end time
                valid_file_logic = (file_msts_in_subdir + obj.file_cadence_millisecs - 1 >= start_msts) ...
                    & (file_msts_in_subdir <= end_msts);
                valid_file_msts = file_msts_in_subdir(valid_file_logic);
                for j = 1:length(valid_file_msts)
                    file_basename = sprintf('rf@%i.%03i.h5', idivide(valid_file_msts(j), uint64(1000)), mod(valid_file_msts(j), 1000));
                    full_filename = fullfile(subdir, file_basename);
                    file_list(end+1) = cellstr(full_filename);
                end
            end

        end



    end % end methods
end % end class
