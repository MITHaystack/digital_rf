% ----------------------------------------------------------------------------
% Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------
classdef DigitalMetadataReader
    % class DigitalMetadataReader allows easy read access to  Digital
    % metadata
    %   See testDigitalMetadataReader.m for usage, or run <doc DigitalMetadataReader>
    %
    % $Id$

    properties
        metadataDir % a string of metadata directory
        subdir_cadence_secs % a number of seconds per directory
        file_cadence_secs % number of seconds per filereader
        sample_rate_numerator % samples per second numerator of metadata
        sample_rate_denominator % samples per second denominator of metadata
        samples_per_second % float samples per second of metadata as determined by numerator and denominator
        file_name % file naming prefix
        fields % a char array of field names in metadata
        dir_glob % string to glob for directories
    end % end properties

    methods
        function reader = DigitalMetadataReader(metadataDir)
            % DigitalMetadataReader is the contructor for this class.
            % Inputs - metadataDir - a string of the path to the metadata

            reader.metadataDir = metadataDir;
            % read properties from drf_properties.h5
            propFile = fullfile(metadataDir, 'dmd_properties.h5');
            reader.subdir_cadence_secs = uint64(h5readatt(propFile, '/', 'subdir_cadence_secs'));
            reader.file_cadence_secs = uint64(h5readatt(propFile, '/', 'file_cadence_secs'));
            reader.sample_rate_numerator = uint64(h5readatt(propFile, '/', 'sample_rate_numerator'));
            reader.sample_rate_denominator = uint64(h5readatt(propFile, '/', 'sample_rate_denominator'));
            reader.samples_per_second = double(reader.sample_rate_numerator) / double(reader.sample_rate_denominator);
            reader.file_name = h5readatt(propFile, '/', 'file_name');
            fields = h5read(propFile, '/fields');
            reader.fields = cellstr(fields.column');
            reader.dir_glob = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]';

        end % end DigitalMetadataReader

        function [lower_sample, upper_sample] = get_bounds(obj)
            % get_bounds returns a tuple of first sample, last sample for this metadata. A sample
            % is the unix time times the sample rate as a integer.
            glob_str = fullfile(obj.metadataDir, obj.dir_glob);
            result = glob(glob_str);

            % get first sample
            glob_str = fullfile(result(1), sprintf('%s@*.h5', char(obj.file_name)));
            result2 = glob(char(glob_str));
            h5_summary = h5info(char(result2(1)));
            name = h5_summary.Groups(1).Name;
            lower_sample = uint64(str2double(name(2:end)));

            % get last sample
            glob_str = fullfile(result(end), sprintf('%s@*.h5', char(obj.file_name)));
            result2 = glob(char(glob_str));
            h5_summary = h5info(char(result2(end)));
            name = h5_summary.Groups(end).Name;
            upper_sample = uint64(str2double(name(2:end)));

        end % end get_bounds

        function fields = get_fields(obj)
            fields = obj.fields;
        end % end get_fields

        function fields = get_sample_rate_numerator(obj)
            fields = obj.sample_rate_numerator;
        end % end get_sample_rate_numerator

        function fields = get_sample_rate_denominator(obj)
            fields = obj.sample_rate_denominator;
        end % end get_sample_rate_denominator

        function fields = get_samples_per_second(obj)
            fields = obj.samples_per_second;
        end % end get_samples_per_second

        function fields = get_subdir_cadence_secs(obj)
            fields = obj.subdir_cadence_secs;
        end % end get_subdir_cadence_secs

        function fields = get_file_cadence_secs(obj)
            fields = obj.file_cadence_secs;
        end % end get_file_cadence_secs

        function fields = get_file_name(obj)
            fields = obj.file_name;
        end % end get_file_name

        function data_map = read(obj, sample0, sample1, field)
            % read returns a containers.Map() object containing key=sample as uint64,
            % value = data at that sample for field, or another containers.Map()
            % with its keys = names,values = data or more containers.Maps -
            % no limit to levels
            %
            %   Inputs:
            %       sample0 - first sample for which to return metadata
            %       sample1 - last sample for which to return metadata. A sample
            %           is the unix time times the sample rate as a long.
            %       field - the valid field you which to get
            data_map = containers.Map('KeyType','uint64','ValueType','any');
            sample0 = uint64(sample0);
            sample1 = uint64(sample1);
            file_list = obj.get_file_list(sample0, sample1);
            for i=1:length(file_list)
                obj.add_metadata(data_map, file_list{i}, sample0, sample1, field);
            end % end for file_list
        end % end read


        function file_list = get_file_list(obj, sample0, sample1)
            % get_file_list is a private method that returns a cell array
            % of strings representing the full path to files that exist
            % with data
            %   Inputs:
            %       sample0 - first sample for which to return metadata
            %       sample1 - last sample for which to return metadata. A sample
            %           is the unix time times the sample rate as a long.
            sps_n = obj.sample_rate_numerator;
            sps_d = obj.sample_rate_denominator;
            sample0 = uint64(sample0);
            sample1 = uint64(sample1);
            % get the start and end time in seconds
            start_ts = idivide(sample0, sps_n)*sps_d + idivide(mod(sample0, sps_n)*sps_d, sps_n);
            end_ts = idivide(sample1, sps_n)*sps_d + idivide(mod(sample1, sps_n)*sps_d, sps_n) + 1;

            % convert ts to be divisible by obj.file_cadence_secs
            start_ts = idivide(start_ts, obj.file_cadence_secs) * obj.file_cadence_secs;
            end_ts = idivide(end_ts, obj.file_cadence_secs) * obj.file_cadence_secs;

            % get subdirectory start and end ts
            start_sub_ts = idivide(start_ts, obj.subdir_cadence_secs) * obj.subdir_cadence_secs;
            end_sub_ts = idivide(end_ts, obj.subdir_cadence_secs) * obj.subdir_cadence_secs;

            sub_ts_arr = start_sub_ts:obj.subdir_cadence_secs:end_sub_ts;

            file_list = {};

            for i=1:length(sub_ts_arr)
                sub_ts = sub_ts_arr(i);
                sub_datetime = datetime(sub_ts, 'ConvertFrom', 'posixtime' );
                subdir = fullfile(obj.metadataDir, datestr(sub_datetime, 'yyyy-mm-ddTHH-MM-SS'));
                file_ts_in_subdir = sub_ts:obj.file_cadence_secs:(sub_ts + obj.subdir_cadence_secs - 1);
                % file has valid samples if last time in file is after start time
                % and first time in file is before end time
                valid_file_logic = (file_ts_in_subdir + obj.file_cadence_secs - 1 >= start_ts) ...
                    & (file_ts_in_subdir <= end_ts);
                valid_file_ts = file_ts_in_subdir(valid_file_logic);
                for j=1:length(valid_file_ts)
                    basename = sprintf('%s@%i.h5', char(obj.file_name), valid_file_ts(j));
                    full_file = fullfile(subdir, basename);
                    if exist(full_file, 'file') == 2
                        file_list{1+length(file_list)} = full_file;
                    end % end if exist
                end % end for valid_file_ts_list
            end % end for sub_arr

        end % end get_file_list


        function add_metadata(obj, data_map, filename, sample0, sample1, field)
            % add metadata adds all needed metadata from filename to
            % data_map
            %   Inputs:
            %       data_map - a containers.Map() object containing key=sample as uint64,
            %           value = data at that sample for all fields
            %       filename - full path of file to read
            %       sample0 - first sample for which to return metadata
            %       sample1 - last sample for which to return metadata. A sample
            %           is the unix time times the sample rate as a long.
            %       field - the valid field you which to get
            h5_summary = h5info(filename);
            keys = h5_summary.Groups;
            for i=1:length(keys)
                sample = uint64(str2double(keys(i).Name(2:end)));
                if (sample >= sample0 && sample <= sample1)
                    path = fullfile(keys(i).Name, field);
                    map = containers.Map('KeyType','char','ValueType','any');
                    data_map(sample) = map;
                    obj.recursive_get_metadata(filename, path, map, field);
                end
            end % end for keys
        end % end add_metadata


        function recursive_get_metadata(obj, filename, path, map, key)
            % recursive_get_metadata is a recursive function that adds
            % either datasets or containers.Map objects to map(key).
            %   Inputs:
            %      filename - full path to Hdf5 digital metadata file
            %      path - path into this level of file
            %      map - containers.Map being added to
            %      key - ket to add dataset to
            h5_subdata = h5info(filename, path);
            if (isfield(h5_subdata, 'Groups'))
                for i=1:length(h5_subdata.Datasets)
                    name = h5_subdata.Datasets(i).Name;
                    newPath = strcat(path, '/', name);
                    map(name) = h5read(filename, newPath);
                end % end Datasets
                for i=1:length(h5_subdata.Groups)
                    % create new map
                    data_map = containers.Map('KeyType','char','ValueType','any');
                    newPath = h5_subdata.Groups(i).Name;
                    map(name) = data_map;
                    recursive_get_metadata(obj, filename, newPath, data_map, name);

                end % end Groups

            else
                map(key)=h5read(filename, path);
            end % end group test

        end % end recursive_get_metadata

    end % end methods


end % end DigitalMetadataReader class
