classdef DigitalMetadataReader
    % class DigitalMetadataReader allows easy read access to  Digital
    % metadata
    %   See testDigitalMetadataReader.m for usage, or run <doc DigitalMetadataReader>
    %
    % $Id$

    properties
        metadataDir % a string of metadata directory
        subdirectory_cadence_seconds % a number of seconds per directory
        file_cadence_seconds % number of seconds per filereader
        samples_per_second_numerator % samples per second numerator of metadata
        samples_per_second_denominator % samples per second denominator of metadata
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
            reader.subdirectory_cadence_seconds = uint64(h5readatt(propFile, '/', 'subdirectory_cadence_seconds'));
            reader.file_cadence_seconds = uint64(h5readatt(propFile, '/', 'file_cadence_seconds'));
            reader.samples_per_second_numerator = uint64(h5readatt(propFile, '/', 'samples_per_second_numerator'));
            reader.samples_per_second_denominator = uint64(h5readatt(propFile, '/', 'samples_per_second_denominator'));
            reader.samples_per_second = double(reader.samples_per_second_numerator) / double(reader.samples_per_second_denominator);
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

        function fields = get_samples_per_second_numerator(obj)
            fields = obj.samples_per_second_numerator;
        end % end get_samples_per_second_numerator

        function fields = get_samples_per_second_denominator(obj)
            fields = obj.samples_per_second_denominator;
        end % end get_samples_per_second_denominator

        function fields = get_samples_per_second(obj)
            fields = obj.samples_per_second;
        end % end get_samples_per_second

        function fields = get_subdirectory_cadence_seconds(obj)
            fields = obj.subdirectory_cadence_seconds;
        end % end get_subdirectory_cadence_seconds

        function fields = get_file_cadence_seconds(obj)
            fields = obj.file_cadence_seconds;
        end % end get_file_cadence_seconds

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
            start_ts = uint64(sample0/obj.samples_per_second);
            end_ts = uint64(sample1/obj.samples_per_second);

            % convert ts to be divisible by obj.file_cadence_seconds
            start_ts = (start_ts ./ obj.file_cadence_seconds) * obj.file_cadence_seconds;
            end_ts = (end_ts ./ obj.file_cadence_seconds) * obj.file_cadence_seconds;

            % get subdirectory start and end ts
            start_sub_ts = (start_ts ./ obj.subdirectory_cadence_seconds) * obj.subdirectory_cadence_seconds;
            end_sub_ts = (end_ts ./ obj.subdirectory_cadence_seconds) * obj.subdirectory_cadence_seconds;

            num_sub = uint64(1 + ((end_sub_ts - start_sub_ts) ./ obj.subdirectory_cadence_seconds));

            sub_arr = linspace(double(start_sub_ts), double(end_sub_ts), double(num_sub));

            file_list = {};

            for i=1:length(sub_arr)
                sub_ts = uint64(sub_arr(i));
                sub_datetime = datetime( sub_ts, 'ConvertFrom', 'posixtime' );
                subdir = fullfile(obj.metadataDir, datestr(sub_datetime, 'yyyy-mm-ddTHH-MM-SS'));
                num_file_ts = uint64(1 + (obj.subdirectory_cadence_seconds - obj.file_cadence_seconds) ./ obj.file_cadence_seconds);
                file_ts_in_subdir = linspace(double(sub_ts), ...
                    double(sub_ts + (obj.subdirectory_cadence_seconds - obj.file_cadence_seconds)), double(num_file_ts));
                file_ts_in_subdir = uint64(file_ts_in_subdir);
                ind = find(file_ts_in_subdir >= start_ts & file_ts_in_subdir <= end_ts);
                valid_file_ts_list = file_ts_in_subdir(ind);
                for j=1:length(valid_file_ts_list)
                    basename = sprintf('%s@%i.h5', char(obj.file_name), valid_file_ts_list(j));
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
