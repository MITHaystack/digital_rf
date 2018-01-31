% ----------------------------------------------------------------------------
% Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------
% example usage of DigitalMetadataReader.m
% Requires python test_write_digital_metadata.py be run first to create test data
% $Id$
metadataDir = '/tmp/test_metadata';

% init the object
reader = DigitalMetadataReader(metadataDir);

% get the sample bounds
[b0, b1] = reader.get_bounds();

% access all the object attributes
fields = reader.get_fields();
disp('The fields are:');
disp(fields);
disp('The samples per sec numerator, denominator, and float values are:');
disp(reader.get_samples_per_second_numerator());
disp(reader.get_samples_per_second_denominator());
disp(reader.get_samples_per_second());
disp('The subdirectory cadence in seconds is:');
disp(reader.get_subdirectory_cadence_seconds());
disp('The file cadence in seconds is:');
disp(reader.get_file_cadence_seconds());
disp('The file name prefix is:');
disp(reader.get_file_name());

% call the main method read for each field
for i=1:length(fields)
    data_map = reader.read(b0, b0+1, char(fields(i)));

    disp(sprintf('Displaying all data relating to field %s', fields{i}));
    recursive_disp_map(data_map);

end
