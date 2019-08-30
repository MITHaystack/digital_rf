% test and example usage of DigitalMetadataReader.m

% ----------------------------------------------------------------------------
% Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------

example_dir = 'data/example';
metadata_dir = [example_dir, '/ch0/metadata'];

% get the metadata reader through the DigitalRFReader
reader = DigitalRFReader(example_dir);
chs = reader.get_channels();
ch = chs{1};
md_reader = reader.get_digital_metadata(ch);

% get the metadata reader directly
md_reader = DigitalMetadataReader(metadata_dir);
fprintf('Reading metadata from directory: %s\n', metadata_dir)
disp(md_reader)

[start_sample, end_sample] = md_reader.get_bounds();
disp('These are the sample bounds:');
disp([start_sample, end_sample]);

disp('The fields are:');
disp(md_reader.fields);

% call the main method read for each field
for i=1:length(md_reader.fields)
    data_map = md_reader.read(start_sample, end_sample, md_reader.fields{i});

    fprintf('Displaying all data relating to field %s\n', md_reader.fields{i});
    recursive_disp_map(data_map);

end

% get one field specifically
field = 'description';
data_map = md_reader.read(start_sample, end_sample, field);
data_sample_indices = data_map.keys();
sample_index = data_sample_indices{1};
value = data_map(sample_index);
fprintf('Reading value of %s at index %d:\n', field, sample_index)
desc = char(value)
