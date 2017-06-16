% ----------------------------------------------------------------------------
% Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------
% example usage of DigitalRFReader.m
% Requires python test_digital_rf_hdf5.py be run first to create test data
% $Id$
top_level_directories = char('/tmp/hdf5', '/tmp/hdf52');
reader = DigitalRFReader(top_level_directories);
disp(reader.get_channels());

disp('First test is using gappy data');
[lower_sample, upper_sample] = reader.get_bounds('junk4.1');
disp([lower_sample, upper_sample]);
disp(reader.get_subdir_cadence_secs('junk4.1'));
disp(reader.get_file_cadence_millisecs('junk4.1'));
disp(reader.get_samples_per_second('junk4.1'));
disp(reader.get_is_complex('junk4.1'));
disp(reader.get_num_subchannels('junk4.1'));

disp('Read data itself');
gap_arr = reader.read('junk4.1', lower_sample, upper_sample, 2);
keys = gap_arr.keys();
disp(length(keys));
for i = 1:3
    key = keys{i}
    gap_arr(key)
end

disp('Now just get block lengths');
gap_arr = reader.read('junk4.1', lower_sample, upper_sample, -1);
keys = gap_arr.keys();
disp(length(keys));
for i = 1:3
    key = keys{i}
    gap_arr(key)
end

disp('Second test is using continuous data');
[lower_sample, upper_sample] = reader.get_bounds('junk1.2');
disp([lower_sample, upper_sample]);
arr = reader.read('junk1.2', lower_sample, upper_sample, 2);
keys = arr.keys();
disp(length(keys));
for i = 1:3
    key = keys{i}
    length(arr(key))
end

disp('now just get block lengths for junk1.2');
arr = reader.read('junk1.2', lower_sample, upper_sample, -1);
keys = arr.keys();
disp(length(keys));
for i = 1:3
    key = keys{i}
    arr(key)
end
