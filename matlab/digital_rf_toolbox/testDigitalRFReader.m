% test and example usage of DigitalRFReader.m

% ----------------------------------------------------------------------------
% Copyright (c) 2017, 2019 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------

% first test the reader on synthetic data
synthetic_directory = 'data/synthetic';
synthetic_reader = DigitalRFReader(synthetic_directory);
fprintf('Reading from directory: %s\n', synthetic_directory)
disp('These are the data channels available:')
chs = synthetic_reader.get_channels();
disp(chs);

for k = 1:length(chs)
    ch = chs{k};
    fprintf('-- Testing channel %s --\n\n', ch);
    [start_sample, end_sample] = synthetic_reader.get_bounds(ch);
    disp('These are the sample bounds:');
    disp([start_sample, end_sample]);
    disp('These are the channel parameters:');
    fprintf('subdir cadence: %d secs\n', synthetic_reader.get_subdir_cadence_secs(ch));
    fprintf('file cadence: %d msecs\n', synthetic_reader.get_file_cadence_millisecs(ch));
    fprintf('sample rate: %f Hz\n', synthetic_reader.get_samples_per_second(ch));
    fprintf('sample rate numerator: %d\n', synthetic_reader.get_sample_rate_numerator(ch));
    fprintf('sample rate denominator: %d\n', synthetic_reader.get_sample_rate_denominator(ch));
    fprintf('number of subchannels: %d\n', synthetic_reader.get_num_subchannels(ch));
    fprintf('is_complex: %d\n', synthetic_reader.get_is_complex(ch));
    fprintf('\n');

    disp('Read the first 10 samples using read_vector:');
    data = synthetic_reader.read_vector(ch, start_sample, 10);
    disp(data);

    disp('Get data block start samples and lengths using read:');
    length_map = synthetic_reader.read(ch, start_sample, end_sample, -1);
    block_start_samples = length_map.keys();
    for i = 1:length(block_start_samples)
        block_length = length_map(block_start_samples{i});
        fprintf('Block @ %d: %d samples\n', block_start_samples{i}, block_length);
    end
    fprintf('\n');

    disp('Read data by blocks using read:');
    data_map = synthetic_reader.read(ch, start_sample, end_sample, 0);
    block_start_samples = data_map.keys();
    for i = 1:length(block_start_samples)
        block_data = data_map(block_start_samples{i});
        fprintf('Block @ %d: (%d x %d)\n', block_start_samples{i}, size(block_data));
        fprintf('block_data(1:10, :):\n')
        disp(block_data(1:10, :));
    end
end

% demonstrate normal operation with example data
example_directory = 'data/example';
reader = DigitalRFReader(example_directory);
fprintf('Reading from directory: %s\n', example_directory)

chs = reader.get_channels();
ch = chs{1};
[start_sample, end_sample] = reader.get_bounds(ch);
fprintf('Channel %s has samples from %d to %d\n', ch, start_sample, end_sample);
data = reader.read_vector(ch, start_sample, 1000);
disp(data(1:10))
