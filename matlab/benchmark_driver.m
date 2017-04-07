% benchmark_driver is the function to test read speed
% $Id$

function result = benchmark_driver(channel)
    SAMPLES = 1000000;
    LOOPS = 1000;
    top_level_directories = char('/tmp/benchmark');
    reader = DigitalRFReader(top_level_directories);
    disp(sprintf('Running benchmark with channel %s', channel));
    [lower_sample, upper_sample] = reader.get_bounds(channel);
    for i = 1:LOOPS
        data = reader.read_vector(channel, lower_sample + (i-1)*SAMPLES, SAMPLES);
        if mod(i, 100) == 0
            disp(sprintf('%i out of 1000', i));
        end
    end
    result = 1;
end