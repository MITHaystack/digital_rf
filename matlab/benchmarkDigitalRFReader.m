% benchmarkDigitalRFReader.m is a script to benchmark reading speed
% requires Hdf5 test data in /tmp/benchmark as produced by
% benchmark_fr_write_hdf5.py
% $Id$

disp('Test of reading files with no compression and no checksum');
startTime = datenum(clock);
result = benchmark_driver('junk0');
endTime = datenum(clock);
t = endTime - startTime;
MBperSec = 4.0E3/(t*86400);
disp(sprintf('No compression, no checksum read rate %f MB/sec', MBperSec));

disp('Test of reading files with compression and checksum');
startTime = datenum(clock);
result = benchmark_driver('junk2');
endTime = datenum(clock);
t = endTime - startTime;
MBperSec = 4.0E3/(t*86400);
disp(sprintf('Compression and checksum read rate %f MB/sec', MBperSec));