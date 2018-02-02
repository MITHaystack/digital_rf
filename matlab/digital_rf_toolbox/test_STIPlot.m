% ----------------------------------------------------------------------------
% Copyright (c) 2018 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------
%
% Contributed by: Miguel Martinez Ledesma
%

%% clean environment
clear all

%% Global Configurations

% SUBSTITUTE YOUR OWN DIRECTORY OF DATA
top_level_directories = '/data/Example_Beacon_Pass/JROMWH_CASSIOPE_20170115_0842_combined';



%% Channel 1 Configurations

channel = 'cha';
subchannel = 1;

%% Initialize STIPlot of Channel 1

drf_plotter1 = STIPlot(top_level_directories ,channel, subchannel);

%% Visual Configurations of Channel 1

drf_plotter1.options_frames = 1;            %number of frames
drf_plotter1.options_num_fft = 2048;        %number of fft bins
drf_plotter1.options_bins = 512;            %number of time bins
drf_plotter1.options_integration = 2;       %number of samples integrated
drf_plotter1.options_decimation = 60;       %subsampling division number

%representation start time (string or datenum).
drf_plotter1.options_start_date = '2017-01-15 08:44:00';
%representation end time (string or datenum).
drf_plotter1.options_end_date = datenum('2017-01-15 08:49:00');

%title string
drf_plotter1.options_title = 'Example of Use of STIPlot';
%save file name
drf_plotter1.options_savename = 'save_example_channel_1.png';

%% Generate Plot of Channel 1

[fig, psd_data, time_axis, freq_axis] = drf_plotter1.plot();

%% Use the data obtained from the plotter to generate a 3D plot

fig3D=figure();
colormap('hot')

surf(time_axis,freq_axis./1e3,psd_data);
set(gcf, 'renderer', 'zbuffer');
shading interp

caxis([-70 5])
view([70 60])
axis tight
grid on
box on
datetick('x','dd HH:MM:SS','keepticks','keeplimits')
ylabel('f (kHz)', 'fontsize',8)
xlabel('time (UTC)', 'fontsize',8)
hc=colorbar;
ylabel(hc, 'Power Spectral Density (dB/Hz)')
title(sprintf('3D plot of \"%s\" (%s:%i)',top_level_directories,channel,subchannel), 'Interpreter', 'none')
set(gcf,'Position', [10, 10, 1000, 600])

print(fig3D,'save_example_channel_1_3D','-dpng');











%% Channel 2 Configurations

channel = 'chb';
subchannel = 1;

%% Initialize STIPlot of Channel 2

drf_plotter2 = STIPlot(top_level_directories ,channel, subchannel);

%% Visual Configurations of Channel 2

drf_plotter2.options_verbose = 0;           %show extra info
drf_plotter2.options_frames = 1;            %number of frames
drf_plotter2.options_num_fft = 2048;        %number of fft bins
drf_plotter2.options_bins = 512;            %number of time bins
drf_plotter2.options_integration = 2;       %number of samples integrated
drf_plotter2.options_decimation = 60;       %subsampling division number
drf_plotter2.options_mean = 0;              %detrend (subtract mean value)
drf_plotter2.options_zaxis = [-60 -25];     %color range control
drf_plotter2.options_colormap = 'default';  %colormap to use
drf_plotter2.options_windowlength = 900;    %window length
drf_plotter2.options_windowwidth = 600;     %window width

%representation start time (string or datenum).
drf_plotter2.options_start_date = '2017-01-15 08:44:00';
%representation end time (string or datenum).
drf_plotter2.options_end_date = datenum('2017-01-15 08:49:00');

%title string
drf_plotter2.options_title = 'Example of Use of STIPlot';
%save file name
drf_plotter2.options_savename = 'save_example_channel_2.png';

%% Generate Plot of Channel 2

[fig, psd_data, time_axis, freq_axis] = drf_plotter2.plot();

%% Use the data obtained from the plotter to generate a 3D plot

fig3D=figure();
colormap('bone')

surf(time_axis,freq_axis./1e3,psd_data);
set(gcf, 'renderer', 'zbuffer');
shading interp

caxis([-70 -30])
view([-20 70])
axis tight
grid on
box on
datetick('x','dd HH:MM:SS','keepticks','keeplimits')
ylabel('f (kHz)', 'fontsize',8)
xlabel('time (UTC)', 'fontsize',8)
hc=colorbar;
ylabel(hc, 'Power Spectral Density (dB/Hz)')
title(sprintf('3D plot of \"%s\" (%s:%i)',top_level_directories,channel,subchannel), 'Interpreter', 'none')
set(gcf,'Position', [10, 10, 1000, 600])

print(fig3D,'save_example_channel_2_3D','-dpng');
