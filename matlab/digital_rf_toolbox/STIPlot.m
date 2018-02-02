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

classdef STIPlot
    % class STIPlot allows an easy spectrogram plotting of Digital RF data

    properties (SetAccess = public)
        % Main object properties
        datadir                     % folder that contains all files
        drf_reader                  % the DigitalRFReader object
        drf_metadata_reader         % the DigitalMetadataReader object
        channel                     % the char array of channel name
        subchannel                  % the integer of subchannel index

        % Information about the channel (see DigitalRFReader object)
        channel_lower_sample
        channel_upper_sample
        channel_lower_datenum
        channel_upper_datenum
        channel_length
        channel_subdir_cadence_secs
        channel_file_cadence_millisecs
        channel_sample_rate_numerator
        channel_sample_rate_denominator
        channel_samples_per_second
        channel_is_complex
        channel_num_subchannels

        % Configurations
        options_title='Digital RF Data';% Use title provided for the data (string).
        options_start_date = [];        % Use the provided start time instead of the first time in the data (string or datenum).
        options_end_date = [];          % Use the provided end time for the plot (string or datenum).
        options_bins = 128;             % The number of time bins for the STI (integer).
        options_frames = 4;             % The number of sub-panel frames in the plot (integer).
        options_num_fft = 128;          % The number of FFT bints for the STI (integer).
        options_integration = 1;        % The number of rasters to integrate for each plot (integer).
        options_decimation = 1;         % The decimation factor for the data (integer).
        options_mean = 0;               % Remove the mean from the data at the PSD processing step (0 or 1).
        options_zaxis = [];             % zaxis colorbar setting e.g. -50:50 (values range)
        options_verbose= 0;             % Print status messages to stdout (0 or 1).
        options_savename = [];          % Name of file that figure will be saved under (string).
        options_colormap = [];          % Colormap to represent the color of image
        options_windowlength = 800;     % Length of the window in pixels (integer)
        options_windowwidth = 600;      % Width of the window in pixels (integer)
    end

    methods

        %------------------------------------------------------------------
        % INITIALIZATION FUNCTION

        function drf_plotter = STIPlot(datadir,channel,subchannel)
            % Initialize the STIPlot object and verify status
            %
            % Input:
            % - datadir: DigitalRFReader object
            % - channel: char array with channel name
            % - subchannel: integer with subchannel value

            % save properties
            drf_plotter.datadir = datadir;

            % open reader object
            drf_plotter.drf_reader = DigitalRFReader(datadir);

            % load available channels information
            drf_channels = drf_plotter.drf_reader.get_channels();
            % Verify if channel requested exists
            channel_found=0;
            for ch_num=1:length(drf_channels)
                if strcmp(channel,drf_channels{ch_num})==1
                    channel_found=1;
                    drf_plotter.channel=channel;
                    break
                end
            end
            if ~channel_found
                ME = MException('STIPlot:invalidChannel', ...
                  'no channel found \"%s\"',channel);
                throw(ME)
            end

            % Verify if subchannel requested exists
            num_subchannels=drf_plotter.drf_reader.get_num_subchannels(drf_plotter.channel);
            if subchannel <= num_subchannels
                drf_plotter.subchannel=subchannel;
            else
                ME = MException('STIPlot:invalidSubchannel', ...
                  'number of subchannels found is smaller than required subchannel (%i)',subchannel);
                throw(ME)
            end

            % get metadata reader object
            try
                drf_plotter.drf_metadata_reader = drf_plotter.drf_reader.get_digital_metadata(channel);
            catch
                drf_plotter.drf_metadata_reader = [];
                warning('Unable to open METADATA file...')
            end

            % Get some relevant information

            % calculate channel bounds
            [drf_plotter.channel_lower_sample, ...
                drf_plotter.channel_upper_sample] = ...
                drf_plotter.drf_reader.get_bounds(channel);
            % calculate channel length
            drf_plotter.channel_length = ...
                drf_plotter.channel_upper_sample - ...
                drf_plotter.channel_lower_sample + 1;
            % get channel properties
            drf_plotter.channel_subdir_cadence_secs = ...
                drf_plotter.drf_reader.get_subdir_cadence_secs(drf_plotter.channel);
            drf_plotter.channel_file_cadence_millisecs = ...
                drf_plotter.drf_reader.get_file_cadence_millisecs(drf_plotter.channel);
            drf_plotter.channel_sample_rate_numerator = ...
                drf_plotter.drf_reader.get_sample_rate_numerator(drf_plotter.channel);
            drf_plotter.channel_sample_rate_denominator = ...
                drf_plotter.drf_reader.get_sample_rate_denominator(drf_plotter.channel);
            drf_plotter.channel_samples_per_second = ...
                drf_plotter.drf_reader.get_samples_per_second(drf_plotter.channel);
            drf_plotter.channel_is_complex = ...
                drf_plotter.drf_reader.get_is_complex(drf_plotter.channel);
            drf_plotter.channel_num_subchannels = ...
                drf_plotter.drf_reader.get_num_subchannels(drf_plotter.channel);
            % convert bounds to datenums
            drf_plotter.channel_lower_datenum=drf_plotter.unixtime2datenum(drf_plotter.channel_lower_sample);
            drf_plotter.channel_upper_datenum=drf_plotter.unixtime2datenum(drf_plotter.channel_upper_sample);

        end


        %------------------------------------------------------------------
        % PLOT FUNCTION

        function [fig,PSD,time_axis,freq_axis]=plot(obj)
            % Show the Power Spectral Density of the STIPlot
            %
            % Input:
            %   none

            % initialize variables
            PSD=[];
            time_axis=[];
            freq_axis=[];

            % create the output figure
            fig=figure();
            set(gcf,'Color','white')

            % Verbose info
            if obj.options_verbose
                fprintf('# Sample rate:       %i\n', obj.channel_samples_per_second);
                fprintf('# Lower sample:      %i\n', obj.channel_lower_sample);
                fprintf('# Upper sample:      %i\n', obj.channel_upper_sample);
                fprintf('# Number of samples: %i\n', obj.channel_length);
            end

            % initial time configuration
            if ~isempty(obj.options_start_date)
                if ischar(obj.options_start_date)
                    datevalue = datenum(obj.options_start_date);
                elseif isnumeric(obj.options_start_date)
                    datevalue = obj.options_start_date;
                else
                    ME = MException('STIPlot:invalidconfiguration', ...
                    'unable to determine type of start date configuration');
                    throw(ME)
                end
                st0 = uint64((datevalue - datenum(1970,1,1))*86400);
                st0 = idivide(st0*obj.channel_sample_rate_numerator, ...
                              obj.channel_sample_rate_denominator);
            else
                st0 = obj.channel_lower_sample;
            end

            % ending time configuration
            if obj.options_end_date
                if ischar(obj.options_end_date)
                    datevalue = datenum(obj.options_end_date);
                elseif isnumeric(obj.options_end_date)
                    datevalue = obj.options_end_date;
                else
                    ME = MException('STIPlot:invalidconfiguration', ...
                    'unable to determine type of end date configuration');
                    throw(ME)
                end
                et0 = uint64((datevalue - datenum(1970,1,1))*86400);
                et0 = idivide(et0*obj.channel_sample_rate_numerator, ...
                              obj.channel_sample_rate_denominator);
            else
                et0 = obj.channel_upper_sample;
            end

            % Verbose info
            if obj.options_verbose
                fprintf('# Start sample st0:  %i\n', st0);
                fprintf('# End   sample et0:  %i\n', et0);
            end

            % Calculate number of blocks (at each frame)
            blocks = obj.options_bins * obj.options_frames;
            % Calculate number of sampes read each time
            samples_per_read = obj.options_num_fft *...
                                 obj.options_integration * ...
                                 obj.options_decimation;
            % Calculate the total number of samples to read
            total_samples = blocks * samples_per_read;

            % Verify if the total number of values is larger than the
            % available data
            if total_samples > (et0 - st0)
                ME = MException('STIPlot:invalidconfiguration', ...
                    'Insufficient samples for %i samples per read and %d blocks between %i and %i',samples_per_read, blocks, st0, et0);
                throw(ME)
            end

            % get number of steps
            read_step = (et0 - st0) / blocks;
            % get step size
            bin_step = read_step / obj.options_bins;

            % initialize sample number
            start_sample = st0;

            % Verbose info
            if obj.options_verbose
                fprintf ('# First sample:       %i\n', start_sample);
            end

            % Get Center Frequency (try to get it from metadata)
            cfreq = 0.0;
            if ~isempty(obj.drf_metadata_reader)
                [mdstart, mdend] = obj.drf_metadata_reader.get_bounds();
                md_map = obj.drf_metadata_reader.read(mdstart, mdend, 'center_frequencies');
                keys_cell = md_map.keys();
                keys = [keys_cell{:}];
                keyidx = find((st0 <= keys) & (keys <= et0), 1);
                if isempty(keyidx) && ~isempty(keys)
                    keyidx = find(keys <= st0, 1, 'last');
                end
                if ~isempty(keyidx)
                    md_val_map = md_map(keys(keyidx));
                    cfreqs = md_val_map('center_frequencies');
                    cfreq = cfreqs(obj.subchannel);
                end
            end

            % Verbose info
            if obj.options_verbose
                fprintf('# processing info :\n\tFrames: %i\n\tBins: %i\n\tsamples_per_read: %i\n\tbin_step:%i\n', ...
                    obj.options_frames, obj.options_bins, samples_per_read, bin_step);
            end

            % Loop for each frame configured
            for frame_num = 1:obj.options_frames

                % initialize data and times
                current_PSD = zeros(obj.options_num_fft, obj.options_bins);
                current_time_axis = zeros(obj.options_bins,1);

                for bin_num = 1:obj.options_bins

                    % Verbose info
                    if obj.options_verbose
                        fprintf('# Read vector (%i/%i) : [Channel: %s] [Start: %18i] [Time: %s] [Length: %i]\n', ...
                            bin_num, obj.options_bins, obj.channel, start_sample, datestr(obj.unixtime2datenum(start_sample)), samples_per_read);
                    end

                    % read data
                    d_vec = obj.drf_reader.read_vector(obj.channel, start_sample, samples_per_read);
                    data = double(d_vec(:, obj.subchannel));

                    %decimate sampling frequency
                    if obj.options_decimation > 1
                        data = decimate(data, obj.options_decimation);
                        sample_freq = obj.channel_samples_per_second / obj.options_decimation;
                    else
                        sample_freq = obj.channel_samples_per_second;
                    end

                    % integrate data
                    if obj.options_integration > 1
                        data = reshape(data,[length(data)/obj.options_integration obj.options_integration]);
                        data = mean(data,2);
                    end

                    % detrend the mean value (remove the mean of each segment before fft)
                    if obj.options_mean == 1
                         data = data-mean(data);
                    end

                    % calculate PSD
                    [psd_data,freq_axis] = obj.power_spectral_density (data, sample_freq);

                    % sequentially save psd data and time
                    current_PSD(:, bin_num) = real(10.*log10(abs(psd_data) + 1E-12));
                    current_time_axis(bin_num) = obj.unixtime2datenum(start_sample);

                    %increment sample time
                    start_sample = start_sample + read_step;
                end

                %Joint all values
                PSD=[PSD current_PSD];
                time_axis=[time_axis; current_time_axis];

                % Now Plot the Data
                subplot(obj.options_frames,1,frame_num);

                % determine image color extent in log scale units
                Pss = current_PSD;
                if ~isempty(obj.options_zaxis)
                    color_min = obj.options_zaxis(1);
                    color_max = obj.options_zaxis(2);

                    color_range=[color_min color_max];
                    color_min = min(color_range);
                    color_max = max(color_range);
                else
                    color_min = real(median(median(Pss)) - 6.0);
                    color_max = real(median(median(Pss)) + (max(max(Pss)) - median(median(Pss))) * 0.61803398875 + 50.0);
                end

                %change colormap
                if ~isempty(obj.options_colormap)
                    colormap(obj.options_colormap)
                end

                %plot image
                hi=image(current_time_axis,freq_axis/1e3,current_PSD,'CDataMapping','scaled');
                set(gca,'YDir','normal')
                caxis ([color_min color_max]);

                %graphic details
                grid on
                box on
                set(gca,'fontsize',8)
                ylabel('f (kHz)', 'fontsize',8)
                xlabel('time (UTC)', 'fontsize',8)

                %colorbar
                hc=colorbar('fontsize',8);
                ylabel(hc, 'Power Spectral Density (dB/Hz)', 'fontsize',8)

                %show date (x axys)
                set(gca,'XTick',unique(datenum(datestr(current_time_axis,'yyyy-mm-dd HH:MM'))))
                datetick('x','HH:MM:SS','keepticks','keeplimits')

                % Verbose info
                if obj.options_verbose
                    fprintf ('# Last sample:       %i\n', start_sample);
                end

                %create title
                start_time = obj.unixtime2datenum(st0);
                sub_second = (round((start_time - round(start_time)) * 100));
                timestamp = datestr(start_time,'dd-mm-yyyy HH:MM:SS');
                timestamp = sprintf('%s.%02d UT',timestamp,sub_second);
                title(sprintf('%s %s %4.2f MHz (%s) (%s:%i)', ...
                    obj.options_title, timestamp, cfreq / 1E6, obj.datadir,obj.channel,obj.subchannel), ...
                    'fontsize', 8, 'Interpreter', 'none');

                %resize window
                set(gcf,'Position', [10, 10, obj.options_windowlength, obj.options_windowwidth])

            end

            %save file
            if ~isempty(obj.options_savename)
                [~,fname,ext] = fileparts(obj.options_savename);
                if isempty(ext)
                    ext = '.png';
                end
                savefilename = sprintf('%s%s',fname,ext);
                fprintf('Saving plot as \"%s\"...',savefilename);
                print(fig,savefilename,'-dpng');
                fprintf('done!\n');
            end

        end

        %------------------------------------------------------------------
        % PSD GENERATE FUNCTION


        function [pds,freq] = power_spectral_density (obj,values, Fs)
            % Calculate Power Spectral Density from FFT
            % without the need of the periodogram function
            % of the Signal Processing Toolbox
            %
            % Input:
            % - values: data values
            % - Fs: sampling frequency

            N = length(values);
            spectrum = fftshift(fft(values));
            pds = (1/(Fs*N)) * abs(spectrum).^2;
            dF = Fs/N;
            freq = -Fs/2:dF:Fs/2 - dF;
        end

        function [datenum_value] = unixtime2datenum(obj,unixdate)
            % Convert Unix internal format to matlab datenum

            datenum_value = (double(unixdate) ./ obj.channel_samples_per_second)./86400.0 + datenum(1970,1,1);
        end


    end % end methods

end % end STIPlot class
