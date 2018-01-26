# example prc transmit, record, and processing

# see the following paper for a description and application of the technique:
# Vierinen, J., Chau, J. L., Pfeffer, N., Clahsen, M., and Stober, G.,
# Coded continuous wave meteor radar, Atmos. Meas. Tech., 9, 829-839,
# doi:10.5194/amt-9-829-2016, 2016.

# create a waveform
python create_waveform.py -l 10000 -b 10 -s 0

# tx
python tx.py -m 192.168.10.2 -d "A:A" -f 3.6e6 -G 0.25 -g 0 -r 1e6 code-l10000-b10-000000.bin

# rx
thor.py -m 192.168.30.2 -d "A:A" -c hfrx -f 3.6e6 -r 1e6 -i 10 /data/prc

# analysis
python prc_analyze.py /data/prc -c hfrx -l 10000 -s 0
