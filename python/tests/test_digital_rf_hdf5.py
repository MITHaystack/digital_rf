# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Tests for the digital_rf.digital_rf_hdf5 module."""

from __future__ import annotations

import itertools
import os

import h5py
import numpy as np
import packaging.version
import pytest

import digital_rf


def test_get_unix_time(
    sample_rate_numerator, sample_rate_denominator, start_datetime, start_global_index
):
    # test conversion of the start time
    dt, picoseconds = digital_rf.get_unix_time(
        start_global_index, sample_rate_numerator, sample_rate_denominator
    )
    assert dt == start_datetime.replace(tzinfo=None)
    assert picoseconds == start_datetime.microsecond * 1000000


class TestDigitalRFChannel:
    """Test writing and reading of a Digital RF channel."""

    @pytest.mark.firstonly(
        "data_params", "file_params", "hdf_filter_params", "sample_params"
    )
    def test_writer_init(
        self, channel, chdir, drf_writer_param_dict, drf_writer_factory
    ):
        """Test writer object's init."""
        channel = f"{channel}_init_test"

        # error when directory doesn't exist
        with pytest.raises(IOError):
            drf_writer_factory(directory=str(chdir.dirpath(channel)))

        chdir = chdir.dirpath().mkdir(channel)

        # invalid dtype
        with pytest.raises(TypeError):
            drf_writer_factory(directory=str(chdir), dtype="c7")
        with pytest.raises(RuntimeError):
            drf_writer_factory(directory=str(chdir), dtype="S8")
        dtype = np.dtype([("notr", "i2"), ("noti", "i2")])
        with pytest.raises(ValueError):
            drf_writer_factory(directory=str(chdir), dtype=dtype)

        # invalid subdir_cadence_secs
        for scs in (0, -10, 1.1):
            with pytest.raises(ValueError):
                drf_writer_factory(directory=str(chdir), subdir_cadence_secs=scs)

        # invalid file_cadence_millisecs
        for fcms in (0, -10, 1.1):
            with pytest.raises(ValueError):
                drf_writer_factory(directory=str(chdir), file_cadence_millisecs=fcms)

        # invalid subdir_cadence_secs and file_cadence_secs together
        # (subdir_cadence_secs*1000 % file_cadence_millisecs) != 0
        with pytest.raises(ValueError):
            drf_writer_factory(
                directory=str(chdir),
                file_cadence_millisecs=1001,
                subdir_cadence_secs=3600,
            )

        # invalid start_global_index
        with pytest.raises(ValueError):
            drf_writer_factory(directory=str(chdir), start_global_index=-1)

        # invalid sample_rate_numerator
        for srn in (0, -10, 1.1):
            with pytest.raises(ValueError):
                drf_writer_factory(directory=str(chdir), sample_rate_numerator=srn)

        # invalid sample_rate_denominator
        for srd in (0, -10, 1.1):
            with pytest.raises(ValueError):
                drf_writer_factory(directory=str(chdir), sample_rate_denominator=srd)

        # invalid UUID
        with pytest.raises(ValueError):
            drf_writer_factory(directory=str(chdir), uuid_str=0)

        # invalid compression_level
        for cl in (-1, 0.5, 10, 100):
            with pytest.raises(ValueError):
                drf_writer_factory(directory=str(chdir), compression_level=cl)

        # invalid num_subchannels
        with pytest.raises(ValueError):
            drf_writer_factory(directory=str(chdir), num_subchannels=0)

        # now create valid writer and check attributes
        p = drf_writer_param_dict
        with drf_writer_factory(directory=str(chdir)) as dwo:
            assert dwo.directory == str(chdir)
            assert dwo.dtype == p["dtype"]
            assert dwo.is_complex == p["is_complex"]
            assert dwo.subdir_cadence_secs == p["subdir_cadence_secs"]
            assert dwo.file_cadence_millisecs == p["file_cadence_millisecs"]
            assert dwo.start_global_index == p["start_global_index"]
            assert dwo.uuid == p["uuid_str"]
            assert dwo.compression_level == p["compression_level"]
            assert dwo.checksum == p["checksum"]
            assert dwo.num_subchannels == p["num_subchannels"]
            assert dwo.is_continuous == p["is_continuous"]

        # test failure when trying to use different writer settings
        different_settings = [
            dict(dtype="c8" if p["dtype"] is not np.dtype("c8") else "f8"),
            dict(
                subdir_cadence_secs=(
                    (
                        (p["subdir_cadence_secs"] * 1000) // p["file_cadence_millisecs"]
                        + 1
                    )
                    * p["file_cadence_millisecs"]
                )
            ),
            dict(file_cadence_millisecs=p["subdir_cadence_secs"] * 1000),
            dict(sample_rate_numerator=p["sample_rate_numerator"] + 1),
            dict(sample_rate_denominator=p["sample_rate_denominator"] + 1),
            dict(num_subchannels=p["num_subchannels"] + 1),
            dict(is_continuous=not p["is_continuous"]),
        ]
        for kwargs in different_settings:
            with pytest.raises(RuntimeError):
                drf_writer_factory(directory=str(chdir), **kwargs)

        # test success when trying to use the same settings
        drf_writer_factory(directory=str(chdir))

    @pytest.mark.firstonly("file_params", "hdf_filter_params", "sample_params")
    def test_writer_write_input_checking(
        self, channel, chdir, dtype, drf_writer_factory, is_complex, num_subchannels
    ):
        """Test input checking of writer object's write method."""
        # by always using two dimensions even for num_subchannels==1,
        # we test both the 2d (here) and 1d (other test fcn) input cases
        shape = (100, num_subchannels)
        chdir = chdir.dirpath().mkdir(f"{channel}_input_tests")
        with drf_writer_factory(directory=str(chdir)) as dwo:
            # test interleaved input format (complex writer only)
            if is_complex:
                next_rel_index = dwo.rf_write(
                    np.ones((shape[0], 2 * shape[1]), dtype=dwo.realdtype)
                )

            # test writing data of other types
            # base_type should upcast sucessfully, other_type should fail
            # (one byte integer values will upcast to almost anything else)
            if np.issubdtype(dwo.realdtype, np.unsignedinteger):
                base_type = "u1"
                other_type = f"i{dwo.realdtype.itemsize}"
            else:
                base_type = "i1"
                if np.issubdtype(dwo.realdtype, np.integer):
                    other_type = f"u{dwo.realdtype.itemsize}"
                else:
                    other_type = f"f{dwo.realdtype.itemsize * 2}"
            if is_complex:
                # test upcast from complex input
                # also tests struct dtype input when floating complex
                next_rel_index = dwo.rf_write(
                    np.ones(shape, dtype=np.dtype([("r", base_type), ("i", base_type)]))
                )
                # test upcast from interleaved format
                next_rel_index = dwo.rf_write(
                    np.ones((shape[0], 2 * shape[1]), dtype=base_type)
                )
                # fail writing data of complex but incompatible type
                with pytest.raises(TypeError):
                    dwo.rf_write(np.ones(shape, dtype=other_type))
                # fail writing real data when complex is expected
                with pytest.raises(ValueError):
                    dwo.rf_write(np.ones(shape, dtype=dwo.realdtype))
            else:
                next_rel_index = dwo.rf_write(np.ones(shape, dtype=base_type))
                # fail writing data of real but incompatible type
                with pytest.raises(TypeError):
                    dwo.rf_write(np.ones(shape, dtype=other_type))
                # fail writing complex data when real is expected
                with pytest.raises(TypeError):
                    dwo.rf_write(
                        np.ones(shape, dtype=np.dtype([("r", dtype), ("i", dtype)]))
                    )

            # fail at overwriting
            with pytest.raises(ValueError):
                dwo.rf_write(np.ones(shape, dtype=dtype), next_rel_index - 1)

            # fail at writing wrong shape
            with pytest.raises(ValueError):
                dwo.rf_write(np.ones(shape + (1,), dtype=dtype))

            # fail at writing wrong number of subchannels
            with pytest.raises(ValueError):
                dwo.rf_write(np.ones((shape[0], shape[1] + 1), dtype=dtype))

        # test initializing with interleaved format
        if is_complex:
            chdir = chdir.dirpath().mkdir(f"{channel}_input_tests_interleaved")
            if dtype.names is not None:
                realdtype = dtype["r"]
            else:
                realdtype = np.dtype(f"f{dtype.itemsize // 2}")
            with drf_writer_factory(
                directory=str(chdir), dtype=realdtype, is_complex=True
            ) as dwo:
                assert dwo.is_complex
                assert dwo.realdtype == realdtype
                assert dwo.dtype == dtype
                # test writing interleaved data
                dwo.rf_write(np.ones((shape[0], 2 * shape[1]), dtype=realdtype))
                # test writing complex/structured data
                dwo.rf_write(np.ones(shape, dtype=dtype))

    @pytest.mark.firstonly("hdf_filter_params")
    def test_writer_write_blocks(
        self,
        bounds,
        channel,
        chdir,
        data,
        data_block_slices,
        drf_writer_factory,
        start_global_index,
    ):
        """Test writer object's write_blocks method."""
        chdir = chdir.dirpath().mkdir(f"{channel}_write_blocks")
        with drf_writer_factory(directory=str(chdir)) as dwo:
            # write the specified data blocks from data_block_slices
            block_starts, block_stops = zip(*data_block_slices)
            global_sample_arr = np.asarray(block_starts) - start_global_index
            block_lengths = np.asarray(block_stops) - np.asarray(block_starts)
            block_sample_arr = np.concatenate(([0], np.cumsum(block_lengths)[:-1]))
            data_arr = np.concatenate(
                tuple(
                    data[(bs - bounds[0]) : (be - bounds[0])]
                    for bs, be in zip(block_starts, block_stops)
                )
            )

            next_rel_index = dwo.rf_write_blocks(
                data_arr,
                global_sample_arr=global_sample_arr,
                block_sample_arr=block_sample_arr,
            )
            assert next_rel_index == (block_stops[-1] - start_global_index)

            # fail with non-matching length of sample lists
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[next_rel_index, next_rel_index + 10],
                    block_sample_arr=[0],
                )
            # fail with bad type for sample lists
            with pytest.raises(TypeError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[next_rel_index + 1.1],
                    block_sample_arr=[0],
                )
            # fail with bad shape for sample lists
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[next_rel_index],
                    block_sample_arr=np.zeros((1, 1), dtype=np.uint64),
                )
            # fail with nonzero block_sample_arr[0]
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr, global_sample_arr=[next_rel_index], block_sample_arr=[10]
                )
            # fail with invalid increment in block_sample_arr
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[next_rel_index, next_rel_index + 10],
                    block_sample_arr=[0, 11],
                )
            # fail with non-increasing block_sample_arr
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[
                        next_rel_index,
                        next_rel_index + 1,
                        next_rel_index + 2,
                    ],
                    block_sample_arr=[0, 1, 0],
                )
            # fail with block_sample_arr value > len(arr)
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr[:5],
                    global_sample_arr=[next_rel_index, next_rel_index + 10],
                    block_sample_arr=[0, 10],
                )
            # fail at overwriting
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[next_rel_index - 1],
                    block_sample_arr=[0],
                )
            # fail with non-increasing global_sample_arr
            with pytest.raises(ValueError):
                dwo.rf_write_blocks(
                    data_arr,
                    global_sample_arr=[
                        bounds[1] - start_global_index + 1,
                        bounds[1] - start_global_index,
                    ],
                    block_sample_arr=[0, 1],
                )

    @pytest.mark.firstonly("hdf_filter_params")
    def test_writer_output_files_write_blocks(self, channel, chdir, data_file_list):
        """Test output files created by writer object in write_blocks test."""
        # check that the correct subdirs and filenames are produced
        # (we don't check the contents of the written files here since testing
        #  the reader will accomplish that as long as we keep in mind that
        #  errors there could actually indicate a problem in writing)
        chdir = chdir.dirpath(f"{channel}_write_blocks")
        drf_files = digital_rf.lsdrf(
            str(chdir),
            include_drf=True,
            include_dmd=False,
            include_drf_properties=False,
        )
        drf_files = [os.path.relpath(p, str(chdir)) for p in sorted(drf_files)]
        assert drf_files == data_file_list

    def test_writer_data_write(
        self,
        bounds,
        chdir,
        data,
        data_block_slices,
        data_file_list,
        drf_writer,
        start_global_index,
    ):
        """Test writing specific data with writer object's write method."""
        # write the specified data blocks from data_block_slices
        next_rel_index = drf_writer.get_next_available_sample()
        for k, (sstart, sstop) in enumerate(data_block_slices):
            rel_index = sstart - start_global_index
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            wdata = data[bstart:bstop]
            if k == (len(data_block_slices) - 1):
                # test writing data from a row-order (Fortran-style) array
                wdata = np.asfortranarray(wdata)
            if rel_index == next_rel_index:
                # test writing automatically to next sample
                next_rel_index = drf_writer.rf_write(wdata)
            else:
                # test writing to specified sample
                next_rel_index = drf_writer.rf_write(wdata, next_sample=rel_index)
            assert next_rel_index == rel_index + (bstop - bstart)

        # close writer to close out last file
        # (IMPORTANT: can't write using drf_writer after this test!)
        drf_writer.close()
        with pytest.raises(IOError):
            drf_writer.rf_write(data)

    def test_writer_output_files_write(
        self, bounds, chdir, data, data_file_list, num_subchannels
    ):
        """Test output files created by writer object in write test."""
        # check that the correct subdirs and filenames are produced
        # and that data that exists in the files is correct
        # Note: this does not fully check that everything we intended to write
        # was written correctly
        drf_files = digital_rf.lsdrf(
            str(chdir),
            include_drf=True,
            include_dmd=False,
            include_drf_properties=False,
        )
        drf_files = [os.path.relpath(p, str(chdir)) for p in sorted(drf_files)]
        assert drf_files == data_file_list

        for fname in data_file_list:
            fpath = str(chdir.join(fname))
            with h5py.File(fpath, "r") as f:
                rdata = f["rf_data"]
                data_index = f["rf_data_index"]
                for blk in range(data_index.shape[0]):
                    sstart = int(data_index[blk, 0])
                    bstart = int(data_index[blk, 1])
                    if blk + 1 == data_index.shape[0]:
                        bstop = rdata.shape[0]
                    else:
                        bstop = int(data_index[blk + 1, 1])
                    blk_data = rdata[bstart:bstop]
                    assert len(blk_data.shape) == 2
                    dstart = sstart - bounds[0]
                    dstop = sstart + (bstop - bstart) - bounds[0]
                    test_data = data[dstart:dstop]
                    with np.errstate(invalid="ignore"):
                        np.testing.assert_equal(
                            blk_data, test_data.reshape((-1, num_subchannels))
                        )

    def test_writer_get_total_samples_written(self, data_block_slices, drf_writer):
        """Test writer object's get_total_samples_written method."""
        num_written = sum([se - ss for ss, se in data_block_slices])
        assert drf_writer.get_total_samples_written() == num_written

    def test_writer_get_next_available_sample(
        self, data_block_slices, drf_writer, start_global_index
    ):
        """Test writer object's get_next_available_sample method."""
        next_sample = data_block_slices[-1][1] - start_global_index
        assert drf_writer.get_next_available_sample() == next_sample

    def test_writer_get_total_gap_samples(self, drf_writer):
        """Test writer object's get_total_gap_samples method."""
        gap_samples = (
            drf_writer.get_next_available_sample()
            - drf_writer.get_total_samples_written()
        )
        assert drf_writer.get_total_gap_samples() == gap_samples

    def test_writer_get_last_file_written(self, chdir, data_file_list, drf_writer):
        """Test writer object's get_last_file_written method."""
        last_file = os.path.normpath(drf_writer.get_last_file_written())
        last_file_full_path = str(chdir.join(data_file_list[-1]))
        assert last_file == last_file_full_path

    def test_writer_get_last_dir_written(self, chdir, data_file_list, drf_writer):
        """Test writer object's get_last_dir_written method."""
        last_dir = os.path.normpath(drf_writer.get_last_dir_written())
        last_dir_full_path = chdir.join(data_file_list[-1]).dirname
        assert last_dir == last_dir_full_path

    def test_writer_get_last_utc_timestamp(self, drf_writer):
        """Test writer object's get_last_utc_timestamp method."""
        assert drf_writer.get_last_utc_timestamp()

    @pytest.mark.firstonly(
        "data_params", "file_params", "hdf_filter_params", "sample_params"
    )
    def test_writer_get_version(self, drf_writer):
        """Test writer object's get_version method."""
        assert packaging.version.Version(drf_writer.get_version())

    def test_reader_get_channels(self, channel, drf_reader):
        """Test reader object's get_channels method."""
        channels = drf_reader.get_channels()
        assert channel in channels

    @pytest.mark.firstonly("data_params", "hdf_filter_params")
    def test_reader_get_bounds(self, bounds, channel, drf_reader):
        """Test reader object's get_bounds method."""
        bounds = drf_reader.get_bounds(channel)
        assert bounds == bounds

        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.get_bounds("not_a_channel")

    @pytest.mark.firstonly("hdf_filter_params")
    def test_reader_get_properties(
        self,
        bounds,
        channel,
        drf_reader,
        drf_writer_param_dict,
        sample_rate,
        start_timestamp_tuple,
        start_global_index,
    ):
        """Test reader object's get_properties method."""
        p = drf_writer_param_dict
        expected_properties = dict(
            H5Tget_class=None,
            H5Tget_offset=None,
            H5Tget_order=None,
            H5Tget_precision=None,
            H5Tget_size=None,
            digital_rf_time_description=None,
            digital_rf_version=None,
            epoch="1970-01-01T00:00:00Z",
            file_cadence_millisecs=p["file_cadence_millisecs"],
            is_complex=p["is_complex"],
            is_continuous=p["is_continuous"],
            num_subchannels=p["num_subchannels"],
            samples_per_second=(
                np.longdouble(sample_rate.numerator) / sample_rate.denominator
            ),
            sample_rate_numerator=p["sample_rate_numerator"],
            sample_rate_denominator=p["sample_rate_denominator"],
            sample_rate=sample_rate,
            subdir_cadence_secs=p["subdir_cadence_secs"],
        )
        props = drf_reader.get_properties(channel)
        for k, v in expected_properties.items():
            if v is not None:
                assert props[k] == v
            else:
                props[k]

        expected_sample_properties = dict(
            computer_time=None,
            init_utc_timestamp=start_timestamp_tuple[0],
            sequence_num=0,
            uuid_str=p["uuid_str"],
        )
        sample_props = drf_reader.get_properties(channel, sample=start_global_index)
        for k, v in expected_sample_properties.items():
            if v is not None:
                assert sample_props[k] == v
            else:
                sample_props[k]

        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.get_properties("not_a_channel")
        # fail to get properties when sample doesn't exist
        with pytest.raises(IOError):
            drf_reader.get_properties(channel, sample=bounds[0] - 1)
        with pytest.raises(IOError):
            drf_reader.get_properties(channel, sample=bounds[1] + 1)

    @pytest.mark.firstonly("data_params", "file_params", "hdf_filter_params")
    def test_reader_get_digital_metadata(
        self, channel, chdir, drf_reader, sample_params, start_global_index
    ):
        """Test reader object's get_digital_metadata method."""
        # first we need to write to the accompanying metadata channel
        metadata_dir = chdir.mkdir("metadata")
        dmd_writer = digital_rf.DigitalMetadataWriter(
            metadata_dir=str(metadata_dir),
            subdir_cadence_secs=sample_params["subdir_cadence_secs"],
            file_cadence_secs=1,
            sample_rate_numerator=(sample_params["sample_rate_numerator"]),
            sample_rate_denominator=(sample_params["sample_rate_denominator"]),
            file_name="metadata",
        )
        dmd_writer.write(start_global_index, dict(test=True))

        dmd_reader = drf_reader.get_digital_metadata(channel)
        assert type(dmd_reader) is digital_rf.DigitalMetadataReader

        # fail when channel doesn't exist
        with pytest.raises(IOError):
            drf_reader.get_digital_metadata("not_a_channel")

    @pytest.mark.firstonly("data_params", "file_params", "hdf_filter_params")
    def test_reader_read_metadata(
        self,
        bounds,
        channel,
        drf_reader,
        sample_rate,
        sample_rate_numerator,
        sample_rate_denominator,
        start_global_index,
    ):
        """Test reader object's read_metadata method."""
        # must be run after test_reader_get_digital_metadata which creates
        # the metadata that we'll be reading
        expected_metadata = dict(
            samples_per_second=(
                np.longdouble(sample_rate.numerator) / sample_rate.denominator
            ),
            sample_rate_numerator=sample_rate_numerator,
            sample_rate_denominator=sample_rate_denominator,
            sample_rate=sample_rate,
        )
        # read the blocks channel first, which has no Digital Metadata channel
        # all metadata
        block_channel = f"{channel}_write_blocks"
        md = drf_reader.read_metadata(bounds[0], bounds[1], block_channel, method=None)
        assert list(md.keys()) == [bounds[0]]
        for k, v in expected_metadata.items():
            assert md[start_global_index][k] == v
        md2 = drf_reader.read_metadata(
            bounds[0] + 1, bounds[1], block_channel, method="ffill"
        )
        assert list(md2.keys()) == [bounds[0] + 1]
        for k, v in expected_metadata.items():
            assert md2[start_global_index + 1][k] == v

        # read the regular channel, which has a Digital Metadata channel
        expected_metadata["test"] = True
        md = drf_reader.read_metadata(bounds[0], bounds[1], channel, method=None)
        assert list(md.keys()) == [start_global_index]
        for k, v in expected_metadata.items():
            assert md[start_global_index][k] == v
        md2 = drf_reader.read_metadata(
            start_global_index + 1, bounds[1], channel, method="ffill"
        )
        assert md == md2

    @pytest.mark.firstonly("data_params", "file_params", "hdf_filter_params")
    def test_reader_get_last_write(self, channel, chdir, data_file_list, drf_reader):
        """Test reader object's get_last_write method."""
        timestamp, fpath = drf_reader.get_last_write(channel)
        assert timestamp
        assert fpath == str(chdir.join(data_file_list[-1]))

        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.get_last_write("not_a_channel")

    @pytest.mark.firstonly("data_params", "file_params", "hdf_filter_params")
    def test_reader_get_continuous_blocks(
        self, bounds, channel, data_block_slices, drf_reader
    ):
        """Test reader object's get_continuous_blocks method."""
        # read all of the data blocks that were written in writer test
        for sstart, sstop in data_block_slices:
            data_dict = drf_reader.get_continuous_blocks(sstart, sstop - 1, channel)
            assert len(data_dict) == 1
            sample, nsamples = list(data_dict.items())[0]
            assert sample == sstart
            assert nsamples == sstop - sstart

        # check that we get all of the data if we read entire bounds
        data_dict = drf_reader.get_continuous_blocks(bounds[0], bounds[1], channel)
        assert bounds[0] in data_dict
        sample, nsamples = sorted(data_dict.items())[-1]
        assert sample + nsamples - 1 == bounds[1]

        # test read where data doesn't exist
        data_dict = drf_reader.get_continuous_blocks(
            bounds[0] - 100, bounds[0] - 1, channel
        )
        assert data_dict == {}

        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.get_continuous_blocks(bounds[0], bounds[1], "not_a_channel")

    def test_reader_read(
        self, bounds, channel, data, data_block_slices, drf_reader, num_subchannels
    ):
        """Test reader object's read method."""
        # read all of the data blocks that were written in writer test
        for sstart, sstop in data_block_slices:
            data_dict = drf_reader.read(sstart, sstop - 1, channel)
            assert len(data_dict) == 1
            sample, rdata = list(data_dict.items())[0]
            assert sample == sstart
            assert len(rdata.shape) == 2
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(
                    rdata, data.reshape((-1, num_subchannels))[bstart:bstop, :]
                )

        # test reading a single subchannel (the first)
        for sstart, sstop in data_block_slices:
            data_dict = drf_reader.read(sstart, sstop - 1, channel, sub_channel=0)
            assert len(data_dict) == 1
            sample, rdata = list(data_dict.items())[0]
            assert sample == sstart
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(
                    rdata, data.reshape((-1, num_subchannels))[bstart:bstop, 0]
                )

        # check that we get all of the data if we read entire bounds
        data_dict = drf_reader.read(bounds[0], bounds[1], channel)
        for sstart, rdata in data_dict.items():
            bstart = sstart - bounds[0]
            bstop = sstart + len(rdata) - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(
                    rdata, data.reshape((-1, num_subchannels))[bstart:bstop, :]
                )

        # test read where data doesn't exist
        data_dict = drf_reader.read(bounds[0] - 100, bounds[0] - 1, channel)
        assert data_dict == {}

        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.read(bounds[0], bounds[1], "not_a_channel")
        # fail when end_sample < start_sample (e.g. nsamples passed instead)
        with pytest.raises(ValueError):
            drf_reader.read(bounds[0], bounds[1] - bounds[0] + 1, channel)
        # fail when subchannel doesn't exist
        with pytest.raises(ValueError):
            drf_reader.read(bounds[0], bounds[1], channel, sub_channel=num_subchannels)

    def test_reader_read_vector_raw(
        self, bounds, channel, data, data_block_slices, drf_reader, num_subchannels
    ):
        """Test reader object's read_vector_raw method."""
        # read all of the data blocks that were written in writer test
        for sstart, sstop in data_block_slices:
            rdata = drf_reader.read_vector_raw(sstart, sstop - sstart, channel)
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(rdata, data[bstart:bstop])

        # test reading a single subchannel (the first)
        for sstart, sstop in data_block_slices:
            rdata = drf_reader.read_vector_raw(
                sstart, sstop - sstart, channel, sub_channel=0
            )
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(
                    rdata, data.reshape((-1, num_subchannels))[bstart:bstop, 0]
                )

        # fail to read all data at once because of gap
        with pytest.raises(IOError):
            rdata = drf_reader.read_vector_raw(
                bounds[0], bounds[1] - bounds[0] + 1, channel
            )

        # fail when data doesn't exist
        with pytest.raises(IOError):
            drf_reader.read_vector_raw(bounds[0] - 100, 99, channel)
        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.read_vector_raw(
                bounds[0], bounds[1] - bounds[0] + 1, "not_a_channel"
            )
        # fail when subchannel doesn't exist
        with pytest.raises(ValueError):
            drf_reader.read_vector_raw(
                bounds[0],
                bounds[1] - bounds[0] + 1,
                channel,
                sub_channel=num_subchannels,
            )

    def test_reader_read_vector(
        self, bounds, channel, data, data_block_slices, drf_reader, num_subchannels
    ):
        """Test reader object's read_vector method."""
        if data.dtype.names is not None:
            data_real = data["r"]
            data_imag = data["i"]
        else:
            data_real = data.real
            data_imag = data.imag

        # read all of the data blocks that were written in writer test
        for sstart, sstop in data_block_slices:
            rdata = drf_reader.read_vector(sstart, sstop - sstart, channel)
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(rdata.real, data_real[bstart:bstop])
                np.testing.assert_equal(rdata.imag, data_imag[bstart:bstop])

        # test reading a single subchannel (the first)
        for sstart, sstop in data_block_slices:
            rdata = drf_reader.read_vector(
                sstart, sstop - sstart, channel, sub_channel=0
            )
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(
                    rdata.real,
                    data_real.reshape((-1, num_subchannels))[bstart:bstop, 0],
                )
                np.testing.assert_equal(
                    rdata.imag,
                    data_imag.reshape((-1, num_subchannels))[bstart:bstop, 0],
                )

        # fail to read all data at once because of gap
        with pytest.raises(IOError):
            drf_reader.read_vector(bounds[0], bounds[1] - bounds[0] + 1, channel)

        # fail when data doesn't exist
        with pytest.raises(IOError):
            drf_reader.read_vector(bounds[0] - 100, 99, channel)
        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.read_vector(
                bounds[0], bounds[1] - bounds[0] + 1, "not_a_channel"
            )
        # fail when subchannel doesn't exist
        with pytest.raises(ValueError):
            drf_reader.read_vector(
                bounds[0],
                bounds[1] - bounds[0] + 1,
                channel,
                sub_channel=num_subchannels,
            )

    def test_reader_read_vector_1d(
        self, bounds, channel, data, data_block_slices, drf_reader, num_subchannels
    ):
        """Test reader object's read_vector_1d method."""
        if data.dtype.names is not None:
            data_real = data["r"]
            data_imag = data["i"]
        else:
            data_real = data.real
            data_imag = data.imag
        # make data have two dimensions regardless of shape, for reliable indexing
        data_real = data_real.reshape((-1, num_subchannels))
        data_imag = data_imag.reshape((-1, num_subchannels))

        # read all of the data blocks that were written in writer test
        for sstart, sstop in data_block_slices:
            rdata = drf_reader.read_vector_1d(sstart, sstop - sstart, channel)
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(rdata.real, data_real[bstart:bstop, 0])
                np.testing.assert_equal(rdata.imag, data_imag[bstart:bstop, 0])

        # test reading a different subchannel (the last)
        for sstart, sstop in data_block_slices:
            rdata = drf_reader.read_vector_1d(
                sstart, sstop - sstart, channel, sub_channel=num_subchannels - 1
            )
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            with np.errstate(invalid="ignore"):
                np.testing.assert_equal(
                    rdata.real, data_real[bstart:bstop, num_subchannels - 1]
                )
                np.testing.assert_equal(
                    rdata.imag, data_imag[bstart:bstop, num_subchannels - 1]
                )

        # fail to read all data at once because of gap
        with pytest.raises(IOError):
            drf_reader.read_vector_1d(bounds[0], bounds[1] - bounds[0] + 1, channel)

        # fail when data doesn't exist
        with pytest.raises(IOError):
            drf_reader.read_vector_1d(bounds[0] - 100, 99, channel)
        # fail when channel doesn't exist
        with pytest.raises(KeyError):
            drf_reader.read_vector_1d(
                bounds[0], bounds[1] - bounds[0] + 1, "not_a_channel"
            )
        # fail when subchannel doesn't exist
        with pytest.raises(ValueError):
            drf_reader.read_vector_1d(
                bounds[0],
                bounds[1] - bounds[0] + 1,
                channel,
                sub_channel=num_subchannels,
            )

    @pytest.mark.firstonly("data_params", "hdf_filter_params", "sample_params")
    def test_reader_multiple_topleveldirs(
        self,
        bounds,
        data,
        data_block_slices,
        drf_writer_factory,
        is_continuous,
        num_subchannels,
        start_global_index,
        tmpdir_factory,
    ):
        """Test reader object with multiple top-level directories."""
        # write data blocks alternately in multiple top-level directories
        channel = "ch0"
        tlds = [tmpdir_factory.mktemp("tld_test_") for k in range(3)]
        writers = [
            drf_writer_factory(directory=str(tld.mkdir(channel))) for tld in tlds
        ]
        for dwo, (sstart, sstop) in zip(itertools.cycle(writers), data_block_slices):
            rel_index = sstart - start_global_index
            bstart = sstart - bounds[0]
            bstop = sstop - bounds[0]
            dwo.rf_write(data[bstart:bstop], next_sample=rel_index)
        for dwo in writers:
            dwo.close()

        # get reader for all of the top-level directories
        with digital_rf.DigitalRFReader([str(d) for d in tlds]) as dro:
            # check bounds
            assert dro.get_bounds(channel) == bounds

            if not is_continuous:
                # read all of the data blocks
                for sstart, sstop in data_block_slices:
                    data_dict = dro.read(sstart, sstop - 1, channel)
                    assert len(data_dict) == 1
                    sample, rdata = list(data_dict.items())[0]
                    assert sample == sstart
                    bstart = sstart - bounds[0]
                    bstop = sstop - bounds[0]
                    with np.errstate(invalid="ignore"):
                        np.testing.assert_equal(
                            rdata, data.reshape((-1, num_subchannels))[bstart:bstop]
                        )
