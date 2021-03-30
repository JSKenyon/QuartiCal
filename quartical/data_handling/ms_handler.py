# -*- coding: utf-8 -*-
import dask
import dask.array as da
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
from quartical.weights.weights import initialize_weights
from quartical.flagging.flagging import initialise_flags
from quartical.data_handling.bda import process_bda_input, process_bda_output
from quartical.scheduling import annotate, dataset_partition
from dask.graph_manipulation import clone
from loguru import logger


def read_xds_list(opts):
    """Reads a measurement set and generates a list of xarray data sets.

    Args:
        opts: A Namepsace of global options.

    Returns:
        data_xds_list: A list of appropriately chunked xarray datasets.
    """

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row. TODO: Add blocking
    # based on arbitrary columns/jumps. Figure out behaviour on multi-SPW/field
    # data. Figure out chunking based on a memory budget rather than as an
    # option.

    logger.debug("Setting up indexing xarray dataset.")

    indexing_xds_list = xds_from_ms(opts.input_ms_name,
                                    columns=("TIME", "INTERVAL"),
                                    index_cols=("TIME",),
                                    group_cols=opts.input_ms_group_by,
                                    taql_where="ANTENNA1 != ANTENNA2",
                                    chunks={"row": -1})

    # Read the antenna table and add the number of antennas to the options
    # namespace/dictionary. Leading underscore indiciates that this option is
    # private and added internally.

    antenna_xds = xds_from_table(opts.input_ms_name + "::ANTENNA")[0]

    n_ant = antenna_xds.dims["row"]

    logger.info("Antenna table indicates {} antennas were present for this "
                "observation.", n_ant)

    # Determine the number of correlations present in the measurement set.

    polarization_xds = xds_from_table(opts.input_ms_name + "::POLARIZATION")[0]

    opts._ms_ncorr = polarization_xds.dims["corr"]

    if opts._ms_ncorr not in (1, 2, 4):
        raise ValueError("Measurement set contains {} correlations - this "
                         "is not supported.".format(opts._ms_ncorr))

    logger.info("Polarization table indicates {} correlations are present in "
                "the measurement set.", opts._ms_ncorr)

    # Determine the feed types present in the measurement set.

    feed_xds = xds_from_table(opts.input_ms_name + "::FEED")[0]

    feeds = feed_xds.POLARIZATION_TYPE.data.compute()
    unique_feeds = np.unique(feeds)

    if np.all([feed in "XxYy" for feed in unique_feeds]):
        opts._feed_type = "linear"
    elif np.all([feed in "LlRr" for feed in unique_feeds]):
        opts._feed_type = "circular"
    else:
        raise ValueError("Unsupported feed type/configuration.")

    logger.info("Feed table indicates {} ({}) feeds are present in the "
                "measurement set.", unique_feeds, opts._feed_type)

    # Determine the phase direction from the measurement set. TODO: This will
    # probably need to be done on a per xds basis. Can probably be accomplished
    # by merging the field xds grouped by DDID into data grouped by DDID.

    field_xds = xds_from_table(opts.input_ms_name + "::FIELD")[0]
    opts._phase_dir = np.squeeze(field_xds.PHASE_DIR.data.compute())

    logger.info("Field table indicates phase centre is at ({} {}).",
                opts._phase_dir[0], opts._phase_dir[1])

    # Check whether the specified weight column exists. If not, log a warning
    # and fall back to unity weights. TODO: Figure out how to prevent this
    # thowing a message wall.

    opts._unity_weights = opts.input_ms_weight_column.lower() == "unity"

    if not opts._unity_weights:
        col_names = list(xds_from_ms(opts.input_ms_name)[0].keys())
        if opts.input_ms_weight_column in col_names:
            logger.info(f"Using {opts.input_ms_weight_column} for weights.")
        else:
            logger.warning("Specified weight column was not present. "
                           "Falling back to unity weights.")
            opts._unity_weights = True

    # Determine the channels in the measurement set. Or handles unchunked case.
    # TODO: Handle multiple SPWs and specification in bandwidth.

    spw_xds_list = xds_from_table(
        opts.input_ms_name + "::SPECTRAL_WINDOW",
        group_cols=["__row__"],
        columns=["CHAN_FREQ", "CHAN_WIDTH"],
        chunks={"row": 1, "chan": opts.input_ms_freq_chunk or -1})

    # The spectral window xds should be correctly chunked in frequency.

    utime_chunking_per_xds, chunking_per_xds = \
        compute_chunking(indexing_xds_list, spw_xds_list, opts)

    # Once we have determined the row chunks from the indexing columns, we set
    # up an xarray data set for the data. Note that we will reload certain
    # indexing columns so that they are consistent with the chunking strategy.

    extra_columns = tuple(opts._model_columns)
    if not opts._unity_weights:
        extra_columns += (opts.input_ms_weight_column,)

    data_columns = ("TIME", "INTERVAL", "ANTENNA1", "ANTENNA2", "DATA", "FLAG",
                    "FLAG_ROW", "UVW") + extra_columns

    extra_schema = {cn: {'dims': ('chan', 'corr')}
                    for cn in opts._model_columns}

    data_xds_list = xds_from_ms(
        opts.input_ms_name,
        columns=data_columns,
        index_cols=("TIME",),
        group_cols=opts.input_ms_group_by,
        taql_where="ANTENNA1 != ANTENNA2",
        chunks=chunking_per_xds,
        table_schema=["MS", {**extra_schema}])

    # Preserve a copy of the xds_list prior to any BDA/assignment. Necessary
    # for undoing BDA.
    ref_xds_list = data_xds_list

    # BDA data needs to be processed into something more manageable.
    if opts.input_ms_is_bda:
        data_xds_list, utime_chunking_per_xds = \
            process_bda_input(data_xds_list, spw_xds_list, opts)

    # Add coordinates to the xarray datasets - this becomes immensely useful
    # down the line.
    data_xds_list = [xds.assign_coords({"corr": np.arange(xds.dims["corr"]),
                                        "chan": np.arange(xds.dims["chan"]),
                                        "ant": np.arange(n_ant)})
                     for xds in data_xds_list]

    # Add the actual channel frequecies to the xds - this is in preparation
    # for solvers which require this information. Also adds the antenna names
    # which will be useful when reference antennas are required.

    tmp_xds_list = []

    for xds in data_xds_list:
        chan_freqs = clone(spw_xds_list[xds.DATA_DESC_ID].CHAN_FREQ.data)
        chan_widths = clone(spw_xds_list[xds.DATA_DESC_ID].CHAN_FREQ.data)
        annotate(chan_freqs, dims=("chan",), partition=dataset_partition(xds))
        annotate(chan_widths, dims=("chan",), partition=dataset_partition(xds))
        tmp_xds_list.append(xds.assign(
            {"CHAN_FREQ": (("chan",), chan_freqs[0]),
             "CHAN_WIDTH": (("chan",), chan_widths[0]),
             "ANT_NAME": (("ant",), antenna_xds.NAME.data)}))

    data_xds_list = tmp_xds_list

    # Add an attribute to the xds on which we will store the names of fields
    # which must be written to the MS. Also add the attribute which stores
    # the unique time chunking per xds. We have to convert the chunking to
    # python integers to avoid problems with serialization.

    data_xds_list = \
        [xds.assign_attrs({
            "WRITE_COLS": (),
            "UTIME_CHUNKS": list(map(int, utime_chunking_per_xds[xds_ind]))})
         for xds_ind, xds in enumerate(data_xds_list)]

    # We may only want to use some of the input correlation values. xarray
    # has a neat syntax for this. #TODO: This needs to depend on the number of
    # correlations actually present in the MS/on the xds.

    if opts.input_ms_correlation_mode == "diag" and opts._ms_ncorr == 4:
        data_xds_list = [xds.sel(corr=[0, 3]) for xds in data_xds_list]
    elif opts.input_ms_correlation_mode == "full" and opts._ms_ncorr != 4:
        raise ValueError(f"--input-ms-correlation-mode was set to full, "
                         f"but the measurement set only contains "
                         f"{opts._ms_ncorr} correlations")

    annotate(data_xds_list)

    return data_xds_list, ref_xds_list


def write_xds_list(xds_list, ref_xds_list, opts):
    """Writes fields spicified in the WRITE_COLS attribute to the MS.

    Args:
        xds_list: A list of xarray datasets.
        ref_xds_list: A list of reference xarray.Dataset objects.
        opts: A Namepsace of global options.

    Returns:
        write_xds_list: A list of xarray datasets indicating success of writes.
    """

    # If we selected some correlations, we need to be sure that whatever we
    # attempt to write back to the MS is still consistent. This does this using
    # the magic of reindex. TODO: Check whether it would be better to let
    # dask-ms handle this. This also might need some further consideration,
    # as the fill_value might cause problems.

    if opts._ms_ncorr != xds_list[0].corr.size:
        xds_list = \
            [xds.reindex({"corr": np.arange(opts._ms_ncorr)}, fill_value=0)
             for xds in xds_list]

    output_cols = tuple(set([cn for xds in xds_list for cn in xds.WRITE_COLS]))

    if opts.output_visibility_product:
        # Drop variables from columns we intend to overwrite.
        xds_list = [xds.drop_vars(opts.output_column, errors="ignore")
                    for xds in xds_list]

        vis_prod_map = {"residual": "_RESIDUAL",
                        "corrected_residual": "_CORRECTED_RESIDUAL",
                        "corrected_data": "_CORRECTED_DATA"}
        n_vis_prod = len(opts.output_visibility_product)

        # Rename QuartiCal's underscore prefixed results so that they will be
        # written to the appropriate column.
        xds_list = \
            [xds.rename({vis_prod_map[prod]: opts.output_column[ind]
             for ind, prod in enumerate(opts.output_visibility_product)})
             for xds in xds_list]

        output_cols += tuple(opts.output_column[:n_vis_prod])

    if opts.input_ms_is_bda:
        xds_list = process_bda_output(xds_list, ref_xds_list, output_cols,
                                      opts)

    logger.info("Outputs will be written to {}.".format(
        ", ".join(output_cols)))

    # TODO: Nasty hack due to bug in daskms. Remove ASAP.
    xds_list = [xds.drop_vars(["ANT_NAME", "CHAN_FREQ", "CHAN_WIDTH"],
                              errors='ignore')
                for xds in xds_list]
    annotate(xds_list)

    write_xds_list = xds_to_table(xds_list, opts.input_ms_name,
                                  columns=output_cols)
    # This is a kludge to handle the fact that xds_to_table doesn't preserve
    # the annotation information/partition attributes. TODO: Improve upstream.
    [wds.attrs.update(ds.attrs) for ds, wds in zip(xds_list, write_xds_list)]
    annotate(write_xds_list)

    return write_xds_list


def preprocess_xds_list(xds_list, opts):
    """Adds data preprocessing steps - inits flags, weights and fixes bad data.

    Given a list of xarray.DataSet objects, initializes the flag data,
    the weight data and fixes bad data points (NaN, inf, etc). TODO: This
    function can likely be improved/extended.

    Args:
        xds_list: A list of xarray.DataSet objects containing MS data.
        opts: A Namepsace object of global options.

    Returns:
        output_xds_list: A list of xarray.DataSet objects containing MS data
            with preprocessing operations applied.
    """

    output_xds_list = []

    for xds_ind, xds in enumerate(xds_list):

        # Unpack the data on the xds into variables with understandable names.
        # We create copies of arrays we intend to mutate as otherwise we end
        # up implicitly updating the xds.
        data_col = xds.DATA.data
        flag_col = xds.FLAG.data
        flag_row_col = xds.FLAG_ROW.data

        # Anywhere we have a broken datapoint, zero it. These points will
        # be flagged below.
        finite_data = da.isfinite(data_col)
        annotate(finite_data,
                 dims=("row", "chan", "corr"),
                 partition=dataset_partition(xds))

        data_col = da.where(finite_data, data_col, 0)

        weight_col = initialize_weights(xds, data_col, opts)

        flag_col = initialise_flags(data_col,
                                    weight_col,
                                    flag_col,
                                    flag_row_col)

        # Anywhere we have a flag, we set the weight to 0.
        weight_col = da.where(flag_col, 0, weight_col)

        output_xds = xds.assign(
            {"DATA": (("row", "chan", "corr"), data_col),
             "WEIGHT": (("row", "chan", "corr"), weight_col),
             "FLAG": (("row", "chan", "corr"), flag_col)})

        output_xds_list.append(output_xds)

    annotate(output_xds_list)

    return output_xds_list


def compute_chunking(indexing_xds_list, spw_xds_list, opts, compute=True):
    """Compute time and frequency chunks for the input data.

    Given a list of indexing xds's, and a list of spw xds's, determines how to
    chunk the data given the chunking parameters.

    Args:
        indexing_xds_list: List of xarray.dataset objects contatining indexing
            information.
        spw_xds_list: List of xarray.dataset objects containing spectral window
            information.
        opts: A Namespace object containing options.
        compute: Boolean indicating whether or not to compute the result.

    Returns:
        A tuple of utime_chunking_per_xds and chunking_per_xds which describe
        the chunking of the data.
    """

    chan_chunks = {i: xds.chunks["chan"] for i, xds in enumerate(spw_xds_list)}

    # row_chunks is a list of dictionaries containing row chunks per data set.

    chunking_per_xds = []

    utime_chunking_per_xds = []

    for xds in indexing_xds_list:

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a duration rather than a number of intervals. TODO:
        # Need to take resulting chunks and reprocess them based on chunk-on
        # columns and jumps.

        # TODO: BDA will assume no chunking, and in general we can skip this
        # bit if the row axis is unchunked.

        if isinstance(opts.input_ms_time_chunk, float):

            def interval_chunking(time_col, interval_col, time_chunk):

                utimes, uinds, ucounts = \
                    np.unique(time_col, return_counts=True, return_index=True)
                cumulative_interval = np.cumsum(interval_col[uinds])
                cumulative_interval -= cumulative_interval[0]
                chunk_map = \
                    (cumulative_interval // time_chunk).astype(np.int32)

                _, utime_chunks = np.unique(chunk_map, return_counts=True)

                chunk_starts = np.zeros(utime_chunks.size, dtype=np.int32)
                chunk_starts[1:] = np.cumsum(utime_chunks)[:-1]

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks))

            chunking = da.map_blocks(interval_chunking,
                                     xds.TIME.data,
                                     xds.INTERVAL.data,
                                     opts.input_ms_time_chunk,
                                     chunks=((2,), (np.nan,)),
                                     dtype=np.int32)

        else:

            def integer_chunking(time_col, time_chunk):

                utimes, ucounts = np.unique(time_col, return_counts=True)
                n_utime = utimes.size
                time_chunk = time_chunk or n_utime  # Catch time_chunk == 0.

                utime_chunks = [time_chunk] * (n_utime // time_chunk)
                last_chunk = n_utime % time_chunk

                utime_chunks += [last_chunk] if last_chunk else []
                utime_chunks = np.array(utime_chunks)

                chunk_starts = np.arange(0, n_utime, time_chunk)

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks))

            chunking = da.map_blocks(integer_chunking,
                                     xds.TIME.data,
                                     opts.input_ms_time_chunk,
                                     chunks=((2,), (np.nan,)),
                                     dtype=np.int32)

        utime_per_chunk = dask.delayed(tuple)(chunking[0, :])
        row_chunks = dask.delayed(tuple)(chunking[1, :])

        utime_chunking_per_xds.append(utime_per_chunk)

        chunking_per_xds.append({"row": row_chunks,
                                 "chan": chan_chunks[xds.DATA_DESC_ID]})

    if compute:
        return da.compute(utime_chunking_per_xds, chunking_per_xds)
    else:
        return utime_chunking_per_xds, chunking_per_xds
