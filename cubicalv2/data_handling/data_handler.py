# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
from loguru import logger
import warnings


def read_ms(opts):
    """Reads an input measurement set and generates a number of data sets."""

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row. TODO: Add blocking
    # based on arbitrary columns/jumps. Figure out behaviour on multi-SPW/field
    # data. Figure out chunking based on a memory budget rather than as an
    # option.

    logger.debug("Setting up indexing xarray dataset.")

    indexing_xds = xds_from_ms(opts.input_ms_name,
                               columns=("TIME", "INTERVAL"),
                               index_cols=("TIME",),
                               group_cols=("SCAN_NUMBER",))

    # Read the antenna table and add the number of antennas to the options
    # namespace/dictionary. Leading underscore indiciates that this option is
    # private.

    antenna_xds = xds_from_table(opts.input_ms_name+"::ANTENNA")

    opts._n_ant = antenna_xds[0].dims["row"]

    logger.info("Antenna table indicates {} antennas were present for this "
                "observation.", opts._n_ant)

    # Check whether the BITFLAG column exists - if not, we will need to add it
    # or ignore it. We suppress xarray_ms warnings here as the columns may not
    # exist. TODO: Add the column, or re-add it if we think the column is
    # dodgy.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bitflag_xds = xds_from_ms(opts.input_ms_name,
                                  columns=("BITFLAG", "BITFLAG_ROW"))[0]

    vars(opts)["_bitflag_exists"] = "BITFLAG" in bitflag_xds
    vars(opts)["_bitflagrow_exists"] = "BITFLAG_ROW" in bitflag_xds

    logger.info("BITFLAG column {} present.",
                "is" if opts._bitflag_exists else "isn't")
    logger.info("BITFLAG_ROW column {} present.",
                "is" if opts._bitflagrow_exists else "isn't")

    # row_chunks is a dictionary containing row chunks per data set.

    row_chunks_per_xds = []

    for xds in indexing_xds:

        time_col = xds.TIME.data

        # Compute unique times, indices of their first ocurrence and number of
        # appearances.

        da_utimes, da_utime_inds, da_utime_counts = \
            da.unique(time_col, return_counts=True, return_index=True)

        utimes, utime_inds, utime_counts = da.compute(da_utimes,
                                                      da_utime_inds,
                                                      da_utime_counts)

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a duration rather than a number of intervals. TODO:
        # Need to take resulting chunks and reprocess them based on chunk-on
        # columns and jumps.

        if isinstance(opts.input_ms_time_chunk, float):

            interval_col = indexing_xds[0].INTERVAL.data

            da_cumint = da.cumsum(interval_col[utime_inds])
            da_cumint = da_cumint - da_cumint[0]
            da_cumint_ind = \
                (da_cumint//opts.input_ms_time_chunk).astype(np.int32)
            _, da_utime_per_chunk = \
                da.unique(da_cumint_ind, return_counts=True)
            utime_per_chunk = da_utime_per_chunk.compute()

            cum_utime_per_chunk = np.cumsum(utime_per_chunk)
            cum_utime_per_chunk = cum_utime_per_chunk - cum_utime_per_chunk[0]

        else:

            cum_utime_per_chunk = range(0,
                                        len(utimes),
                                        opts.input_ms_time_chunk)

        chunks = np.add.reduceat(utime_counts, cum_utime_per_chunk).tolist()

        row_chunks_per_xds.append({"row": chunks})

        logger.debug("Scan {}: row chunks: {}", xds.SCAN_NUMBER, chunks)

    # Once we have determined the row chunks from the indexing columns, we set
    # up an xarray data set for the data. Note that we will reload certain
    # indexing columns so that they are consistent with the chunking strategy.

    extra_columns = ()
    extra_columns += ("BITFLAG",) if opts._bitflag_exists else ()
    extra_columns += ("BITFLAG_ROW",) if opts._bitflagrow_exists else ()

    data_columns = ("TIME", "ANTENNA1", "ANTENNA2", "DATA", "MODEL_DATA",
                    "FLAG", "FLAG_ROW") + extra_columns

    data_xds = xds_from_ms(opts.input_ms_name,
                           columns=data_columns,
                           index_cols=("TIME",),
                           group_cols=("SCAN_NUMBER",),
                           chunks=row_chunks_per_xds)

    # If the BITFLAG and BITFLAG_ROW columns were missing, we simply add
    # appropriately sized dask arrays to the data sets. These are initialised
    # from the existing flag data. If required, these can be written back to
    # the MS at the end. TODO: Add writing functionality.

    for xds_ind, xds in enumerate(data_xds):
        xds_updates = {}
        if not opts._bitflag_exists:
            data = xds.FLAG.data.astype(np.int32) << 1
            schema = ("row", "chan", "corr")
            xds_updates["BITFLAG"] = (schema, data)
        if not opts._bitflagrow_exists:
            data = xds.FLAG_ROW.data.astype(np.int32) << 1
            schema = ("row",)
            xds_updates["BITFLAG_ROW"] = (schema, data)
        if xds_updates:
            data_xds[xds_ind] = xds.assign(xds_updates)

    return data_xds


def write_ms(xds_list, opts):

    return xds_to_table(xds_list, opts.input_ms_name, column="BITFLAG")


def handle_model(opts):

    pass
