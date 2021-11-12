import dask.array as da
import numpy as np
from uuid import uuid4
from loguru import logger  # noqa
from quartical.flagging.flagging_kernels import (compute_bl_mad_and_med,
                                                 compute_gbl_mad_and_med,
                                                 compute_chisq,
                                                 compute_mad_flags)


def finalise_flags(xds_list):
    """Finishes processing flags to produce writable flag data.

    Given a list of xarray.Dataset objects, uses the updated flag column to
    create appropriate flags for writing to disk. Removes all temporary flags.

    Args:
        xds_list: A list of xarray datasets.

    Returns:
        writable_xds: A list of xarray datasets.
    """

    writable_xds = []

    for xds in xds_list:

        data_col = xds.DATA.data
        flag_col = xds.FLAG.data

        # Remove QuartiCal's temporary flagging.
        flag_col = da.where(flag_col == -1, 0, flag_col)

        # Reintroduce the correlation axis.
        flag_col = da.broadcast_to(flag_col[:, :, None],
                                   data_col.shape,
                                   chunks=data_col.chunks)

        # Convert back to a boolean array.
        flag_col = flag_col.astype(np.bool)

        # Make the FLAG_ROW column consistent with FLAG.
        flag_row_col = da.all(flag_col, axis=(1, 2))

        updated_xds = xds.assign(
            {
                "FLAG": (xds.DATA.dims, flag_col),
                "FLAG_ROW": (xds.FLAG_ROW.dims, flag_row_col)
            }
        )

        writable_xds.append(updated_xds)

    return writable_xds


def initialise_flags(data_col, weight_col, flag_col, flag_row_col):
    """Given input data, weights and flags, initialise the aggregate flags.

    Populates the internal flag array based on existing flags and data
    points/weights which appear invalid.

    Args:
        data_col: A dask.array containing the data.
        weight_col: A dask.array containing the weights.
        flag_col: A dask.array containing the conventional flags.
        flag_row_col: A dask.array containing the conventional row flags.

    Returns:
        flags: A dask.array containing the initialized aggregate flags.
    """

    return da.blockwise(_initialise_flags, ("rowlike", "chan"),
                        data_col, ("rowlike", "chan", "corr"),
                        weight_col, ("rowlike", "chan", "corr"),
                        flag_col, ("rowlike", "chan", "corr"),
                        flag_row_col, ("rowlike",),
                        dtype=np.int8,
                        name="init_flags-" + uuid4().hex,
                        adjust_chunks=data_col.chunks,
                        align_arrays=False,
                        concatenate=True)


def _initialise_flags(data_col, weight_col, flag_col, flag_row_col):
    """See docstring for initialise_flags."""

    # Combine the flags from both the flag and flag_row columns.
    flags = flag_col | flag_row_col[:, None, None]

    # The following does some sanity checking on the input data and
    # weights. Specifically, we look for points with missing/broken data, and
    # points with null weights. TODO: We can do this with a much smaller
    # memory footprint by passing this into a numba loop which makes these
    # decisions per element.

    # We assume that the first and last entries of the correlation axis
    # are the on-diagonal terms. TODO: This should be safe provided we don't
    # have off-diagonal only data, although in that case the flagging
    # logic is probablly equally applicable.

    missing_points = np.any(data_col[..., (0, -1)] == 0, axis=-1)
    flags[missing_points] = True

    noweight_points = np.any(weight_col[..., (0, -1)] == 0, axis=-1)
    flags[noweight_points] = True

    # At this point, if any correlation is flagged, flag other correlations.
    flags = np.any(flags, axis=-1).astype(np.int8)

    return flags


def initialise_gainflags(gain, empty_intervals):
    """Given input data, weights and flags, initialise the internal bitflags.

    Populates the internal bitflag array based on existing flags and data
    points/weights which appear invalid.

    Args:
        gain: A dask.array containing the gains.
        empty_intervals: A dask.array containing intervals containing no data.

    Returns:
        A dask.array containing the initialized gainflags.
    """

    return da.map_blocks(_initialise_gainflags, gain, empty_intervals,
                         dtype=np.uint8, name="gflags-" + uuid4().hex)


def _initialise_gainflags(gain, empty_intervals):
    """See docstring for initialise_gainflags."""

    gainflags = np.zeros(gain.shape, dtype=np.uint8)

    return gainflags


def valid_median(arr):
    return np.median(arr[np.isfinite(arr) & (arr > 0)], keepdims=True)


def add_mad_graph(data_xds_list, mad_opts):

    bl_thresh = mad_opts.threshold_bl
    gbl_thresh = mad_opts.threshold_global
    max_deviation = mad_opts.max_deviation

    flagged_data_xds_list = []

    for xds in data_xds_list:
        residuals = xds._RESIDUAL.data
        weight_col = xds._WEIGHT.data
        flag_col = xds.FLAG.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        n_ant = xds.dims["ant"]
        n_t_chunk = residuals.numblocks[0]

        chisq = da.blockwise(compute_chisq, ("rowlike", "chan"),
                             residuals, ("rowlike", "chan", "corr"),
                             weight_col, ("rowlike", "chan", "corr"),
                             dtype=residuals.real.dtype,
                             align_arrays=False,
                             concatenate=True)

        bl_mad_and_med = da.blockwise(
            compute_bl_mad_and_med, ("rowlike", "ant1", "ant2"),
            chisq, ("rowlike", "chan"),
            flag_col, ("rowlike", "chan"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            n_ant, None,
            dtype=chisq.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (2,)*n_t_chunk},
            new_axes={"ant1": n_ant,
                      "ant2": n_ant}
        )

        gbl_mad_and_med = da.blockwise(
            compute_gbl_mad_and_med, ("rowlike",),
            chisq, ("rowlike", "chan"),
            flag_col, ("rowlike", "chan"),
            dtype=chisq.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (2,)*n_t_chunk}
        )

        row_chunks = residuals.chunks[0]

        mad_flags = da.blockwise(compute_mad_flags, ("rowlike", "chan"),
                                 chisq, ("rowlike", "chan"),
                                 gbl_mad_and_med, ("rowlike",),
                                 bl_mad_and_med, ("rowlike", "ant1", "ant2"),
                                 ant1_col, ("rowlike",),
                                 ant2_col, ("rowlike",),
                                 gbl_thresh, None,
                                 bl_thresh, None,
                                 max_deviation, None,
                                 dtype=np.int8,
                                 align_arrays=False,
                                 concatenate=True,
                                 adjust_chunks={"rowlike": row_chunks},)

        flag_col = da.where(mad_flags, 1, flag_col)

        flagged_data_xds = xds.assign({"FLAG": (("row", "chan"), flag_col)})

        flagged_data_xds_list.append(flagged_data_xds)

    return flagged_data_xds_list
