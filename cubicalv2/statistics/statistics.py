from cubicalv2.statistics.stat_kernels import (estimate_noise_kernel,
                                            #    accumulate_intervals,
                                            #    logical_and_intervals,
                                               column_to_tfadc,
                                               column_to_tfac)
from cubicalv2.utils.intervals import (column_to_tifiac,
                                       sum_intervals,
                                       data_schema,
                                       model_schema,
                                       gain_schema)

from cubicalv2.utils.maths import cabs2
import dask.array as da
import numpy as np
import xarray


def create_data_stats_xds(utime_val, n_chan, n_ant, n_chunks):
    """Set up a stats xarray dataset and define its coordinates."""

    stats_xds = xarray.Dataset(
        coords={"ant": ("ant", da.arange(n_ant, dtype=np.int16)),
                "time": ("time", utime_val),
                "chan": ("chan", da.arange(n_chan, dtype=np.int16)),
                "chunk": ("chunk", da.arange(n_chunks, dtype=np.int16))})

    return stats_xds


def create_gain_stats_xds(n_tint, n_fint, n_ant, n_dir, n_corr, n_chunk, name,
                          ind):
    """Set up a stats xarray dataset and define its coordinates."""

    stats_xds = xarray.Dataset(
        coords={"ant": ("ant", da.arange(n_ant, dtype=np.int16)),
                "time_int": ("time_int", da.arange(n_tint, dtype=np.int16)),
                "freq_int": ("freq_int", da.arange(n_fint, dtype=np.int16)),
                "dir": ("dir", da.arange(n_dir, dtype=np.int16)),
                "corr": ("corr", da.arange(n_corr, dtype=np.int16)),
                "chunk": ("chunk", da.arange(n_chunk, dtype=np.int16))},
        attrs={"name": "{}-{}".format(name, ind)})

    return stats_xds


def assign_noise_estimates(stats_xds, data_col, fullres_bitflags, ant1_col,
                           ant2_col, n_ant):
    """Wrapper and unpacker for the Numba noise estimator code.

    Uses blockwise and the numba kernel function to produce a noise estimate
    and inverse variance per channel per chunk of data_col.

    Args:
        data_col: A chunked dask array containing data (or the residual).
        fullres_bitflags: An chunked dask array containing bitflags.
        ant1_col: A chunked dask array of antenna values.
        ant2_col: A chunked dask array of antenna values.
        n_ant: Integer number of antennas.

    Returns:
        noise_est: Graph which produces noise estimates.
        inv_var_per_chan: Graph which produces inverse variance per channel.
    """

    noise_tuple = da.blockwise(
        estimate_noise_kernel, ("rowlike", "chan"),
        data_col, ("rowlike", "chan", "corr"),
        fullres_bitflags, ("rowlike", "chan", "corr"),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        n_ant, None,
        adjust_chunks={"rowlike": 1},
        concatenate=True,
        dtype=np.float32,
        align_arrays=False,
        meta=np.empty((0, 0), dtype=np.float32)
    )

    # The following unpacks values from the noise tuple of (noise_estimate,
    # inv_var_per_chan). Noise estimate is embedded in a 2D array in order
    # to make these blockwise calls less complicated - the channel dimension
    # is not meaningful and we immediately squeeze it out.

    noise_est = da.blockwise(
        lambda nt: nt[0], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        adjust_chunks={"rowlike": 1,
                       "chan": 1},
        dtype=np.float32).squeeze(axis=1)

    inv_var_per_chan = da.blockwise(
        lambda nt: nt[1], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        dtype=np.float32)

    updated_stats_xds = stats_xds.assign(
        {"inv_var_per_chan": (("chunk", "chan"), inv_var_per_chan),
         "noise_est": (("chunk",), noise_est)})

    return updated_stats_xds


def assign_tf_stats(stats_xds, fullres_bitflags, ant1_col,
                    ant2_col, time_ind, n_time_ind, n_ant, n_chunk,
                    n_chan, chunk_spec):

    # Get all the unflagged points.

    unflagged = fullres_bitflags == 0

    # Compute the number of unflagged points per row. Note that this includes
    # a summation over channel and correlation - version 1 did not have a
    # a correlation axis in the flags but I believe we should.
    rows_unflagged = unflagged.map_blocks(np.sum, axis=(1, 2),
                                          chunks=(unflagged.chunks[0],),
                                          drop_axis=(1, 2))

    # Determine the number of equations per antenna by summing the appropriate
    # values from the per-row unflagged values.
    eqs_per_ant = da.map_blocks(sum_eqs_per_ant, rows_unflagged, ant1_col,
                                ant2_col, n_ant, dtype=np.int64,
                                new_axis=1,
                                chunks=((1,)*n_chunk, (n_ant,)))

    # Determine the number of equations per time-frequency slot.
    eqs_per_tf = da.blockwise(sum_eqs_per_tf, ("rowlike", "chan"),
                              unflagged, ("rowlike", "chan", "corr"),
                              time_ind, ("rowlike",),
                              n_time_ind, ("rowlike",),
                              dtype=np.int64,
                              concatenate=True,
                              align_arrays=False,
                              adjust_chunks={"rowlike": tuple(chunk_spec)})

    # Determine the normalisation factor as the reciprocal of the equations
    # per time-frequency bin.
    tf_norm_factor = da.map_blocks(silent_divide,
                                   1, eqs_per_tf, dtype=np.float64)

    # Compute the total number of equations per chunk.
    total_eqs = da.map_blocks(lambda x: np.atleast_1d(np.sum(x)),
                              eqs_per_tf, dtype=np.int64,
                              drop_axis=1,
                              chunks=(1,))

    # Compute the overall normalisation factor.
    total_norm_factor = da.map_blocks(silent_divide,
                                      1, total_eqs, dtype=np.float64)

    # Assign the relevant values to the xds.
    modified_stats_xds = \
        stats_xds.assign({"eqs_per_ant": (("chunk", "ant"), eqs_per_ant),
                          "eqs_per_tf": (("time", "chan"), eqs_per_tf),
                          "tf_norm_factor": (("time", "chan"), tf_norm_factor),
                          "tot_norm_factor": (("chunk",), total_norm_factor)})

    return modified_stats_xds


def compute_model_stats(stats_xds, model_col, fullres_bitflags, ant1_col,
                        ant2_col, utime_ind, n_utime, n_ant, n_chunk,
                        n_chan, n_dir, chunk_spec):

    # Get all the unflagged points.

    unflagged = fullres_bitflags == 0

    abs_sqrd_model = model_col.map_blocks(cabs2, dtype=model_col.real.dtype)

    # Note that we currently do not use the weights here - this differs from
    # V1 and needs to be discussed. This collapses the abs^2 values into a
    # (time, freq, ant, dir, corr) array.

    abs_sqrd_model_tfadc = \
        da.blockwise(column_to_tfadc, ("rowlike", "chan", "ant", "dir", "corr"),
                     abs_sqrd_model, ("rowlike", "chan", "dir", "corr"),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     utime_ind, ("rowlike",),
                     n_utime, ("rowlike",),
                     n_ant, None,
                     dtype=abs_sqrd_model.dtype,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": chunk_spec})

    # Sum over the correlation axis as is done in V1. Note that we retain
    # correlation as a dummy index (D) so that we don't confuse arrays with
    # similar dimensions.

    abs_sqrd_model_tfadD = \
        abs_sqrd_model_tfadc.map_blocks(np.sum, axis=4, drop_axis=4,
                                        new_axis=4, keepdims=True)

    # This collapses the unflagged values into a (time, freq, ant, corr) array.

    unflagged_tfac = \
        da.blockwise(column_to_tfac, ("rowlike", "chan", "ant", "corr"),
                     unflagged, ("rowlike", "chan", "corr"),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     utime_ind, ("rowlike",),
                     n_utime, ("rowlike",),
                     n_ant, None,
                     dtype=np.int32,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": chunk_spec})

    # Sum over the correlation axis as is done in V1. Note that we retain
    # correlation as a dummy index (D) so that we don't confuse arrays with
    # similar dimensions.

    unflagged_tfaD = unflagged_tfac.map_blocks(np.sum, axis=3, drop_axis=3,
                                               new_axis=3, keepdims=True)

    # This is appropriate for the case where we sum over correlation.

    avg_abs_sqrd_model = \
        abs_sqrd_model_tfadD.map_blocks(silent_divide,
                                        unflagged_tfaD[..., None, :])

    # In the event that we want to retain the correlation axis, this code is
    # appropriate.

    # avg_abs_sqrd_model = \
    #     abs_sqrd_model_tfadc.map_blocks(silent_divide,
    #                                     unflagged_tfac[..., None, :])

    return avg_abs_sqrd_model


def assign_interval_stats(gain_xds, data_stats_xds, fullres_bitflags,
                          avg_abs_sqrd_model, ant1_col, ant2_col, t_map, f_map,
                          t_int_per_chunk, f_int_per_chunk, ti_chunks,
                          fi_chunks, t_int, f_int, n_utime):

    unflagged = fullres_bitflags == 0
    n_ant = gain_xds.dims["ant"]

    # This creates an (n_t_int, n_f_int, n_ant, n_corr) array of unflagged
    # points. Note that V1 did not retain a correlation axis.

    unflagged_tifiac = \
        da.blockwise(column_to_tifiac, ("rowlike", "chan", "ant", "corr"),
                     unflagged, ("rowlike", "chan", "corr"),
                     t_map, ("rowlike",),
                     f_map, ("rowlike",),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     t_int_per_chunk, ("rowlike",),
                     f_int_per_chunk, ("rowlike",),
                     n_ant, None,
                     dtype=np.int32,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": ti_chunks,
                                    "chan": fi_chunks[0]})

    # Antennas which ahve no unflagged points in an interval must be fully
    # flagged. Note that we reduce over the correlation axis here.

    flagged_tifia = da.all(unflagged_tifiac == 0, axis=-1)

    missing_fraction = \
        da.map_blocks(lambda x: np.atleast_1d(np.sum(x)/x.size),
                      flagged_tifia,
                      chunks=(1,),
                      drop_axis=(1, 2),
                      dtype=np.int32)

    updated_gain_xds = gain_xds.assign(
        {"missing_fraction": (("chunk",), missing_fraction)})

    # Sum the average abs^2 model over solution intervals.

    avg_abs_sqrd_model_int = \
        da.blockwise(sum_intervals, model_schema,
                     avg_abs_sqrd_model, model_schema,
                     t_int, None,
                     f_int, None,
                     dtype=np.float32,
                     concatenate=True,
                     align_arrays=False,
                     adjust_chunks={"rowlike": ti_chunks,
                                    "chan": fi_chunks[0]})

    sigma_sqrd_per_chan = \
        da.map_blocks(silent_divide, 1, data_stats_xds.inv_var_per_chan.data,
                      dtype=np.float64)

    sigma_sqrd_per_int = \
        da.blockwise(per_chan_to_per_int, model_schema,
                     sigma_sqrd_per_chan, ("rowlike", "chan"),
                     avg_abs_sqrd_model_int, model_schema,
                     n_utime, ("rowlike",),
                     t_int, None,
                     f_int, None,
                     dtype=np.float32,
                     concatenate=True,
                     align_arrays=False)

    unflagged_tifiaD = unflagged_tifiac.map_blocks(np.sum, axis=3, drop_axis=3,
                                                   new_axis=3, keepdims=True)

    # Note the egregious fudge factor of four. This was introduced to be
    # consistent with V1 which abandons doesn't count correlations. TODO:
    # Sit down with Oleg and figure out exactly what we want ot happen in V2.

    noise_to_signal_ratio = (4*sigma_sqrd_per_int /
        (unflagged_tifiaD[:, :, :, None, :]*avg_abs_sqrd_model_int))

    prior_gain_error = da.sqrt(noise_to_signal_ratio)



    # TODO: Handle direction pinning. Handle logging/stat reporting.

    return updated_gain_xds, flagged_tifia


def sum_eqs_per_ant(rows_unflagged, ant1_col, ant2_col, n_ant):

    eqs_per_ant = np.zeros((1, n_ant), dtype=np.int64)

    np.add.at(eqs_per_ant[0, :], ant1_col, rows_unflagged)
    np.add.at(eqs_per_ant[0, :], ant2_col, rows_unflagged)

    return 2*eqs_per_ant  # The conjugate points double the eqs.


def sum_eqs_per_tf(unflagged, time_ind, n_time_ind):

    _, n_chan, _ = unflagged.shape

    eqs_per_tf = np.zeros((n_time_ind.item(), n_chan), dtype=np.int64)

    np.add.at(eqs_per_tf, time_ind, unflagged.sum(axis=-1))

    return 4*eqs_per_tf  # Conjugate points + each row contributes to 2 ants.


def silent_divide(in1, in2):
    """Divides in1 by in2, supressing warnings. Division by zero gives zero."""

    with np.errstate(divide='ignore', invalid='ignore'):
        out_arr = np.where(in2 != 0, in1/in2, 0)

    return out_arr


def per_chan_to_per_int(sigma_sqrd_per_chan, avg_abs_sqrd_model_int, n_time,
                        t_int, f_int):
    """Converts per channel sigma squared into per interval sigma squared."""

    n_chan = sigma_sqrd_per_chan.shape[1]

    sigma_sqrd_per_int = np.zeros_like(avg_abs_sqrd_model_int,
                                       dtype=sigma_sqrd_per_chan.dtype)

    chan_per_int = np.add.reduceat(sigma_sqrd_per_chan,
                                   np.arange(0, n_chan, f_int),
                                   axis=1)
    time_per_int = np.add.reduceat(np.ones(n_time),
                                   np.arange(0, n_time, t_int))

    sigma_sqrd_per_int[:] = \
        (time_per_int[:, None]*chan_per_int)[..., None, None, None]

    return sigma_sqrd_per_int
