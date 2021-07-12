# -*- coding: utf-8 -*-
from quartical.config.external import Gain
from quartical.config.internal import yield_from
from loguru import logger  # noqa
import numpy as np
import dask.array as da
import pathlib
import shutil
from daskms.experimental.zarr import xds_to_zarr
from quartical.gains import TERM_TYPES
from quartical.utils.dask import blockwise_unique
from quartical.utils.maths import mean_for_index
from quartical.gains.general.generics import combine_gains


def make_gain_xds_lod(data_xds_list,
                      tipc_list,
                      fipc_list,
                      coords_per_xds,
                      chain_opts):
    """Returns a list of dicts of xarray.Dataset objects describing the gains.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        tipc_list: List of numpy.ndarray objects containing number of time
            intervals in a chunk.
        fipc_list: List of numpy.ndarray objects containing number of freq
            intervals in a chunk.
        coords_per_xds: A List of Dicts containing coordinates.
        chain_opts: A Chain config object.

    Returns:
        gain_xds_lod: A List of Dicts of xarray.Dataset objects describing the
            gain terms assosciated with each data xarray.Dataset.
    """

    gain_xds_lod = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        term_xds_dict = {}

        term_coords = coords_per_xds[xds_ind]

        for loop_vars in enumerate(yield_from(chain_opts, "type")):
            term_ind, (term_name, term_type) = loop_vars

            term_t_chunks = tipc_list[xds_ind][:, :, term_ind]
            term_f_chunks = fipc_list[xds_ind][:, :, term_ind]
            term_opts = getattr(chain_opts, term_name)

            term_obj = TERM_TYPES[term_type](term_name,
                                             term_opts,
                                             data_xds,
                                             term_coords,
                                             term_t_chunks,
                                             term_f_chunks)

            term_xds_dict[term_name] = term_obj.make_xds()

        gain_xds_lod.append(term_xds_dict)

    return gain_xds_lod


def compute_interval_chunking(data_xds_list, t_map_list, f_map_list):
    '''Compute the per-term chunking of the gains.

    Given a list of data xarray.Datasets as well as information about the
    time and frequency mappings, computes the chunk sizes of the gain terms.

    Args:
        data_xds_list: A list of data-containing xarray.Dataset objects.
        t_map_list: A list of arrays describing how times map to solint.
        f_map_list: A list of arrays describing how freqs map to solint.

    Returns:
        A tuple of lists containing arrays which descibe the chunking.
    '''

    tipc_list = []
    fipc_list = []

    for xds_ind, _ in enumerate(data_xds_list):

        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]

        tipc_per_term = da.map_blocks(lambda arr: arr[:, -1:, :] + 1,
                                      t_map_arr,
                                      chunks=((2,),
                                              (1,)*t_map_arr.numblocks[1],
                                              t_map_arr.chunks[2]))

        fipc_per_term = da.map_blocks(lambda arr: arr[:, -1:, :] + 1,
                                      f_map_arr,
                                      chunks=((2,),
                                              (1,)*f_map_arr.numblocks[1],
                                              f_map_arr.chunks[2]))

        tipc_list.append(tipc_per_term)
        fipc_list.append(fipc_per_term)

    # This is an early compute which is necessary to figure out the gain dims.
    return da.compute(tipc_list, fipc_list)


def compute_dataset_coords(data_xds_list,
                           t_bin_list,
                           f_map_list,
                           tipc_list,
                           fipc_list,
                           terms):
    '''Compute the cooridnates for the gain datasets.

    Given a list of data xarray.Datasets as well as information about the
    binning along the time and frequency axes, computes the true coordinate
    values for the gain xarray.Datasets.

    Args:
        data_xds_list: A list of data-containing xarray.Dataset objects.
        t_bin_list: A list of arrays describing how times map to solint.
        f_map_list: A list of arrays describing how freqs map to solint.
        tipc_list: A list of arrays contatining the number of time intervals
            per chunk.
        fipc_list: A list of arrays contatining the number of freq intervals
            per chunk.

    Returns:
        A list of dictionaries containing the computed coordinate values.
    '''

    coords_per_xds = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        utime_chunks = list(map(int, data_xds.UTIME_CHUNKS))

        unique_times = blockwise_unique(data_xds.TIME.data,
                                        chunks=(utime_chunks,))
        unique_freqs = data_xds.CHAN_FREQ.data

        coord_dict = {"time": unique_times,  # Doesn't vary with term.
                      "freq": unique_freqs}  # Doesn't vary with term.

        for term_ind, term_name in enumerate(terms):

            # This indexing corresponds to grabbing the info per xds, per term.
            tipc = tipc_list[xds_ind][:, :, term_ind]
            fipc = fipc_list[xds_ind][:, :, term_ind]
            term_t_bins = t_bin_list[xds_ind][:, :, term_ind]
            term_f_map = f_map_list[xds_ind][:, :, term_ind]

            mean_gtimes = da.map_blocks(mean_for_index,
                                        unique_times,
                                        term_t_bins[0],
                                        dtype=unique_times.dtype,
                                        chunks=(tuple(map(int, tipc[0])),))

            mean_ptimes = da.map_blocks(mean_for_index,
                                        unique_times,
                                        term_t_bins[1],
                                        dtype=unique_times.dtype,
                                        chunks=(tuple(map(int, tipc[1])),))

            mean_gfreqs = da.map_blocks(mean_for_index,
                                        unique_freqs,
                                        term_f_map[0],
                                        dtype=unique_freqs.dtype,
                                        chunks=(tuple(map(int, fipc[0])),))

            mean_pfreqs = da.map_blocks(mean_for_index,
                                        unique_freqs,
                                        term_f_map[1],
                                        dtype=unique_freqs.dtype,
                                        chunks=(tuple(map(int, fipc[1])),))

            coord_dict[f"{term_name}_mean_gtime"] = mean_gtimes
            coord_dict[f"{term_name}_mean_ptime"] = mean_ptimes
            coord_dict[f"{term_name}_mean_gfreq"] = mean_gfreqs
            coord_dict[f"{term_name}_mean_pfreq"] = mean_pfreqs

        coords_per_xds.append(coord_dict)

    # We take the hit on a second early compute in order to make loading and
    # interpolating gains a less complicated operation.
    return da.compute(coords_per_xds)[0]


def make_net_xds_list(data_xds_list, coords_per_xds):
    """Construct a list of dicts of xarray.Datasets to house the net gains.

    Args:
        data_xds_list: A List of xarray.Dataset objects containing MS data.
        coords_per_xds: A List of Dicts containing dataset coords.

    Returns:
        net_gain_xds_list: A List of xarray.Dataset objects to house
            the net gains.
    """

    net_gain_xds_list = []

    for data_xds, xds_coords in zip(data_xds_list, coords_per_xds):

        net_t_chunks = np.tile(data_xds.UTIME_CHUNKS, 2).reshape(2, -1)
        net_f_chunks = np.tile(data_xds.chunks["chan"], 2).reshape(2, -1)

        # TODO: The net gain doesn't understand directions (yet).
        net_obj = TERM_TYPES["complex"]("NET",
                                        Gain(),
                                        data_xds,
                                        xds_coords,
                                        net_t_chunks,
                                        net_f_chunks)

        net_gain_xds_list.append(net_obj.make_xds())

    return net_gain_xds_list


def populate_net_xds_list(net_gain_xds_list,
                          solved_gain_xds_lod,
                          t_bin_list,
                          f_map_list,
                          d_map_list):
    """Poplulate the list net gain datasets with net gain values.

    Args:
        net_gain_xds_list: A List of xarray.Dataset objects to house the
            net gains.
        solved_gain_xds_lol: A List of Lists of xarray.Dataset objects housing
            the solved gain terms.
        t_bin_list: A List of dask.Arrays containing mappings from unique
            time to solution interval.
        f_map_list: A List of dask.Arrays containing mappings from channel
            to solution interval.
        d_map_list: A List of numpy.ndarrays containing mappings between
            direction dependent terms and direction independent terms.

    Returns:
        net_gain_xds_list: A List of xarray.Dataset objects to house the
            net gains.
    """

    populated_net_gain_xds_list = []

    for ind, (terms, net_xds) in enumerate(zip(solved_gain_xds_lod,
                                               net_gain_xds_list)):

        net_shape = tuple(net_xds.dims[d]
                          for d in ["gain_t", "gain_f", "ant", "dir", "corr"])

        gain_schema = ("time", "chan", "ant", "dir", "corr")

        gains = [x for xds in terms.values()
                 for x in (xds.gains.data, gain_schema)]
        corr_mode = "diag" if net_shape[-1] == 2 else "full"  # TODO: Use int.
        dtype = np.find_common_type(
            [xds.gains.dtype for xds in terms.values()], []
        )

        net_gain = da.blockwise(
            combine_gains, ("time", "chan", "ant", "dir", "corr"),
            t_bin_list[ind], ("param", "time", "term"),
            f_map_list[ind], ("param", "chan", "term"),
            d_map_list[ind], None,
            net_shape, None,
            corr_mode, None,
            *gains,
            dtype=dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"time": net_xds.GAIN_SPEC.tchunk,
                           "chan": net_xds.GAIN_SPEC.fchunk}
        )

        net_xds = net_xds.assign({"gains": (net_xds.GAIN_AXES, net_gain)})

        populated_net_gain_xds_list.append(net_xds)

    return populated_net_gain_xds_list


def write_gain_datasets(gain_xds_lod, net_xds_list, output_opts):
    """Write the contents of gain_xds_lol to zarr in accordance with opts."""

    root_path = pathlib.Path().absolute()  # Wherever the script is being run.
    gain_path = root_path.joinpath(output_opts.gain_dir)

    term_names = [xds.NAME for xds in gain_xds_lod[0].values()]

    writable_xds_dol = {tn: [d[tn] for d in gain_xds_lod] for tn in term_names}

    # If we are writing out the net/effective gains.
    if output_opts.net_gain:
        net_name = net_xds_list[0].NAME
        term_names.append(net_name)
        writable_xds_dol[net_name] = net_xds_list

    # If the directory in which we intend to store a gain already exists, we
    # remove it to make sure that we don't end up with a mix of old and new.
    for term_name in term_names:
        term_path = gain_path.joinpath(term_name)
        if term_path.is_dir():
            logger.info(f"Removing preexisting gain folder {term_path}.")
            try:
                shutil.rmtree(term_path)
            except Exception as e:
                logger.warning(f"Failed to delete {term_path}. Reason: {e}.")

    gain_writes_lol = []

    for term_name, term_xds_list in writable_xds_dol.items():

        # Remove chunking along all axes. Not ideal but necessary.
        term_xds_list = [xds.chunk({dim: -1 for dim in xds.dims})
                         for xds in term_xds_list]

        output_path = f"{gain_path}{'::' + term_name}"

        term_writes = xds_to_zarr(term_xds_list, output_path)

        gain_writes_lol.append(term_writes)

    # This converts the interpolated list of lists into a list of dicts.
    write_xds_lod = [{tn: term for tn, term in zip(term_names, terms)}
                     for terms in zip(*gain_writes_lol)]

    return write_xds_lod
