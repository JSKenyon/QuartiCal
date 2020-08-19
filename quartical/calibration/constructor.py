import numpy as np
from quartical.calibration.solver import solver_wrapper
from quartical.utils.dask import Blocker
from collections import namedtuple
from itertools import product


term_spec_tup = namedtuple("term_spec", "name type shape")


def construct_solver(model_col,
                     data_col,
                     ant1_col,
                     ant2_col,
                     weight_col,
                     t_map_arr,
                     f_map_arr,
                     d_map_arr,
                     corr_mode,
                     gain_xds_list,
                     opts):
    """Constructs the dask graph for the solver layer.

    This constructs a custom dask graph for the solver layer given the slew
    of solver inputs. This is arguably the most important function in V2 and
    should not be tampered with without a certain level of expertise with dask.

    Args:
        model_col: dask.Array containing the model column.
        data_col: dask.Array containing the data column.
        ant1_col: dask.Array containing the first antenna column.
        ant2_col: dask.Array containing the second antenna column.
        wegith_col: dask.Array containing the weight column.
        t_map_arr: dask.Array containing time mappings.
        f_map_arr: dask.Array containing frequency mappings.
        d_map_arr: dask.Array containing direction mappings.
        corr_mode: A string indicating the correlation mode.
        gain_xds_list: A list of xarray.Dataset objects describing gain terms.
        opts: A Namespace object containing global options.

    Returns:
        gain_list: A list of dask.Arrays containing the gains.
        conv_perc_list: A list of dask.Arrays containing the converged
            percentages.
        conv_iter_list: A list of dask.Arrays containing the iterations taken
            to reach convergence.
    """

    # Grab the number of input chunks - doing this on the data should be safe.
    n_t_chunks, n_f_chunks, _ = data_col.numblocks

    # Take the compact chunking info on the gain xdss and expand it.
    spec_list = expand_specs(gain_xds_list)

    # Create a blocker object.
    blocker = Blocker(solver_wrapper, "rf")

    # Add relevant inputs to the blocker object.
    blocker.add_input("model", model_col, "rfdc")
    blocker.add_input("data", data_col, "rfc")
    blocker.add_input("a1", ant1_col, "r")
    blocker.add_input("a2", ant2_col, "r")
    blocker.add_input("weights", weight_col, "rfc")
    blocker.add_input("t_map_arr", t_map_arr, "rj")
    blocker.add_input("f_map_arr", f_map_arr, "fj")
    blocker.add_input("d_map_arr", d_map_arr)
    blocker.add_input("corr_mode", corr_mode)
    blocker.add_input("term_spec_list", spec_list, "rf")

    # Add relevant outputs to blocker object.
    for gi, gn in enumerate(opts.solver_gain_terms):

        chunks = gain_xds_list[gi].CHUNK_SPEC
        blocker.add_output(f"{gn}-gain", "rfadc", chunks, np.complex128)

        chunks = ((1,)*n_t_chunks, (1,)*n_f_chunks)
        blocker.add_output(f"{gn}-conviter", "rf", chunks, np.int64)
        blocker.add_output(f"{gn}-convperc", "rf", chunks, np.float64)

    # Apply function to inputs to produce dask array outputs (as dict).
    output_array_dict = blocker.get_dask_outputs()

    # Assign results to the relevant gain xarray.Dataset object.
    solved_xds_list = []

    for gi, gain_xds in enumerate(gain_xds_list):

        gain = output_array_dict[f"{gain_xds.NAME}-gain"]
        convperc = output_array_dict[f"{gain_xds.NAME}-convperc"]
        conviter = output_array_dict[f"{gain_xds.NAME}-conviter"]

        solved_xds = gain_xds.assign(
            {"gains": (("time_int", "freq_int", "ant", "dir", "corr"), gain),
             "conv_perc": (("t_chunk", "f_chunk"), convperc),
             "conv_iter": (("t_chunk", "f_chunk"), conviter)})

        solved_xds_list.append(solved_xds)

    return solved_xds_list


def expand_specs(gain_xds_list):
    """Convert compact spec to a per-term list per-chunk."""

    spec_lists = []

    for gxds in gain_xds_list:

        term_name = gxds.NAME
        term_type = gxds.TYPE
        chunk_spec = gxds.CHUNK_SPEC

        ac = chunk_spec.achunk[0]  # No chunking along antenna axis.
        dc = chunk_spec.dchunk[0]  # No chunking along direction axis.
        cc = chunk_spec.cchunk[0]  # No chunking along correlation axis.

        shapes = [(tc, fc, ac, dc, cc)
                  for tc, fc in product(chunk_spec.tchunk, chunk_spec.fchunk)]

        term_spec_list = [term_spec_tup(term_name, term_type, shape)
                          for shape in shapes]

        spec_lists.append(term_spec_list)

    return list(zip(*spec_lists))
