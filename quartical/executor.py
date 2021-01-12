# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.

from contextlib import ExitStack

import quartical.logging.init_logger  # noqa
from loguru import logger
from quartical.parser import parser, preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from quartical.flagging.flagging import finalise_flags
import time
from dask.diagnostics import ProgressBar
import dask
from dask.distributed import Client, LocalCluster
import zarr


@logger.catch
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    opts = parser.parse_inputs()

    # TODO: This check needs to be fleshed out substantially.

    preprocess.check_opts(opts)
    preprocess.interpret_model(opts)

    if opts.parallel_scheduler == "localcluster":
        logger.info("Initializing distributed client using LocalCluster.")
        cluster = LocalCluster(processes=opts.parallel_nworker > 1,
                               n_workers=opts.parallel_nworker,
                               threads_per_worker=opts.parallel_nthread,
                               memory_limit=0)
        cluster = exitstack.enter_context(cluster)
        exitstack.enter_context(Client(cluster))
        logger.info("Distributed client sucessfully initialized.")
    elif opts.parallel_scheduler == "distributed":
        logger.info("Initializing distributed client.")
        client = Client(opts.parallel_address)
        exitstack.enter_context(client)
        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    ms_xds_list, ref_xds_list, col_kwrds = read_xds_list(opts)

    # ms_xds_list = ms_xds_list[:2]
    # ref_xds_list = ref_xds_list[:16]

    # Preprocess the xds_list - initialise some values and fix bad data.
    preprocessed_xds_list = preprocess_xds_list(ms_xds_list, col_kwrds, opts)

    # Model xds is a list of xdss onto which appropriate model data has been
    # assigned.
    model_xds_list = add_model_graph(preprocessed_xds_list, opts)

    # Adds the dask graph describing the calibration of the data.
    gains_per_xds, post_gain_xds = \
        add_calibration_graph(model_xds_list, col_kwrds, opts)

    writable_xds = finalise_flags(post_gain_xds, col_kwrds, opts)

    writes = write_xds_list(writable_xds, ref_xds_list, col_kwrds, opts)

    # This shouldn't be here. TODO: Move into the calibrate code. In fact, this
    # entire write construction needs some tidying.
    store = zarr.DirectoryStore("cc_gains")

    for g_name, g_list in gains_per_xds.items():
        for g_ind, g in enumerate(g_list):
            g_list[g_ind] = g.chunk({"time_int": -1}).to_zarr(
                store,
                mode="w",
                group="{}{}".format(g_name, g_ind),
                compute=False)

    writes = [writes] if not isinstance(writes, list) else writes

    gain_writes = list(zip(*[gain for gain in gains_per_xds.values()]))

    stride = len(writes)//len(gain_writes)

    # Match up column and gain writes - avoids recompute, and necessary for
    # handling BDA data.
    outputs = []
    for ind in range(len(gain_writes)):

        ms_writes = writes[ind*stride: (ind + 1)*stride]

        outputs.append(dask.delayed(tuple)([*ms_writes, *gain_writes[ind]]))

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()

    with ProgressBar():
        dask.compute(outputs,
                     num_workers=opts.parallel_nthread,
                     optimize_graph=True,
                     scheduler=opts.parallel_scheduler)

    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # dask.visualize(outputs,
    #                color='order', cmap='autumn',
    #                filename='order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize(outputs,
    #                filename='graph.pdf',
    #                optimize_graph=True)

    # dask.visualize(*gains_per_xds["G"],
    #                filename='gain_graph',
    #                format="pdf",
    #                optimize_graph=True,
    #                collapse_outputs=True,
    #                node_attr={'penwidth': '4',
    #                           'fontsize': '18',
    #                           'fontname': 'helvetica'},
    #                edge_attr={'penwidth': '2', })
