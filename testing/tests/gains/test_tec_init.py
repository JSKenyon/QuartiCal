from copy import deepcopy
import pytest
import numpy as np
import dask.array as da
from quartical.calibration.calibrate import add_calibration_graph
from testing.utils.gains import apply_gains, reference_gains


@pytest.fixture(scope="module")
def opts(base_opts, select_corr):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.select_corr = select_corr
    _opts.solver.terms = ['G']
    _opts.solver.iter_recipe = [100]
    _opts.solver.propagate_flags = False
    _opts.solver.convergence_criteria = 1e-7
    _opts.solver.convergence_fraction = 1
    _opts.solver.threads = 2
    _opts.G.type = "tec"
    _opts.G.freq_interval = 0
    _opts.G.solve_per = "antenna"

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


@pytest.fixture(scope="module")
def true_values(predicted_xds_list):

    gain_list = []
    tec_list = []

    for xds in predicted_xds_list:

        n_ant = xds.dims["ant"]
        utime_chunks = xds.UTIME_CHUNKS
        n_time = sum(utime_chunks)
        chan_chunks = xds.chunks["chan"]
        n_chan = xds.dims["chan"]
        n_dir = xds.dims["dir"]
        n_corr = xds.dims["corr"]

        chan_freq = xds.CHAN_FREQ.data
        
        #bw bandwidth
        bw = chan_freq[-1] - chan_freq[0]

        single_wrap_delay = 1/bw

        chunking = (utime_chunks, chan_chunks, n_ant, n_dir, n_corr)

        da.random.seed(0)
        tecs = da.random.normal(size=(n_time, 1, n_ant, n_dir, n_corr),
                                  loc=0,
                                  scale=single_wrap_delay/5)
        # Make reference antenna zero for simplicity.
        tecs[:, :, 0] = 0  
        
        amp = da.ones((n_time, n_chan, n_ant, n_dir, n_corr),
                      chunks=chunking)

        if n_corr == 4:  # This solver only considers the diagonal elements.
            tecs *= da.array([1, 0, 0, 1])
            amp *= da.array([1, 0, 0, 1])

        phase = (2*np.pi*tecs/chan_freq[None, :, None, None, None])
        gains = amp*da.exp(1j*phase)

        tec_list.append(tecs)
        gain_list.append(gains)

    return gain_list, tec_list


@pytest.fixture(scope="module")
def true_gain_list(true_values):
    return true_values[0]


@pytest.fixture(scope="module")
def true_tec_list(true_values):
    return true_values[1]


@pytest.fixture(scope="module")
def corrupted_data_xds_list(predicted_xds_list, true_gain_list):

    corrupted_data_xds_list = []

    for xds, gains in zip(predicted_xds_list, true_gain_list):

        n_corr = xds.dims["corr"]

        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        time = xds.TIME.data

        row_inds = \
            time.map_blocks(lambda x: np.unique(x, return_inverse=True)[1])

        model = da.ones(xds.MODEL_DATA.data.shape, dtype=np.complex128)

        data = da.blockwise(apply_gains, ("rfc"),
                            model, ("rfdc"),
                            gains, ("rfadc"),
                            ant1, ("r"),
                            ant2, ("r"),
                            row_inds, ("r"),
                            n_corr, None,
                            align_arrays=False,
                            concatenate=True,
                            dtype=model.dtype)

        corrupted_xds = xds.assign({
            "DATA": ((xds.DATA.dims), data),
            "MODEL_DATA": ((xds.MODEL_DATA.dims), model),
            "FLAG": ((xds.FLAG.dims), da.zeros_like(xds.FLAG.data)),
            "WEIGHT": ((xds.WEIGHT.dims), da.ones_like(xds.WEIGHT.data))
            }
        )

        corrupted_data_xds_list.append(corrupted_xds)

    return corrupted_data_xds_list


@pytest.fixture(scope="module")
def add_calibration_graph_outputs(corrupted_data_xds_list, stats_xds_list,
                                  solver_opts, chain_opts, output_opts):
    # Overload this fixture as we need to use the corrupted xdss.
    return add_calibration_graph(corrupted_data_xds_list, stats_xds_list,
                                 solver_opts, chain_opts, output_opts)


# -----------------------------------------------------------------------------


def test_gains(cmp_gain_xds_lod, true_gain_list):

    for solved_gain_dict, true_gain in zip(cmp_gain_xds_lod, true_gain_list):
        solved_gain_xds = solved_gain_dict["G"]
        solved_gain, solved_flags = da.compute(solved_gain_xds.gains.data,
                                               solved_gain_xds.gain_flags.data)
        true_gain = true_gain.compute()  # TODO: This could be done elsewhere.

        n_corr = true_gain.shape[-1]

        solved_gain = reference_gains(solved_gain, n_corr)
        true_gain = reference_gains(true_gain, n_corr)

        true_gain[np.where(solved_flags)] = 0
        solved_gain[np.where(solved_flags)] = 0

        # To ensure the missing antenna handling doesn't render this test
        # useless, check that we have non-zero entries first.
        assert np.any(solved_gain), "All gains are zero!"
        np.testing.assert_array_almost_equal(true_gain, solved_gain, 1)


def test_tecs(cmp_gain_xds_lod, true_tec_list):

    for solved_gain_dict, true_tec in zip(cmp_gain_xds_lod, true_tec_list):
        solved_gain_xds = solved_gain_dict["G"]
        solved_tec, solved_flags = da.compute(
            solved_gain_xds.params.data,
            solved_gain_xds.param_flags.data
        )
        true_tec = true_tec.compute()

        n_corr = true_tec.shape[-1]

        # Pull out the delay values - this is a little confusing as the output
        # parameters are not ordered in the same way as a true values.
        if n_corr == 4:
            solved_tec = solved_tec[..., (1, 3)]
            true_tec = true_tec[..., (0, 3)]
        elif n_corr == 2:
            solved_tec = solved_tec[..., (1, 3)]
            true_tec = true_tec[..., (0, 1)]
        else:
            solved_tec = solved_tec[..., 1:]

        solved_tec -= solved_tec[:, :, :1]

        true_tec[np.where(solved_flags)] = 0
        solved_tec[np.where(solved_flags)] = 0

        # To ensure the missing antenna handling doesn't render this test
        # useless, check that we have non-zero entries first.
        assert np.any(solved_tec), "All tecs are zero!"
        np.testing.assert_array_almost_equal(true_tec, solved_tec)


# -----------------------------------------------------------------------------
