import pytest
from quartical.config import preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import (make_gain_xds_list,
                                             add_calibration_graph)
from quartical.calibration.mapping import (make_t_maps,
                                           make_f_maps)
import numpy as np
from copy import deepcopy


@pytest.fixture(scope="module")
def opts(base_opts, time_chunk, freq_chunk, time_int, freq_int):

    # Don't overwrite base config - instead duplicate and update.

    _opts = deepcopy(base_opts)

    _opts.input_model.recipe = "MODEL_DATA"
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.freq_chunk = freq_chunk
    _opts.G.time_interval = time_int
    _opts.B.time_interval = 2*time_int
    _opts.G.freq_interval = freq_int
    _opts.B.freq_interval = 2*freq_int
    _opts.mad_flags.enable = True

    return _opts


@pytest.fixture(scope="module")
def _recipe(opts):
    return preprocess.transcribe_recipe(opts)


@pytest.fixture(scope="module")
def _read_xds_list(opts, _recipe):
    return read_xds_list(_recipe.ingredients.model_columns, opts)


@pytest.fixture(scope="module")
def data_xds_list(_read_xds_list, _recipe, opts):

    ms_xds_list, _ = _read_xds_list

    preprocessed_xds_list = preprocess_xds_list(ms_xds_list, opts)

    data_xds_list = add_model_graph(preprocessed_xds_list, _recipe, opts)

    return data_xds_list


@pytest.fixture(scope="module")
def data_xds(data_xds_list):

    return data_xds_list[0]  # We only need to test on one.


@pytest.fixture(scope="module")
def expected_t_ints(data_xds, opts):

    n_row = data_xds.dims["row"]
    t_ints = [getattr(opts, term).time_interval or n_row
              for term in opts.solver.gain_terms]

    expected_t_ints = []

    da_ivl_col = data_xds.INTERVAL.data
    da_time_col = data_xds.TIME.data

    for t_int in t_ints:
        if isinstance(t_int, float):
            term_expected_t_ints = []
            for ind in range(da_ivl_col.npartitions):
                ivl_col = da_ivl_col.blocks[ind].compute()
                time_col = da_time_col.blocks[ind].compute()
                _, uinds = np.unique(time_col, return_index=True)
                utime_ivl = ivl_col[uinds]
                num_int = 0
                sol_width = 0
                for ivl in utime_ivl:
                    sol_width += ivl
                    if sol_width >= t_int:
                        num_int += 1
                        sol_width = 0
                if sol_width:
                    num_int += 1
                    sol_width = 0
                term_expected_t_ints.append(num_int)
            expected_t_ints.append(term_expected_t_ints)
        else:
            expected_t_ints.append([np.ceil(tc/t_int)
                                    for tc in data_xds.UTIME_CHUNKS])

    return expected_t_ints


@pytest.fixture(scope="module")
def expected_f_ints(data_xds, opts):

    n_chan = data_xds.dims["chan"]
    f_ints = [getattr(opts, term).freq_interval or n_chan
              for term in opts.solver.gain_terms]

    expected_f_ints = []

    da_chan_freq_col = data_xds.CHAN_FREQ.data
    da_chan_width_col = data_xds.CHAN_WIDTH.data

    for f_int in f_ints:
        if isinstance(f_int, float):
            term_expected_f_ints = []
            for ind in range(da_chan_freq_col.npartitions):
                chan_width_col = da_chan_width_col.blocks[ind].compute()
                num_int = 0
                sol_width = 0
                for cw in chan_width_col:
                    sol_width += cw
                    if sol_width >= f_int:
                        num_int += 1
                        sol_width = 0
                if sol_width:
                    num_int += 1
                    sol_width = 0
                term_expected_f_ints.append(num_int)
            expected_f_ints.append(term_expected_f_ints)
        else:
            expected_f_ints.append([np.ceil(fc/f_int)
                                    for fc in da_chan_freq_col.chunks[0]])

    return expected_f_ints


@pytest.fixture(scope="module")
def _add_calibration_graph(data_xds_list, opts):

    return add_calibration_graph(data_xds_list, opts)


@pytest.fixture(scope="module")
def tbin_list_tmap_list(data_xds_list, opts):
    return make_t_maps(data_xds_list, opts)


@pytest.fixture(scope="module")
def t_map_list(tbin_list_tmap_list):
    return tbin_list_tmap_list[1]


@pytest.fixture(scope="module")
def t_bin_list(tbin_list_tmap_list):
    return tbin_list_tmap_list[0]


@pytest.fixture(scope="module")
def f_map_list(data_xds_list, opts):
    return make_f_maps(data_xds_list, opts)


@pytest.fixture(scope="module")
def gain_xds_list(data_xds_list, t_map_list, t_bin_list, f_map_list, opts):
    return make_gain_xds_list(data_xds_list, t_map_list, t_bin_list,
                              f_map_list, opts)


@pytest.fixture(scope="module")
def term_xds_list(gain_xds_list):
    return gain_xds_list[0]


@pytest.fixture(scope="module")
def solved_gain_xds_list(_add_calibration_graph):

    return _add_calibration_graph[0]


@pytest.fixture(scope="module")
def post_cal_data_xds_list(_add_calibration_graph):

    return _add_calibration_graph[1]


# ----------------------------make_gain_xds_list-------------------------------


def test_nterm(gain_xds_list, opts):
    """Each gain term should produce a gain xds."""

    assert len(opts.solver.gain_terms) == len(gain_xds_list)


def test_data_coords(data_xds, term_xds_list):
    """Check that dimensions shared between the gains and data are the same."""

    data_coords = ["ant", "dir", "corr"]

    assert all(data_xds.dims[d] == gxds.dims[d]
               for gxds in term_xds_list
               for d in data_coords)


def test_t_chunking(data_xds, term_xds_list):
    """Check that time chunking of the gain xds list is correct."""

    assert all(len(data_xds.UTIME_CHUNKS) == gxds.dims["t_chunk"]
               for gxds in term_xds_list)


def test_f_chunking(data_xds, term_xds_list, opts):
    """Check that frequency chunking of the gain xds list is correct."""

    assert all(len(data_xds.chunks["chan"]) == gxds.dims["f_chunk"]
               for gxds in term_xds_list)


def test_t_ints(data_xds, term_xds_list, expected_t_ints):
    """Check that the time intervals are correct."""

    assert all(int(sum(eti)) == gxds.dims["gain_t"]
               for eti, gxds in zip(expected_t_ints, term_xds_list))


def test_f_ints(data_xds, term_xds_list, expected_f_ints):
    """Check that the frequency intervals are correct."""

    assert all(int(sum(efi)) == gxds.dims["gain_f"]
               for efi, gxds in zip(expected_f_ints, term_xds_list))


def test_attributes(data_xds, term_xds_list):
    """Check that the attributes of the gains are the same as the data."""

    data_attributes = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

    assert all(data_xds.attrs[a] == gxds.attrs[a]
               for gxds in term_xds_list
               for a in data_attributes)


def test_chunk_spec(data_xds, term_xds_list, expected_t_ints, expected_f_ints,
                    opts):
    """Check that the chunking specs are correct."""

    n_row, n_chan, n_ant, n_dir, n_corr = \
        [data_xds.dims[d] for d in ["row", "chan", "ant", "dir", "corr"]]

    specs = [tuple([tuple(tic), tuple(fic), (n_ant,), (n_dir,), (n_corr,)])
             for tic, fic in zip(expected_t_ints, expected_f_ints)]

    assert all(spec == gxds.attrs["GAIN_SPEC"]
               for spec, gxds in zip(specs, term_xds_list))

# ---------------------------add_calibration_graph-----------------------------


def test_ngains(solved_gain_xds_list, opts):
    """Check that calibration produces one xds per gain per input xds."""

    assert all([len(term_xds_list) == len(opts.solver.gain_terms)
                for term_xds_list in solved_gain_xds_list])


def test_has_gain_field(solved_gain_xds_list):
    """Check that calibration assigns the gains to the relevant xds."""

    assert all([hasattr(term_xds, "gains")
                for term_xds_list in solved_gain_xds_list
                for term_xds in term_xds_list])

# -----------------------------------------------------------------------------
