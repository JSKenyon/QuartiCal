from collections import namedtuple
import numpy as np
import dask.array as da
from quartical.gains.gain import Gain
from quartical.gains.general.flagging import init_flags

ms_inputs = namedtuple(
    "ms_inputs",
    (
        "MODEL_DATA",
        "DATA",
        "ANTENNA1",
        "ANTENNA2",
        "WEIGHT",
        "FLAG",
        "ROW_MAP",
        "ROW_WEIGHTS",
        "TIME"
    )
)

mapping_inputs = namedtuple(
    "mapping_inputs",
    (
        "time_bins",
        "time_maps",
        "freq_maps",
        "dir_maps",
        "param_time_bins",
        "param_time_maps",
        "param_freq_maps"
    )
)


chain_inputs = namedtuple(
    "chain_inputs",
    (
        "gains",
        "gain_flags",
        "params",
        "param_flags"
    )
)


class ParameterizedGain(Gain):

    ms_inputs = ms_inputs
    mapping_inputs = mapping_inputs
    chain_inputs = chain_inputs

    is_parameterized = True

    param_axes = (
        "param_time",
        "param_freq",
        "antenna",
        "direction",
        "param_name"
    )
    interpolation_targets = ["params", "param_flags"]

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def make_param_time_bins(
        cls,
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries,
        chunks=None
    ):

        time_bins = da.map_blocks(
            cls._make_param_time_bins,
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries,
            dtype=np.int64,
            chunks=chunks
        )

        return time_bins

    @classmethod
    def _make_param_time_bins(
        cls,
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):
        return super()._make_time_bins(
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries
        )

    @classmethod
    def make_param_time_map(cls, time_col, param_time_bins):

        param_time_map = da.map_blocks(
            cls._make_param_time_map,
            time_col,
            param_time_bins,
            dtype=np.int64
        )

        return param_time_map

    @classmethod
    def _make_param_time_map(cls, time_col, param_time_bins):
        return super()._make_time_map(time_col, param_time_bins)

    @classmethod
    def make_param_time_chunks(cls, param_time_bins):

        param_time_chunks = da.map_blocks(
            cls._make_param_time_chunks,
            param_time_bins,
            chunks=(1,),
            dtype=np.int64
        )

        return param_time_chunks

    @classmethod
    def _make_param_time_chunks(cls, param_time_bins):
        return super()._make_time_chunks(param_time_bins)

    @classmethod
    def make_param_freq_map(cls, chan_freqs, chan_widths, freq_interval):

        param_freq_map = da.map_blocks(
            cls._make_param_freq_map,
            chan_freqs,
            chan_widths,
            freq_interval,
            dtype=np.int64
        )

        return param_freq_map

    @classmethod
    def _make_param_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        return super()._make_freq_map(chan_freqs, chan_widths, freq_interval)

    @classmethod
    def make_param_freq_chunks(cls, param_freq_map):

        param_freq_chunks = da.map_blocks(
            cls._make_param_freq_chunks,
            param_freq_map,
            chunks=(1,),
            dtype=np.int64
        )

        return param_freq_chunks

    @classmethod
    def _make_param_freq_chunks(cls, param_freq_map):
        return super()._make_freq_chunks(param_freq_map)

    @classmethod
    def make_param_names(cls, correlations):
        raise NotImplementedError

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        (_, _, gain_shape, param_shape) = term_spec

        if self.load_from:
            params = term_kwargs[f"{self.name}_initial_params"].copy()
        else:
            params = np.zeros(param_shape, dtype=np.float64)

        # Init parameter flags by looking for intervals with no data.
        param_flags = init_flags(
            param_shape,
            term_kwargs[f"{self.name}_param_time_map"],
            term_kwargs[f"{self.name}_param_freq_map"],
            ms_kwargs["FLAG"],
            ms_kwargs["ANTENNA1"],
            ms_kwargs["ANTENNA2"],
            ms_kwargs["ROW_MAP"]
        )

        gains = np.ones(gain_shape, dtype=np.complex128)
        if gain_shape[-1] == 4:
            gains[..., (1, 2)] = 0  # 2-by-2 identity.

        # Init gain flags by looking for intervals with no data.
        gain_flags = init_flags(
            gain_shape,
            term_kwargs[f"{self.name}_time_map"],
            term_kwargs[f"{self.name}_freq_map"],
            ms_kwargs["FLAG"],
            ms_kwargs["ANTENNA1"],
            ms_kwargs["ANTENNA2"],
            ms_kwargs["ROW_MAP"]
        )

        return gains, gain_flags, params, param_flags
