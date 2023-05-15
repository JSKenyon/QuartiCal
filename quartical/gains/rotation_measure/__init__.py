import numpy as np
from quartical.gains.conversion import no_op
from quartical.gains.gain import ParameterizedGain
from quartical.gains.rotation_measure.kernel import (
    rm_solver,
    rm_args,
    rm_params_to_gains
)


class RotationMeasure(ParameterizedGain):

    solver = staticmethod(rm_solver)
    term_args = rm_args

    native_to_converted = (
        (1, (no_op,)),
    )
    converted_to_native = (
        (1, no_op),
    )
    converted_dtype = np.float64
    native_dtype = np.float64

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def _make_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        # Overload gain mapping construction - we evaluate it in every channel.
        return np.arange(chan_freqs.size, dtype=np.int32)

    @classmethod
    def make_param_names(cls, correlations):

        return ["rotation_measure"]

    @staticmethod
    def make_f_maps(chan_freqs, chan_widths, f_int):
        """Internals of the frequency interval mapper."""

        n_chan = chan_freqs.size

        # The leading dimension corresponds (gain, param). For unparameterised
        # gains, the parameter mapping is irrelevant.
        f_map_arr = np.zeros((2, n_chan,), dtype=np.int32)

        if isinstance(f_int, float):
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(chan_widths):
                f_map_arr[1, i] = bin_num
                net_ivl += ivl
                if net_ivl >= f_int:
                    net_ivl = 0
                    bin_num += 1
        else:
            f_map_arr[1, :] = np.arange(n_chan)//f_int

        f_map_arr[0, :] = np.arange(n_chan)

        return f_map_arr

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, params = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        chan_freq = ms_kwargs["CHAN_FREQ"]
        lambda_sq = (299792458/chan_freq)**2

        # Convert the parameters into gains.
        rm_params_to_gains(
            params,
            gains,
            lambda_sq,
            term_kwargs[f"{self.name}-param-freq-map"],
        )

        return gains, params
