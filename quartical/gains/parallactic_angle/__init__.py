import numpy as np
from collections import namedtuple
from quartical.gains.conversion import trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.parallactic_angle.kernel import (
    parallactic_angle_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)
from quartical.data_handling.angles import _make_parangles

# Overload the default measurement set inputs to include information required
# to compute the parallactic angles.
ms_inputs = namedtuple(
    'ms_inputs', 
    ParameterizedGain.ms_inputs._fields + \
        ('RECEPTOR_ANGLE', 'POSITION')
)

class ParallacticAngle(ParameterizedGain):

    solver = None
    ms_inputs = ms_inputs

    native_to_converted = (
        (0, (np.cos,)),
        (1, (np.sin,))
    )
    converted_to_native = (
        (2, trig_to_angle),
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
        return ["parallactic_angle"]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        params[...] = _make_parangles(
            ms_kwargs["TIME"],
            np.arange(gains.shape[2]),  # TODO: Just for determining n_ant.
            ms_kwargs["POSITION"],
            ms_kwargs["RECEPTOR_ANGLE"],
            meta["FIELD_CENTRE"],
            "J2000"  # NOTE: Currently hardcoded - will need to be exposed.
        )[:, None, :, None, :1]  # Assume single value for parallactic angle.

        # Convert the parameters into gains.
        parallactic_angle_params_to_gains(
            params,
            gains,
            term_kwargs[f"{self.name}_param_freq_map"],
            feed_type=meta["FEED_TYPE"],
            corr_mode=gains.shape[-1]
        )

        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 1)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
