import numpy as np
from collections import namedtuple
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.delay_and_offset.kernel import (
    delay_and_offset_solver,
    delay_and_offset_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)

# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + ('CHAN_FREQ',)
)


class DelayAndOffset(ParameterizedGain):

    solver = staticmethod(delay_and_offset_solver)
    ms_inputs = ms_inputs

    native_to_converted = (
        (0, (np.cos,)),
        (1, (np.sin,)),
        (1, (no_op,))
    )
    converted_to_native = (
        (2, trig_to_angle),
        (1, no_op)
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

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        template = ("phase_offset_{}", "delay_{}")

        return [n.format(c) for c in param_corr for n in template]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        delay_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        if self.load_from or not self.initial_estimate:

            apply_param_flags_to_params(param_flags, params, 0)
            apply_gain_flags_to_gains(gain_flags, gains)

            return gains, gain_flags, params, param_flags

        data = ms_kwargs["DATA"]  # (row, chan, corr)
        flags = ms_kwargs["FLAG"]  # (row, chan)
        a1 = ms_kwargs["ANTENNA1"]
        a2 = ms_kwargs["ANTENNA2"]
        chan_freq = ms_kwargs["CHAN_FREQ"]
        t_map = term_kwargs[f"{term_spec.name}_time_map"]
        f_map = term_kwargs[f"{term_spec.name}_param_freq_map"]
        _, n_chan, n_ant, n_dir, n_corr = gains.shape

        # We only need the baselines which include the ref_ant.
        sel = np.where((a1 == ref_ant) | (a2 == ref_ant))
        a1 = a1[sel]
        a2 = a2[sel]
        t_map = t_map[sel]
        data = data[sel]
        flags = flags[sel]

        data[flags == 1] = 0  # Ignore UV-cut, otherwise there may be no est.

        utint = np.unique(t_map)
        ufint = np.unique(f_map)

        for ut in utint:
            sel = np.where((t_map == ut) & (a1 != a2))
            ant_map_pq = np.where(a1[sel] == ref_ant, a2[sel], 0)
            ant_map_qp = np.where(a2[sel] == ref_ant, a1[sel], 0)
            ant_map = ant_map_pq + ant_map_qp

            ref_data = np.zeros((n_ant, n_chan, n_corr), dtype=np.complex128)
            counts = np.zeros((n_ant, n_chan), dtype=int)
            np.add.at(
                ref_data,
                ant_map,
                data[sel]
            )
            np.add.at(
                counts,
                ant_map,
                flags[sel] == 0
            )
            np.divide(
                ref_data,
                counts[:, :, None],
                where=counts[:, :, None] != 0,
                out=ref_data
            )

            for uf in ufint:

                fsel = np.where(f_map == uf)[0]
                sel_n_chan = fsel.size
                n = int(np.ceil(2 ** 15 / sel_n_chan)) * sel_n_chan

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))

                fft_data = np.abs(
                    np.fft.fft(fsel_data, n=n, axis=1)
                )
                fft_data = np.fft.fftshift(fft_data, axes=1)

                delta_freq = chan_freq[1] - chan_freq[0]
                fft_freq = np.fft.fftfreq(n, delta_freq)
                fft_freq = np.fft.fftshift(fft_freq)

                delay_est_ind_00 = np.argmax(fft_data[..., 0], axis=1)
                delay_est_00 = fft_freq[delay_est_ind_00]
                delay_est_00[~valid_ant] = 0


                # path00 = "/home/russeeawon/testing/lofar_ms_1gc_solve/"
                # path00 = "/home/russeeawon/testing/lofar_ms_1gc_solve_solint1/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt1/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt2/"
                path00 = "/home/russeeawon/testing/thesis_figures/expt3/"

                # path01 = "1gc_solve_inparts/"
                # path01 = "1gc_solve_gtkb_parts/"
                # path01 = "1gc_solve_gkb/"
                # path01 = "1gc_solve_gktb_kest/"

                # path01 = "1gc_solve_gktb_bfulljones/"
                path01 = ""


                path0 = path00+path01
                

                np.save(path0+"delayest00.npy", delay_est_00)
                np.save(path0+"delay_fftarr.npy", fft_data)
                np.save(path0+"delay_fft_freq.npy", fft_freq)


                if n_corr > 1:
                    delay_est_ind_11 = np.argmax(fft_data[..., -1], axis=1)
                    delay_est_11 = fft_freq[delay_est_ind_11]
                    delay_est_11[~valid_ant] = 0
                    np.save(path0+"delayest11.npy", delay_est_11)
                

                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        params[t, uf, q, 0, 1] = -delay_est_00[q]
                        if n_corr > 1:
                            params[t, uf, q, 0, 3] = -delay_est_11[q]
                    else:
                        params[t, uf, p, 0, 1] = delay_est_00[p]
                        if n_corr > 1:
                            params[t, uf, p, 0, 3] = delay_est_11[p]

        delay_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
