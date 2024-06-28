import numpy as np
import finufft
from collections import namedtuple
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.tec_and_offset.kernel import (
    tec_and_offset_solver,
    tec_and_offset_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)
from quartical.gains.general.generics import compute_corrected_residual


# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + ('CHAN_FREQ',)
)


class TecAndOffset(ParameterizedGain):

    solver = staticmethod(tec_and_offset_solver)
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

        template = ("phase_offset_{}", "TEC_{}")

        return [n.format(c) for c in param_corr for n in template]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )


        # Convert the parameters into gains.
        tec_and_offset_params_to_gains(
            params,
            gains,
            np.zeros((gains.shape[2])),
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

        #what about dir_maps?
        dir_maps = np.zeros(1, dtype=np.int32)

        #what about row_map and row_weights?
        row_map = ms_inputs.ROW_MAP
        row_weights = ms_inputs.ROW_WEIGHTS


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

                # fsel_data = visibilities corresponding to the fsel
                # frequencies
                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))
                ##in inverse frequency domain
                invfreq = 1./chan_freq

                if n_corr == 1:
                    n_param = 1
                elif n_corr in (2, 4):
                    n_param = 2
                else:
                    raise ValueError("Unsupported number of correlations.")


                #Obtain the delay-related peak
                delta_freqk = chan_freq[1] - chan_freq[0]
                #Maximum reconstructable delay (in time)
                max_delay = (2*np.pi)/delta_freqk
                nyq_rate0 = 1./(2*(chan_freq.max() - chan_freq.min()))
                
                nk = int(max_delay/nyq_rate0)
                print(nk, "this is nk")

                # nk = int(np.ceil(2 ** 15 / sel_n_chan)) * sel_n_chan
                # nk = int(np.ceil(2 ** 20 / sel_n_chan)) * sel_n_chan

                print(nk, "this is nk")

                # fft_datak = np.abs(
                #     np.fft.fft(fsel_data, n=nk, axis=1)
                # )
                # fft_datak = np.fft.fftshift(fft_datak, axes=1)

                fft_freqk = np.fft.fftfreq(nk, delta_freqk)
                fft_freqk = np.fft.fftshift(fft_freqk)

                delay_est = np.zeros((n_ant, n_param), dtype=np.float64)

                # delay_est_ind_00 = np.argmax(fft_datak[..., 0], axis=1)
                # delay_est_00 = fft_freqk[delay_est_ind_00]
                # delay_est_00[~valid_ant] = 0
                # delay_est[:, 0] = delay_est_00

                # if n_corr > 1:
                #     delay_est_ind_11 = np.argmax(fft_datak[..., -1], axis=1)
                #     delay_est_11 = fft_freqk[delay_est_ind_11]
                #     delay_est_11[~valid_ant] = 0
                #     delay_est[:, 1] = delay_est_11


                #Let me try using the nufft for the delay estimation as well
                fft_datak = np.zeros((n_ant, nk, n_param), dtype=fsel_data.dtype)

                for k in range(n_param):
                    vis_finufft = finufft.nufft1d3(
                        2 * np.pi * chan_freq,
                        fsel_data[:, :, k],
                        fft_freqk,
                        eps=1e-6,
                        isign=-1
                    )
                    fft_datak[:, :, k] = vis_finufft
                    fft_data_pk = np.abs(vis_finufft)
                    delay_est[:, k] = fft_freqk[np.argmax(fft_data_pk, axis=1)]

                delay_est[~valid_ant, :] = 0
                

                #Obtain the tec-related peak
                ##factor for rescaling frequency
                ffactor = 1 #1e8
                invfreq *= ffactor

                # delta_freq is the smallest difference between the frequency
                # values
                delta_freqt = invfreq[-2] - invfreq[-1] #frequency resolution
                max_tec = 2 * np.pi / delta_freqt
                nyq_rate = 1./(2*(invfreq.max() - invfreq.min()))
                sr = max_tec #sampling_rate

                # choosing resolution
                nt = int(max_tec/ nyq_rate)
                print(nt, "this is nt")

                # domain to pick tec_est from
                fft_freqt = np.linspace(0.5*-sr, 0.5*sr, nt)


                tec_est = np.zeros((n_ant, n_param), dtype=np.float64)
                fft_datat = np.zeros(
                    (n_ant, nt, n_param), dtype=fsel_data.dtype
                )

                for k in range(n_param):
                    vis_finufft = finufft.nufft1d3(
                        2 * np.pi * invfreq,
                        fsel_data[:, :, k],
                        fft_freqt,
                        eps=1e-6,
                        isign=-1
                    )
                    fft_datat[:, :, k] = vis_finufft
                    fft_data_pk = np.abs(vis_finufft)
                    tec_est[:, k] = fft_freqt[np.argmax(fft_data_pk, axis=1)]

                tec_est[~valid_ant, :] = 0

                #Checking for the dominant and sub-dominant peaks
                # Consider a single array consisting of a mix of "only" dominant \
                # peaks solved by each solver.
                param_est = np.zeros((n_ant, n_param))
                sel_ant = np.zeros((n_ant))

                for p in range(n_ant):
                    for k in range(n_param):
                        if p != ref_ant:
                            #if dominant peak is associated with delay K, assign 1.
                            if np.max(fft_datak[p, :, k]) > np.max(fft_datat[p, :, k]):
                                #In the case, the delay has been corrected, the param to be
                                #estimated next is tec.
                                if np.allclose(delay_est[p, k], 0., atol=1e-10):
                                    param_est[p, k] = tec_est[p, k]
                                else:
                                    param_est[p, k] = delay_est[p, k]
                                    sel_ant[p] = 1
                            elif np.max(fft_datak[p, :, k]) < np.max(fft_datat[p, :, k]):
                                if np.allclose(tec_est[p, k], 0., atol=1e-10):
                                    param_est[p, k] = delay_est[p, k]
                                    sel_ant[p] = 1
                                else:
                                    param_est[p, k] = tec_est[p, k]
                            # elif np.allclose(np.max(fft_datak[p, :, k]), np.max(fft_datat[p, :, k]), atol=1e-8):
                            #     #default to delay peaks if similar
                            #     param_est[p, k] = delay_est[p, k]
                            #     sel_ant[p] = 2
                            else:
                                raise ValueError("Unsupported functionality.")

                
                # path00 = "/home/russeeawon/testing/lofar_ms_1gc_solve/"
                # path00 = "/home/russeeawon/testing/lofar_ms_1gc_solve_solint1/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt4a/"
                path00 = "/home/russeeawon/testing/thesis_figures/expt4b/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt5a/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt6a/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt6b/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt7a/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt7b/"



                # path01 = "1gc_solve_inparts/"
                # path01 = "1gc_solve_gtkb_parts/"
                # path01 = "1gc_solve_gtb/"
                # path01 = "1gc_solve_inparts/"
                path01 = ""

                path0 = path00+path01
                np.save(path0+"delayest.npy", delay_est)
                np.save(path0+"delay_fftarr.npy", fft_datak)
                np.save(path0+"delay_fft_freq.npy", fft_freqk)

                np.save(path0+"tecest.npy", tec_est)
                np.save(path0+"tec_fftarr.npy", fft_datat)
                np.save(path0+"tec_fft_freq.npy", fft_freqt)

                np.save(path0+"sel_ant.npy", sel_ant)


            for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                if p == ref_ant:
                    params[t, uf, q, 0, 1] = -param_est[q, 0]
                    if n_corr > 1:
                        params[t, uf, q, 0, 3] = -param_est[q, 1]
                elif q == ref_ant:
                    params[t, uf, p, 0, 1] = param_est[p, 0]
                    if n_corr > 1:
                        params[t, uf, p, 0, 3] = param_est[p, 1]


        # Convert the parameters into gains using the "set" of dominant effects.
        tec_and_offset_params_to_gains(
            params,
            gains,
            sel_ant,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )


        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 1)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
