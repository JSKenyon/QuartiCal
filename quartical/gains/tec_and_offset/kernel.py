# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.generics import (native_intermediaries,
                                              upsampled_itermediaries,
                                              per_array_jhj_jhr,
                                              resample_solints,
                                              downsample_jhj_jhr)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags_to_flag_col,
                                              update_param_flags,
                                              apply_gain_flags_to_gains,
                                              apply_param_flags_to_params)
from quartical.gains.general.convenience import (get_row,
                                                 get_extents)
import quartical.gains.general.factories as factories
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)



def get_identity_params(corr_mode):

    if corr_mode.literal_value in (2, 4):
        return np.zeros((4,), dtype=np.float64)
    elif corr_mode.literal_value == 1:
        return np.zeros((2,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def tec_and_offset_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return tec_and_offset_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def tec_and_offset_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(tec_and_offset_solver_impl, jit_options=JIT_OPTIONS)
def nb_tec_and_offset_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    # NOTE: This just reuses delay solver functionality.

    coerce_literal(nb_tec_and_offset_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    ):

        gains = chain_inputs.gains
        gain_flags = chain_inputs.gain_flags

        active_term = meta_inputs.active_term
        max_iter = meta_inputs.iters
        solve_per = meta_inputs.solve_per
        dd_term = meta_inputs.dd_term
        n_thread = meta_inputs.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = chain_inputs.params[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_p = mapping_inputs.param_freq_maps[active_term]

        # Create more work to do in paralllel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, param_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_p)

        upsample_shape = resampler.upsample_shape
        upsampled_jhj = np.empty(upsample_shape + (upsample_shape[-1],),
                                 dtype=real_dtype)
        upsampled_jhr = np.empty(upsample_shape, dtype=real_dtype)
        jhj = upsampled_jhj[:param_shape[0]]
        jhr = upsampled_jhr[:param_shape[0]]
        update = np.zeros(param_shape, dtype=real_dtype)

        upsampled_imdry = upsampled_itermediaries(upsampled_jhj, upsampled_jhr)
        native_imdry = native_intermediaries(jhj, jhr, update)

        # We actually solve for TEC' = TEC/bandwidth. This helps avoid
        # numerical issues, but requires some scaling of the parameters.
        chan_freq = ms_inputs.CHAN_FREQ
        bandwidth = chan_freq.max() - chan_freq.min()
        active_params[..., 1::2] /= bandwidth

        for loop_idx in range(max_iter or 1):

            compute_jhj_jhr(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                upsampled_imdry,
                extents,
                corr_mode
            )

            if resampler.active:
                downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)

            if solve_per == "array":
                per_array_jhj_jhr(native_imdry)

            if not max_iter:  # Non-solvable term, we just want jhj.
                conv_perc = 0  # Didn't converge.
                loop_idx = -1  # Did zero iterations.
                break

            compute_update(native_imdry, corr_mode)

            finalize_update(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                native_imdry,
                loop_idx,
                corr_mode
            )

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(
                chain_inputs,
                meta_inputs,
                flag_imdry,
                loop_idx,
                corr_mode,
                numbness=1e9
            )

            # Propagate gain flags to parameter flags.
            update_param_flags(
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                identity_params
            )

            if conv_perc >= meta_inputs.stop_frac:
                break

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(
            chain_inputs,
            meta_inputs,
            flag_imdry,
            corr_mode
        )

        # reference_params(
        #     ms_inputs,
        #     mapping_inputs,
        #     chain_inputs,
        #     meta_inputs
        # )

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags_to_flag_col(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs
            )

        # Undo rescaling so that quantities are in native units.
        active_params[..., 1::2] *= bandwidth
        native_imdry.jhj[..., 1::2, 1::2] /= bandwidth ** 2

        return native_imdry.jhj, loop_idx + 1, conv_perc

    return impl


def compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):
    return NotImplementedError


@overload(compute_jhj_jhr, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):

    coerce_literal(nb_compute_jhj_jhr, ["corr_mode"])

    # We want to dispatch based on this field so we need its type.
    row_weights_idx = ms_inputs.fields.index('ROW_WEIGHTS')
    row_weights_type = ms_inputs[row_weights_idx]

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights_type)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    absv1_idiv_absv2 = factories.absv1_idiv_absv2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    imul = factories.imul_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    isub = factories.isub_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        upsampled_imdry,
        extents,
        corr_mode
    ):

        active_term = meta_inputs.active_term

        data = ms_inputs.DATA
        model = ms_inputs.MODEL_DATA
        weights = ms_inputs.WEIGHT
        flags = ms_inputs.FLAG
        antenna1 = ms_inputs.ANTENNA1
        antenna2 = ms_inputs.ANTENNA2
        row_map = ms_inputs.ROW_MAP
        row_weights = ms_inputs.ROW_WEIGHTS
        chan_freq = ms_inputs.CHAN_FREQ

        # NOTE: These are the gain maps.
        time_maps = mapping_inputs.time_maps
        freq_maps = mapping_inputs.freq_maps
        dir_maps = mapping_inputs.dir_maps

        gains = chain_inputs.gains

        jhj = upsampled_imdry.jhj
        jhr = upsampled_imdry.jhr

        row_starts = extents.row_starts
        row_stops = extents.row_stops
        chan_starts = extents.chan_starts
        chan_stops = extents.chan_stops

        n_row, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_param = jhr.shape
        n_int = n_tint*n_fint

        complex_dtype = gains[active_term].dtype
        weight_dtype = weights.dtype

        n_gains = len(gains)

        cf_min = chan_freq.min()
        cf_max = chan_freq.max()
        bandwidth = cf_max - cf_min
        offset = np.log(cf_min / cf_max)

        # Determine loop variables based on where we are in the chain.
        # gt means greater than (n>j) and lt means less than (n<j).
        all_terms, gt_active, lt_active = make_loop_vars(n_gains, active_term)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            rop_pq = valloc(complex_dtype)  # Right-multiply operator for pq.
            rop_qp = valloc(complex_dtype)  # Right-multiply operator for qp.
            lop_pq = valloc(complex_dtype)  # Left-multiply operator for pq.
            lop_qp = valloc(complex_dtype)  # Left-multiply operator for qp.

            w = valloc(weight_dtype)
            r_pq = valloc(complex_dtype)
            wr_pq = valloc(complex_dtype)
            wr_qp = valloc(complex_dtype)
            v_pqd = valloc(complex_dtype)
            v_pq = valloc(complex_dtype)

            gains_p = valloc(complex_dtype, leading_dims=(n_gains,))
            gains_q = valloc(complex_dtype, leading_dims=(n_gains,))

            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            norm_factors = valloc(complex_dtype)

            jhr_tifi = jhr[ti, fi]
            jhj_tifi = jhj[ti, fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = antenna1[row], antenna2[row]

                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    # Apply row weights in the BDA case, otherwise a no-op.
                    imul_rweight(weights[row, f], w, row_weights, row_ind)
                    iunpack(r_pq, data[row, f])

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0
                    v_pq[:] = 0

                    for d in range(n_dir):

                        set_identity(lop_pq)
                        set_identity(lop_qp)

                        # Construct a small contiguous gain array. This makes
                        # the single term case fractionally slower.
                        for gi in range(n_gains):
                            d_m = dir_maps[gi][d]  # Broadcast dir.
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            iunpack(gains_p[gi], gain[a1_m, d_m])
                            iunpack(gains_q[gi], gain[a2_m, d_m])

                        m = model[row, f, d]
                        iunpack(rop_qp, m)
                        iunpackct(rop_pq, rop_qp)

                        for g in all_terms:

                            g_q = gains_q[g]
                            v1_imul_v2(g_q, rop_pq, rop_pq)

                            g_p = gains_p[g]
                            v1_imul_v2(g_p, rop_qp, rop_qp)

                        for g in gt_active:

                            g_p = gains_p[g]
                            v1_imul_v2ct(rop_pq, g_p, rop_pq)

                            g_q = gains_q[g]
                            v1_imul_v2ct(rop_qp, g_q, rop_qp)

                        for g in lt_active:

                            g_p = gains_p[g]
                            v1ct_imul_v2(g_p, lop_pq, lop_pq)

                            g_q = gains_q[g]
                            v1ct_imul_v2(g_q, lop_qp, lop_qp)

                        out_d = dir_maps[active_term][d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                        v1ct_imul_v2(lop_pq, gains_p[active_term], v_pqd)
                        v1_imul_v2ct(v_pqd, rop_pq, v_pqd)
                        iadd(v_pq, v_pqd)

                    absv1_idiv_absv2(v_pq, r_pq, norm_factors)
                    imul(r_pq, norm_factors)
                    isub(r_pq, v_pq)

                    # Coefficient introduced by differentiation of exponent.
                    coeff = 2 * np.pi * (bandwidth/chan_freq[f] + offset)

                    for d in range(n_gdir):

                        iunpack(wr_pq, r_pq)
                        imul(wr_pq, w)
                        iunpackct(wr_qp, wr_pq)

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj_jhwr_elem(lop_pq_d,
                                               rop_pq_d,
                                               w,
                                               norm_factors,
                                               coeff,
                                               gains_p[active_term],
                                               wr_pq,
                                               jhr_tifi[a1_m, d],
                                               jhj_tifi[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj_jhwr_elem(lop_qp_d,
                                               rop_qp_d,
                                               w,
                                               norm_factors,
                                               coeff,
                                               gains_q[active_term],
                                               wr_qp,
                                               jhr_tifi[a2_m, d],
                                               jhj_tifi[a2_m, d])
        return
    return impl


def compute_update(native_imdry, corr_mode):
    raise NotImplementedError


@overload(compute_update, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_update(native_imdry, corr_mode):

    coerce_literal(nb_compute_update, ["corr_mode"])

    # We want to dispatch based on this field so we need its type.
    jhj = native_imdry[native_imdry.fields.index('jhj')]

    generalised = jhj.ndim == 6
    inversion_buffer = inversion_buffer_factory(generalised=generalised)
    invert = invert_factory(corr_mode, generalised=generalised)

    def impl(native_imdry, corr_mode):

        jhj = native_imdry.jhj
        jhr = native_imdry.jhr
        update = native_imdry.update

        n_tint, n_fint, n_ant, n_dir, n_param = jhr.shape

        n_int = n_tint * n_fint

        result_dtype = jhr.dtype

        for i in prange(n_int):

            t = i // n_fint
            f = i - t * n_fint

            buffers = inversion_buffer(n_param, result_dtype)

            for a in range(n_ant):
                for d in range(n_dir):

                    invert(jhj[t, f, a, d],
                           jhr[t, f, a, d],
                           update[t, f, a, d],
                           buffers)

    return impl


def finalize_update(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):
    raise NotImplementedError


@overload(finalize_update, jit_options=JIT_OPTIONS)
def nb_finalize_update(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):

    coerce_literal(nb_finalize_update, ["corr_mode"])

    set_identity = factories.set_identity_factory(corr_mode)
    param_to_gain = param_to_gain_factory(corr_mode)

    if corr_mode.literal_value in (1, 2, 4):
        def impl(
            ms_inputs,
            mapping_inputs,
            chain_inputs,
            meta_inputs,
            native_imdry,
            loop_idx,
            corr_mode
        ):

            dd_term = meta_inputs.dd_term
            active_term = meta_inputs.active_term
            pinned_directions = meta_inputs.pinned_directions

            gain = chain_inputs.gains[active_term]
            gain_flags = chain_inputs.gain_flags[active_term]
            params = chain_inputs.params[active_term]

            param_freq_map = mapping_inputs.param_freq_maps[active_term]
            dir_map = mapping_inputs.dir_maps[active_term]

            update = native_imdry.update

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            if dd_term:
                for pd in pinned_directions:
                    params[..., pd, :] = 0

            chan_freq = ms_inputs.CHAN_FREQ
            cf_min = chan_freq.min()
            cf_max = chan_freq.max()
            bandwidth = cf_max - cf_min
            offset = np.log(cf_min/cf_max)

            for t in range(n_time):
                for f in range(n_freq):
                    f_m = param_freq_map[f]
                    coeff = 2 * np.pi * (bandwidth / chan_freq[f] + offset)
                    for a in range(n_ant):
                        for d in range(n_dir):

                            d_m = dir_map[d]
                            g = gain[t, f, a, d]
                            fl = gain_flags[t, f, a, d]
                            p = params[t, f_m, a, d_m]

                            if fl == 1:
                                set_identity(g)
                            else:
                                param_to_gain(p, coeff, g)
    else:
        raise ValueError("Unsupported number of correlations.")

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, coeff, gain):
            gain[0] = np.exp(1j * (coeff * params[1] + params[0]))
            gain[3] = np.exp(1j * (coeff * params[3] + params[2]))
    elif corr_mode.literal_value == 2:
        def impl(params, coeff, gain):
            gain[0] = np.exp(1j * (coeff * params[1] + params[0]))
            gain[1] = np.exp(1j * (coeff * params[3] + params[2]))
    elif corr_mode.literal_value == 1:
        def impl(params, coeff, gain):
            gain[0] = np.exp(1j * (coeff * params[1] + params[0]))
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def compute_jhwj_jhwr_elem_factory(corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)
    unpackc = factories.unpackc_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iabsdivsq = factories.iabsdivsq_factory(corr_mode)
    imul = factories.imul_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, normf, coeff, gain, res, jhr, jhj):

            # Effectively apply zero weight to off-diagonal terms.
            # TODO: Can be tidied but requires moving other weighting code.
            res[1] = 0
            res[2] = 0

            # Compute normalization factor.
            v1_imul_v2(lop, rop, normf)
            iabsdivsq(normf)
            imul(res, normf)  # Apply normalization factor to r.

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)
            v1_imul_v2(lop, res, res)

            # Accumulate an element of jhwj.

            r_0, _, _, r_3 = unpack(res)  # NOTE: XX, XY, YX, YY

            _, _, _, g_3 = unpack(gain)
            gc_0, _, _, gc_3 = unpackc(gain)

            drv_00 = -1j*gc_0
            drv_23 = -1j*gc_3

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_23*r_3).real

            jhr[0] += upd_00
            jhr[1] += coeff*upd_00
            jhr[2] += upd_11
            jhr[3] += coeff*upd_11

            w_0, _, _, w_3 = unpack(w)  # NOTE: XX, XY, YX, YY
            n_0, _, _, n_3 = unpack(normf)

            # Apply normalisation factors by scaling w. # Neglect (set weight
            # to zero) off diagonal terms.
            w_0 = n_0 * w_0
            w_3 = n_3 * w_3

            lop_00, lop_01, lop_10, lop_11 = unpack(lop)
            rop_00, rop_10, rop_01, rop_11 = unpack(rop)  # "Transpose"

            jh_00 = lop_00 * rop_00
            jh_03 = lop_01 * rop_01

            j_00 = jh_00.conjugate()
            j_03 = jh_03.conjugate()

            jh_30 = lop_10 * rop_10
            jh_33 = lop_11 * rop_11

            j_30 = jh_30.conjugate()
            j_33 = jh_33.conjugate()

            jhwj_00 = jh_00*w_0*j_00 + jh_03*w_3*j_03
            jhwj_03 = jh_00*w_0*j_30 + jh_03*w_3*j_33
            jhwj_33 = jh_30*w_0*j_30 + jh_33*w_3*j_33

            coeffsq = coeff ** 2

            tmp_0 = jhwj_00.real
            jhj[0, 0] += tmp_0
            jhj[0, 1] += tmp_0*coeff
            tmp_1 = (jhwj_03*gc_0*g_3).real
            jhj[0, 2] += tmp_1
            jhj[0, 3] += tmp_1*coeff

            jhj[1, 0] = jhj[0, 1]
            jhj[1, 1] += tmp_0*coeffsq
            jhj[1, 2] = jhj[0, 3]
            jhj[1, 3] += tmp_1*coeffsq

            jhj[2, 0] = jhj[0, 2]
            jhj[2, 1] = jhj[1, 2]
            tmp_2 = jhwj_33.real
            jhj[2, 2] += tmp_2
            jhj[2, 3] += tmp_2*coeff

            jhj[3, 0] = jhj[0, 3]
            jhj[3, 1] = jhj[1, 3]
            jhj[3, 2] = jhj[2, 3]
            jhj[3, 3] += tmp_2*coeffsq

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, normf, coeff, gain, res, jhr, jhj):

            # Compute normalization factor.
            iunpack(normf, rop)
            iabsdivsq(normf)
            imul(res, normf)  # Apply normalization factor to r.

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)

            r_0, r_1 = unpack(res)
            gc_0, gc_1 = unpackc(gain)

            drv_00 = -1j*gc_0
            drv_23 = -1j*gc_1

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_23*r_1).real

            jhr[0] += upd_00
            jhr[1] += coeff*upd_00
            jhr[2] += upd_11
            jhr[3] += coeff*upd_11

            # Accumulate an element of jhwj.
            jh_00, jh_11 = unpack(rop)
            j_00, j_11 = unpackc(rop)
            w_00, w_11 = unpack(w)
            n_00, n_11 = unpack(normf)

            coeffsq = coeff ** 2

            tmp = (jh_00*n_00*w_00*j_00).real
            jhj[0, 0] += tmp
            jhj[0, 1] += tmp*coeff
            jhj[1, 0] += tmp*coeff
            jhj[1, 1] += tmp*coeffsq

            tmp = (jh_11*n_11*w_11*j_11).real
            jhj[2, 2] += tmp
            jhj[2, 3] += tmp*coeff
            jhj[3, 2] += tmp*coeff
            jhj[3, 3] += tmp*coeffsq

    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, normf, coeff, gain, res, jhr, jhj):

            # Compute normalization factor.
            iunpack(normf, rop)
            iabsdivsq(normf)
            imul(res, normf)  # Apply normalization factor to r.

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)

            r_0 = unpack(res)
            gc_0 = unpackc(gain)

            drv_00 = -1j*gc_0

            upd_00 = (drv_00*r_0).real

            jhr[0] += upd_00
            jhr[1] += coeff*upd_00

            # Accumulate an element of jhwj.
            jh_00 = unpack(rop)
            j_00 = unpackc(rop)
            w_00 = unpack(w)
            n_00 = unpack(normf)

            coeffsq = coeff*coeff

            tmp = (jh_00*n_00*w_00*j_00).real
            jhj[0, 0] += tmp
            jhj[0, 1] += tmp*coeff
            jhj[1, 0] += tmp*coeff
            jhj[1, 1] += tmp*coeffsq
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


# @njit(**JIT_OPTIONS)
# def tec_and_offset_params_to_gains(
#     params,
#     gains,
#     chan_freq,
#     param_freq_map,
#     rescaled=False
# ):

#     n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

#     cf_min = chan_freq.min()
#     cf_max = chan_freq.max()
#     bandwidth = cf_max - cf_min

#     if rescaled:
#         offset = np.log(cf_min/cf_max)
#         numerator = bandwidth
#     else:
#         offset = np.log(cf_min/cf_max)/bandwidth
#         numerator = 1.0

#     for t in range(n_time):
#         for f in range(n_freq):
#             f_m = param_freq_map[f]
#             coeff = 2 * np.pi * (numerator / chan_freq[f] + offset)
#             for a in range(n_ant):
#                 for d in range(n_dir):

#                     g = gains[t, f, a, d]
#                     p = params[t, f_m, a, d]

#                     g[0] = np.exp(1j*(coeff*p[1] + p[0]))

#                     if n_corr > 1:
#                         g[-1] = np.exp(1j*(coeff*p[3] + p[2]))


@njit(**JIT_OPTIONS)
def tec_and_offset_params_to_gains(
    params,
    gains,
    sel_ant,
    chan_freq,
    param_freq_map,
    rescaled=False
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    cf_min = chan_freq.min()
    cf_max = chan_freq.max()
    cf_mid = (cf_min + cf_max) / 2
    bandwidth = cf_max - cf_min

    # if rescaled:
    #     offset = np.log(cf_min/cf_max)
    #     numerator = bandwidth
    # else:
    #     offset = np.log(cf_min/cf_max)/bandwidth
    #     numerator = 1.0

    for t in range(n_time):
        for f in range(n_freq):
            f_m = param_freq_map[f]
            for a in range(n_ant):
                #dominant peak is delay
                if sel_ant[a] in (1, 2):
                    if rescaled:
                        offset = 1.0
                        denom = cf_mid
                    else:
                        offset = cf_mid
                        denom = 1.0
                    coeff = 2 * np.pi * (chan_freq[f] / denom - offset)
                else:
                    #dominant peak is tec
                    if rescaled:
                        offset = np.log(cf_min/cf_max)
                        numerator = bandwidth
                    else:
                        offset = np.log(cf_min/cf_max)/bandwidth
                        numerator = 1.0
                    coeff = 2 * np.pi * (numerator / chan_freq[f] + offset)
            
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    p = params[t, f_m, a, d]

                    g[0] = np.exp(1j*(coeff*p[1] + p[0]))

                    if n_corr > 1:
                        g[-1] = np.exp(1j*(coeff*p[3] + p[2]))


@njit(**JIT_OPTIONS)
def reference_params(ms_inputs, mapping_inputs, chain_inputs, meta_inputs):

    chan_freq = ms_inputs.CHAN_FREQ

    active_term = meta_inputs.active_term
    ref_ant = meta_inputs.reference_antenna

    gains = chain_inputs.gains[active_term]
    gain_flags = chain_inputs.gain_flags[active_term]
    params = chain_inputs.params[active_term]
    param_flags = chain_inputs.param_flags[active_term]

    param_freq_map = mapping_inputs.param_freq_maps[active_term]

    n_ti, n_fi, n_ant, n_dir, n_corr = params.shape

    ref_params = params[:, :, ref_ant: ref_ant + 1, :, :].copy()

    for t in range(n_ti):
        for f in range(n_fi):
            for a in range(n_ant):
                for d in range(n_dir):

                    p = params[t, f, a, d]
                    rp = ref_params[t, f, 0, d]

                    if param_flags[t, f, a, d] == 1:
                        continue
                    else:
                        p -= rp

    tec_and_offset_params_to_gains(
        params, gains, chan_freq, param_freq_map, rescaled=True
    )

    # Referencing may move flagged gains/params from identity.
    apply_param_flags_to_params(param_flags, params, 0)
    apply_gain_flags_to_gains(gain_flags, gains)
