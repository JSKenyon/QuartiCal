# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (compute_residual_solver,
                                              per_array_jhj_jhr)
from quartical.gains.general.flagging import (update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags,
                                              gain_flags_to_param_flags)
from quartical.gains.general.convenience import (get_row,
                                                 get_chan_extents,
                                                 get_row_extents)
import quartical.gains.general.factories as factories
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

delay_args = namedtuple(
    "delay_args",
    (
        "params",
        "param_flags",
        "chan_freqs",
        "t_bin_arr"
    )
)


flag_intermediaries = namedtuple(
    "flag_intermediaries",
    (
        "km1_gain",
        "km1_abs2_diffs",
        "abs2_diffs_trend"
    )

)


solver_intermediaries = namedtuple(
    "solver_intermediaries",
    (
        "jhj",
        "jhr",
        "residual",
        "update"
    )
)


@generated_jit(nopython=True,
               fastmath=True,
               parallel=False,
               cache=True,
               nogil=True)
def delay_solver(base_args, term_args, meta_args, corr_mode):

    coerce_literal(delay_solver, ["corr_mode"])

    def impl(base_args, term_args, meta_args, corr_mode):

        data = base_args.data
        gains = base_args.gains
        gain_flags = base_args.gain_flags

        active_term = meta_args.active_term
        max_iter = meta_args.iters
        solve_per = meta_args.solve_per
        dd_term = meta_args.dd_term

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = term_args.params[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving. TODO: Move?
        real_dtype = active_gain.real.dtype
        pshape = active_params.shape
        jhj = np.empty(pshape + (pshape[-1],), dtype=real_dtype)
        jhr = np.empty(pshape, dtype=real_dtype)
        residual = np.empty_like(data) if dd_term else data
        update = np.zeros_like(jhr)
        solver_imdry = solver_intermediaries(jhj, jhr, residual, update)

        scaled_cf = term_args.chan_freqs.copy()  # Don't mutate.
        min_freq = np.min(scaled_cf)
        scaled_cf /= min_freq  # Scale freqs to avoid precision.
        active_params[..., 1::2] *= min_freq  # Scale delay consistently.

        for loop_idx in range(max_iter):

            if dd_term or len(gains) > 1:
                compute_residual_solver(base_args,
                                        solver_imdry,
                                        corr_mode)

            compute_jhj_jhr(base_args,
                            term_args,
                            meta_args,
                            solver_imdry,
                            scaled_cf,
                            corr_mode)

            if solve_per == "array":
                per_array_jhj_jhr(solver_imdry)

            compute_update(solver_imdry,
                           corr_mode)

            finalize_update(base_args,
                            term_args,
                            meta_args,
                            solver_imdry,
                            scaled_cf,
                            loop_idx,
                            corr_mode)

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(base_args,
                                          term_args,
                                          meta_args,
                                          flag_imdry,
                                          loop_idx,
                                          corr_mode)

            if conv_perc > meta_args.stop_frac:
                break

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(base_args,
                            meta_args,
                            flag_imdry,
                            corr_mode)

        # Propagate gain flags to parameter flags. TODO: Verify that this
        # is adequate. Do we need to consider setting the identity params.
        gain_flags_to_param_flags(base_args,
                                  term_args,
                                  meta_args)

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags(base_args,
                             meta_args)

        active_params[..., 1::2] /= min_freq  # Undo scaling for SI units.

        return jhj, term_conv_info(loop_idx + 1, conv_perc)

    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_jhj_jhr(base_args, term_args, meta_args, solver_imdry, scaled_cf,
                    corr_mode):

    # We want to dispatch based on this field so we need its type.
    row_weights = base_args[base_args.fields.index('row_weights')]

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    imul = factories.imul_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)

    def impl(base_args, term_args, meta_args, solver_imdry, scaled_cf,
             corr_mode):

        active_term = meta_args.active_term

        model = base_args.model
        weights = base_args.weights
        flags = base_args.flags
        a1 = base_args.a1
        a2 = base_args.a2
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        gains = base_args.gains
        t_map_arr = base_args.t_map_arr[0]  # We only need the gain mappings.
        f_map_arr_g = base_args.f_map_arr[0]
        f_map_arr_p = base_args.f_map_arr[1]
        d_map_arr = base_args.d_map_arr

        jhj = solver_imdry.jhj
        jhr = solver_imdry.jhr
        residual = solver_imdry.residual

        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_param = jhr.shape
        n_int = n_tint*n_fint

        complex_dtype = gains[active_term].dtype

        n_gains = len(gains)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval. This could even be moved out for speed.
        row_starts, row_stops = get_row_extents(t_map_arr,
                                                active_term,
                                                n_tint)

        chan_starts, chan_stops = get_chan_extents(f_map_arr_p,
                                                   active_term,
                                                   n_fint,
                                                   n_chan)

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
            r_pq = valloc(complex_dtype)
            r_qp = valloc(complex_dtype)

            gains_p = valloc(complex_dtype, leading_dims=(n_gains,))
            gains_q = valloc(complex_dtype, leading_dims=(n_gains,))

            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            tmp_kprod = np.zeros((4, 4), dtype=complex_dtype)
            tmp_jhr = jhr[ti, fi]
            tmp_jhj = jhj[ti, fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = a1[row], a2[row]

                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    r = residual[row, f]
                    w = weights[row, f]  # Consider a map?

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0

                    for d in range(n_dir):

                        set_identity(lop_pq)
                        set_identity(lop_qp)

                        # Construct a small contiguous gain array. This makes
                        # the single term case fractionally slower.
                        for gi in range(n_gains):
                            d_m = d_map_arr[gi, d]  # Broadcast dir.
                            t_m = t_map_arr[row_ind, gi]
                            f_m = f_map_arr_g[f, gi]

                            gain = gains[gi][t_m, f_m]

                            iunpack(gains_p[gi], gain[a1_m, d_m])
                            iunpack(gains_q[gi], gain[a2_m, d_m])

                        m = model[row, f, d]
                        imul_rweight(m, rop_qp, row_weights, row_ind)
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

                        out_d = d_map_arr[active_term, d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                    nu = scaled_cf[f]

                    for d in range(n_gdir):

                        imul_rweight(r, r_pq, row_weights, row_ind)
                        imul(r_pq, w)  # Check: Not needed as we compute jhwr.
                        iunpackct(r_qp, r_pq)

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj_jhwr_elem(lop_pq_d,
                                               rop_pq_d,
                                               w,
                                               nu,
                                               gains_p[active_term],
                                               tmp_kprod,
                                               r_pq,
                                               tmp_jhr[a1_m, d],
                                               tmp_jhj[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj_jhwr_elem(lop_qp_d,
                                               rop_qp_d,
                                               w,
                                               nu,
                                               gains_q[active_term],
                                               tmp_kprod,
                                               r_qp,
                                               tmp_jhr[a2_m, d],
                                               tmp_jhj[a2_m, d])
        return
    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_update(solver_imdry, corr_mode):

    # We want to dispatch based on this field so we need its type.
    jhj = solver_imdry[solver_imdry.fields.index('jhj')]

    generalised = jhj.ndim == 6
    inversion_buffer = inversion_buffer_factory(generalised=generalised)
    invert = invert_factory(corr_mode, generalised=generalised)

    def impl(solver_imdry, corr_mode):

        jhj = solver_imdry.jhj
        jhr = solver_imdry.jhr
        update = solver_imdry.update

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


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_update(base_args, term_args, meta_args, solver_imdry, scaled_cf,
                    loop_idx, corr_mode):

    set_identity = factories.set_identity_factory(corr_mode)
    param_to_gain = param_to_gain_factory(corr_mode)

    if corr_mode.literal_value in (1, 2, 4):
        def impl(base_args, term_args, meta_args, solver_imdry, scaled_cf,
                 loop_idx, corr_mode):

            active_term = meta_args.active_term

            gain = base_args.gains[active_term]
            gain_flags = base_args.gain_flags[active_term]
            f_map_arr_p = base_args.f_map_arr[1, :, active_term]
            d_map_arr = base_args.d_map_arr[active_term, :]

            params = term_args.params[active_term]

            update = solver_imdry.update

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            for t in range(n_time):
                for f in range(n_freq):
                    cf = scaled_cf[f]
                    f_m = f_map_arr_p[f]
                    for a in range(n_ant):
                        for d in range(n_dir):

                            d_m = d_map_arr[d]
                            g = gain[t, f, a, d]
                            fl = gain_flags[t, f, a, d]
                            p = params[t, f_m, a, d_m]

                            if fl == 1:
                                set_identity(g)
                            else:
                                param_to_gain(p, cf, g)
    else:
        raise ValueError("Unsupported number of correlations.")

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, chanfreq, gain):
            gain[0] = np.exp(1j*(chanfreq*params[1] + params[0]))
            gain[3] = np.exp(1j*(chanfreq*params[3] + params[2]))
    elif corr_mode.literal_value == 2:
        def impl(params, chanfreq, gain):
            gain[0] = np.exp(1j*(chanfreq*params[1] + params[0]))
            gain[1] = np.exp(1j*(chanfreq*params[3] + params[2]))
    elif corr_mode.literal_value == 1:
        def impl(params, chanfreq, gain):
            gain[0] = np.exp(1j*(chanfreq*params[1] + params[0]))
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def compute_jhwj_jhwr_elem_factory(corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    a_kron_bt = factories.a_kron_bt_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)
    unpackc = factories.unpackc_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, nu, gain, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)
            v1_imul_v2(lop, res, res)

            # Accumulate an element of jhwj.

            # WARNING: In this instance we are using the row-major
            # version of the kronecker product identity. This is because the
            # MS stores the correlations in row-major order (XX, XY, YX, YY),
            # whereas the standard maths assumes column-major ordering
            # (XX, YX, XY, YY). This subtle change means we can use the MS
            # data directly without worrying about swapping elements around.
            a_kron_bt(lop, rop, tmp_kprod)

            w_0, w_1, w_2, w_3 = unpack(w)  # NOTE: XX, XY, YX, YY
            r_0, _, _, r_3 = unpack(res)  # NOTE: XX, XY, YX, YY

            g_0, _, _, g_3 = unpack(gain)
            gc_0, _, _, gc_3 = unpackc(gain)

            drv_00 = -1j*gc_0
            drv_23 = -1j*gc_3

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_23*r_3).real

            jhr[0] += upd_00
            jhr[1] += nu*upd_00
            jhr[2] += upd_11
            jhr[3] += nu*upd_11

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[0])
            j_0, j_1, j_2, j_3 = unpackc(tmp_kprod[0])

            jhwj_00 = jh_0*w_0*j_0 + jh_1*w_1*j_1 + jh_2*w_2*j_2 + jh_3*w_3*j_3

            j_0, j_1, j_2, j_3 = unpackc(tmp_kprod[3])

            jhwj_03 = jh_0*w_0*j_0 + jh_1*w_1*j_1 + jh_2*w_2*j_2 + jh_3*w_3*j_3

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[3])
            jhwj_33 = jh_0*w_0*j_0 + jh_1*w_1*j_1 + jh_2*w_2*j_2 + jh_3*w_3*j_3

            nusq = nu*nu

            tmp_0 = jhwj_00.real
            jhj[0, 0] += tmp_0
            jhj[0, 1] += tmp_0*nu
            tmp_1 = (jhwj_03*gc_0*g_3).real
            jhj[0, 2] += tmp_1
            jhj[0, 3] += tmp_1*nu

            jhj[1, 0] = jhj[0, 1]
            jhj[1, 1] += tmp_0*nusq
            jhj[1, 2] = jhj[0, 3]
            jhj[1, 3] += tmp_1*nusq

            jhj[2, 0] = jhj[0, 2]
            jhj[2, 1] = jhj[1, 2]
            tmp_2 = jhwj_33.real
            jhj[2, 2] += tmp_2
            jhj[2, 3] += tmp_2*nu

            jhj[3, 0] = jhj[0, 3]
            jhj[3, 1] = jhj[1, 3]
            jhj[3, 2] = jhj[2, 3]
            jhj[3, 3] += tmp_2*nusq

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, nu, gain, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)

            r_0, r_1 = unpack(res)
            gc_0, gc_1 = unpackc(gain)

            drv_00 = -1j*gc_0
            drv_23 = -1j*gc_1

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_23*r_1).real

            jhr[0] += upd_00
            jhr[1] += nu*upd_00
            jhr[2] += upd_11
            jhr[3] += nu*upd_11

            # Accumulate an element of jhwj.
            jh_00, jh_11 = unpack(rop)
            j_00, j_11 = unpackc(rop)
            w_00, w_11 = unpack(w)

            nusq = nu*nu

            tmp = (jh_00*w_00*j_00).real
            jhj[0, 0] += tmp
            jhj[0, 1] += tmp*nu
            jhj[1, 0] += tmp*nu
            jhj[1, 1] += tmp*nusq

            tmp = (jh_11*w_11*j_11).real
            jhj[2, 2] += tmp
            jhj[2, 3] += tmp*nu
            jhj[3, 2] += tmp*nu
            jhj[3, 3] += tmp*nusq

    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, nu, gain, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)

            r_0 = unpack(res)
            gc_0 = unpackc(gain)

            drv_00 = -1j*gc_0

            upd_00 = (drv_00*r_0).real

            jhr[0] += upd_00
            jhr[1] += nu*upd_00

            # Accumulate an element of jhwj.
            jh_00 = unpack(rop)
            j_00 = unpackc(rop)
            w_00 = unpack(w)

            nusq = nu*nu

            tmp = (jh_00*w_00*j_00).real
            jhj[0, 0] += tmp
            jhj[0, 1] += tmp*nu
            jhj[1, 0] += tmp*nu
            jhj[1, 1] += tmp*nusq
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)
