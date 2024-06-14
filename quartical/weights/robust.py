import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import coerce_literal, JIT_OPTIONS
import quartical.gains.general.factories as factories
from quartical.gains.general.generics import compute_residual


@njit(**JIT_OPTIONS)
def update_icovariance(residuals, flags, etas, icovariance, mode):
    return update_icovariance_impl(residuals, flags, etas, icovariance, mode)


def update_icovariance_impl(residuals, flags, etas, icovariance, mode):
    raise NotImplementedError


@overload(update_icovariance_impl, jit_options=JIT_OPTIONS)
def nb_update_icovariance_impl(residuals, flags, etas, icovariance, mode):

    update_covariance_inner = update_covariance_inner_factory(mode)

    def impl(residuals, flags, etas, icovariance, mode):

        n_row, n_chan, n_corr = residuals.shape

        # NOTE: We don't bother with the off diagonal entries.
        covariance = np.zeros(n_corr, dtype=np.float64)

        n_unflagged = n_row * n_chan

        for r in range(n_row):
            for f in range(n_chan):
                if flags[r, f] == 1:
                    n_unflagged -= 1
                    continue
                else:
                    update_covariance_inner(residuals[r, f],
                                            etas[r, f],
                                            covariance)

        if n_unflagged:
            covariance /= n_unflagged
            icovariance[:] = 1/covariance

    return impl


def update_covariance_inner_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackc = factories.unpackc_factory(mode)

    if mode.literal_value == 4:
        def impl(res, eta, cov):
            r0, r1, r2, r3 = unpack(res)
            r0c, r1c, r2c, r3c = unpackc(res)

            cov[0] += (r0 * eta * r0c).real
            cov[1] += (r1 * eta * r1c).real
            cov[2] += (r2 * eta * r2c).real
            cov[3] += (r3 * eta * r3c).real
    elif mode.literal_value == 2:
        def impl(res, eta, cov):
            r0, r1 = unpack(res)
            r0c, r1c = unpackc(res)

            cov[0] += (r0 * eta * r0c).real
            cov[1] += (r1 * eta * r1c).real
    elif mode.literal_value == 1:
        def impl(res, eta, cov):
            r0 = unpack(res)
            r0c = unpackc(res)

            cov[0] += (r0 * eta * r0c).real
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def update_etas(residuals, flags, etas, icovariance, dof, mode):
    return update_etas_impl(residuals, flags, etas, icovariance, dof, mode)


def update_etas_impl(residuals, flags, etas, icovariance, dof, mode):
    raise NotImplementedError


@overload(update_etas_impl, jit_options=JIT_OPTIONS)
def nb_update_etas_impl(residuals, flags, etas, icovariance, dof, mode):

    update_etas_inner = update_etas_inner_factory(mode)

    def impl(residuals, flags, etas, icovariance, dof, mode):

        n_row, n_chan, n_corr = residuals.shape

        for r in range(n_row):
            for f in range(n_chan):
                if flags[r, f] == 1:
                    continue
                else:
                    numerator = dof + n_corr  # Not correct for diagonal terms.
                    denominator = dof + update_etas_inner(residuals[r, f],
                                                          icovariance)
                    etas[r, f] = numerator/denominator

    return impl


def update_etas_inner_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackc = factories.unpackc_factory(mode)

    if mode.literal_value == 4:
        def impl(res, icov):
            r0, r1, r2, r3 = unpack(res)
            r0c, r1c, r2c, r3c = unpackc(res)
            ic0, ic1, ic2, ic3 = unpack(icov)

            return (r0 * ic0 * r0c).real + \
                   (r1 * ic1 * r1c).real + \
                   (r2 * ic2 * r2c).real + \
                   (r3 * ic3 * r3c).real
    elif mode.literal_value == 2:
        def impl(res, icov):
            r0, r1 = unpack(res)
            r0c, r1c = unpackc(res)
            ic0, ic1 = unpack(icov)

            return (r0 * ic0 * r0c).real + (r1 * ic1 * r1c).real
    elif mode.literal_value == 1:
        def impl(res, icov):
            r0 = unpack(res)
            r0c = unpackc(res)
            ic0 = unpack(icov)

            return (r0 * ic0 * r0c).real
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def digamma(x):

    result = 0
    while x <= 6:
        result -= 1/x  # Recurrence relation, gamma(x) = gamma(x+1) - 1/x
        x += 1
    result += np.log(x) - 1/(2*x)

    coeffs = (-1/12, 1/120, -1/252, 1/240, -1/132, 391/32760, -1/12)
    ix = 1/(x*x)
    for c in coeffs:
        result += c*ix
        ix *= ix

    return result


@njit(**JIT_OPTIONS)
def dof_variable(dof):
    return np.log(dof) - digamma(dof)


@njit(**JIT_OPTIONS)
def dof_constant(etas, flags, dof, n_corr):

    n_row, n_chan = etas.shape

    n_unflagged = n_row * n_chan

    constant = 0
    for r in range(n_row):
        for f in range(n_chan):
            if flags[r, f] == 1:
                n_unflagged -= 1
                continue
            else:
                eta = etas[r, f]  # We have the same weight on all corrs.
                constant += np.log(eta) - eta

    if n_unflagged:
        constant /= n_unflagged

    return constant + 1 + digamma(dof + n_corr) - np.log(dof + n_corr)


@njit(**JIT_OPTIONS)
def compute_dof(etas, flags, dof, n_corr):

    constant = dof_constant(etas, flags, dof, n_corr)

    left = 1
    right = 30

    maxiter = 50
    tol = 1e-8
    for _ in range(maxiter):
        mid = (left + right)/2
        mid_value = dof_variable(mid) + constant
        if mid_value == 0 or ((right - left) < tol):
            break
        left_value = dof_variable(left) + constant
        if np.sign(mid_value) != np.sign(left_value):
            right = mid
        else:
            left = mid

    return mid


@njit(**JIT_OPTIONS)
def mean_weight(weights, flags):

    n_row, n_chan, n_corr = weights.shape

    mean = np.zeros(n_corr, dtype=weights.dtype)

    n_unflagged = n_row * n_chan

    for r in range(n_row):
        for f in range(n_chan):
            if flags[r, f] == 1:
                n_unflagged -= 1
                continue
            else:
                for c in range(n_corr):
                    mean[c] += weights[r, f, c]

    if n_unflagged:
        mean /= n_unflagged

    return mean


@njit(**JIT_OPTIONS)
def update_weights(weights, etas, icovariance):

    n_row, n_chan, n_corr = weights.shape

    for r in range(n_row):
        for f in range(n_chan):
            for c in range(n_corr):

                weights[r, f, c] = etas[r, f] * icovariance[c]


@njit(**JIT_OPTIONS)
def robust_reweighting(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    etas,
    icovariance,
    dof,
    corr_mode
):
    return robust_reweighting_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        etas,
        icovariance,
        dof,
        corr_mode
    )


def robust_reweighting_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    etas,
    icovariance,
    dof,
    corr_mode
):
    raise NotImplementedError


@overload(robust_reweighting_impl, jit_options=JIT_OPTIONS)
def nb_robust_reweighting_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    etas,
    icovariance,
    dof,
    corr_mode
):

    coerce_literal(nb_robust_reweighting_impl, ["corr_mode"])

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        etas,
        icovariance,
        dof,
        corr_mode
    ):
        model = ms_inputs.MODEL_DATA
        data = ms_inputs.DATA
        antenna1 = ms_inputs.ANTENNA1
        antenna2 = ms_inputs.ANTENNA2
        weights = ms_inputs.WEIGHT
        flags = ms_inputs.FLAG
        row_map = ms_inputs.ROW_MAP
        row_weights = ms_inputs.ROW_WEIGHTS

        t_map_arr = mapping_inputs.time_maps
        f_map_arr = mapping_inputs.freq_maps
        d_map_arr = mapping_inputs.dir_maps

        gains = chain_inputs.gains

        residuals = compute_residual(
            data,
            model,
            gains,
            antenna1,
            antenna2,
            t_map_arr,
            f_map_arr,
            d_map_arr,
            row_map,
            row_weights,
            corr_mode
        )

        # First reweighting - we have already calibrated with MS weights.
        # This tries to approximate what that means in terms of initial values.
        if np.all(icovariance == 0):
            icovariance[:] = mean_weight(weights, flags)
            update_etas(residuals, flags, etas, icovariance, dof, corr_mode)

        update_icovariance(residuals, flags, etas, icovariance, corr_mode)

        dof = compute_dof(etas, flags, dof, corr_mode)

        update_etas(residuals, flags, etas, icovariance, dof, corr_mode)

        update_weights(weights, etas, icovariance)

        return dof

    return impl
