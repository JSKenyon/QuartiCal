# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
from scipy.interpolate import interp2d
from numba import jit
from functools import partial
from quartical.utils.maths import (kron_matvec, kron_tensorvec,
                                   fit_hyperplane, pcg, matern52)


def linear2d_interpolate_gains(interp_xds, term_xds):
    """Interpolate from interp_xds to term_xds linearly.

    Args:
        interp_xds: xarray.Dataset containing the data to interpolate from.
        term_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """
    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]
    t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

    i_t_dim = interp_xds.dims[i_t_axis]
    i_f_dim = interp_xds.dims[i_f_axis]

    interp_axes = {}

    if i_t_dim > 1:
        interp_axes[i_t_axis] = term_xds[t_t_axis].data
    if i_f_dim > 1:
        interp_axes[i_f_axis] = term_xds[t_f_axis].data

    output_xds = interp_xds.interp(
        interp_axes,
        kwargs={"fill_value": "extrapolate"}
    )

    if i_t_dim == 1:
        output_xds = output_xds.reindex(
            {i_t_axis: term_xds[t_t_axis].data},
            method="nearest"
        )
    if i_f_dim == 1:
        output_xds = output_xds.reindex(
            {i_f_axis: term_xds[t_f_axis].data},
            method="nearest"
        )

    return output_xds


def spline2d(x, y, z, xx, yy):
    """Constructs a 2D spline using (x,y,z) and evaluates it at (xx,yy)."""

    n_t, n_f, n_a, n_d, n_c = z.shape
    n_ti, n_fi = xx.size, yy.size

    zz = np.zeros((n_ti, n_fi, n_a, n_d, n_c), dtype=z.dtype)

    # NOTE: x are the column coordinates and y and row coordinates.
    for a in range(n_a):
        for d in range(n_d):
            for c in range(n_c):
                z_sel = z[:, :, a, d, c]
                if not np.any(z_sel):
                    continue
                interp_func = interp2d(y, x, z_sel, kind="cubic")
                zz[:, :, a, d, c] = interp_func(yy, xx).reshape(n_ti, n_fi)

    return zz


def spline2d_interpolate_gains(interp_xds, term_xds):
    """Interpolate from interp_xds to term_xds using a 2D spline.

    Args:
        interp_xds: xarray.Dataset containing the data to interpolate from.
        term_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """
    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]
    t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

    output_xds = term_xds

    if interp_xds.dims[i_t_axis] < 4 or interp_xds.dims[i_f_axis] < 4:
        raise ValueError(
            f"Cubic spline interpolation requires at least four "
            f"values along an axis. After concatenation, the "
            f"(time, freq) dimensions of the interpolating dataset were "
            f"{(interp_xds.dims[i_t_axis], interp_xds.dims[i_f_axis])}"
        )

    for data_field in interp_xds.data_vars.keys():
        interp = da.blockwise(spline2d, "tfadc",
                              interp_xds[i_t_axis].values, None,
                              interp_xds[i_f_axis].values, None,
                              interp_xds[data_field].data, "tfadc",
                              term_xds[t_t_axis].values, None,
                              term_xds[t_f_axis].values, None,
                              dtype=np.float64,
                              adjust_chunks={"t": term_xds.dims[t_t_axis],
                                             "f": term_xds.dims[t_f_axis]})

        output_xds = output_xds.assign(
            {data_field: (term_xds.GAIN_AXES, interp)})

    return output_xds


@jit(nopython=True, nogil=True, cache=True)
def _interpolate_missing(x1, x2, y):
    """Interpolate/extend data y along x1 and x2 to fill in missing values."""

    n_t, n_f, n_a, n_d, n_c = y.shape

    yy = y.copy()

    for f in range(n_f):
        for a in range(n_a):
            for d in range(n_d):
                for c in range(n_c):
                    y_sel = y[:, f, a, d, c]
                    good_data = np.where(np.isfinite(y_sel))
                    if len(good_data[0]) == 0:
                        continue

                    yy[:, f, a, d, c] = linterp(x1,
                                                x1[good_data],
                                                y_sel[good_data])

    for t in range(n_t):
        for a in range(n_a):
            for d in range(n_d):
                for c in range(n_c):
                    y_sel = yy[t, :, a, d, c]
                    good_data = np.where(np.isfinite(y_sel))
                    if len(good_data[0]) == 0:
                        # If there is no good data along frequency after
                        # interpolating in time, we have no information
                        # from which to interpolate - we zero these locations.
                        yy[t, :, a, d, c] = 0
                        continue

                    yy[t, :, a, d, c] = linterp(x2,
                                                x2[good_data],
                                                y_sel[good_data])

    return yy


@jit(nopython=True, nogil=True, cache=True)
def linterp(xx, x, y):
    """Basic linear interpolation. Extrapolates with closest good value."""

    xi = 0
    xxi = 0

    yy = np.zeros(xx.shape, dtype=y.dtype)
    xxn = len(xx)

    while xxi < xxn:
        xxel = xx[xxi]
        xel = x[xi]
        if xxel == xel:
            yy[xxi] = y[xi]
            xxi += 1
        elif xxel < x[0]:
            yy[xxi] = y[0]
            xxi += 1
        elif xxel > x[-1]:
            yy[xxi] = y[-1]
            xxi += 1
        elif (xxel > xel) & (xxel < x[xi + 1]):
            slope = (y[xi + 1] - y[xi]) / (x[xi + 1] - xel)
            yy[xxi] = slope * (xxel - xel) + y[xi]
            xxi += 1
        else:
            xi += 1

    return yy


def interpolate_missing(interp_xds):
    """Linear interpolate missing values in the given xarray dataset.

    Args:
        interp_xds: xarray.Dataset containing the data.

    Returns:
        output_xds: xarray.Dataset containing the data after interpolation.
    """

    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]

    output_xds = interp_xds

    for data_field in interp_xds.data_vars:

        interp = da.blockwise(_interpolate_missing, "tfadc",
                              interp_xds[i_t_axis].values, None,
                              interp_xds[i_f_axis].values, None,
                              interp_xds[data_field].data, "tfadc",
                              dtype=np.float64)

        output_xds = output_xds.assign(
            {data_field: (interp_xds[data_field].dims, interp)})

    return output_xds


def gpr_interpolate_gains(interp_xds, term_xds, gpr_params):
    """Interpolate from interp_xds to term_xds using a 2D spline.

    Args:
        interp_xds: xarray.Dataset containing the data to interpolate from.
        term_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """
    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]
    t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]
    # TODO - add spatial axis if more than one direction is present

    output_xds = term_xds

    # This is done externally because we are using the same length
    # scale hyper-parameters for all antennas and correlations.
    # We'll have to check if it's better to compute here or inside
    # the function call in the distributed case
    t = interp_xds[i_t_axis].values
    f = interp_xds[i_f_axis].values
    tp = term_xds[t_t_axis].values
    fp = term_xds[t_f_axis].values

    # currently assume jhj is the same for all data_vars
    jhj = interp_xds["jhj"].data  #.rechunk({2: 1})
    # we don't want to interpolate jhj
    interp_xds = interp_xds.drop_vars({'jhj'})
    for data_field in interp_xds.data_vars:
        lts = gpr_params[0]
        lfs = gpr_params[1]
        ninflates = gpr_params[2]
        if data_field in ['phase','imag']:
            lt = lts[1]
            lf = lfs[1]
            noise_inflation = ninflates[1]
        elif data_field in ['amp', 'real', 'complex']:
            lt = lts[0]
            lf = lfs[0]
            noise_inflation = ninflates[0]

        # rechunk along antenna dimension for parallelism
        gains = interp_xds[data_field].data  #.rechunk({2: 1})

        interp = da.blockwise(interp_gpr, "tfadc",
                              gains, "tfadc",
                              jhj, "tfadc",
                              t, None,
                              f, None,
                              tp, None,
                              fp, None,
                              lt, None,
                              lf, None,
                              noise_inflation, None,
                              data_field, None,
                              dtype=np.complex128,
                              adjust_chunks={"t": term_xds.dims[t_t_axis],
                                             "f": term_xds.dims[t_f_axis]})
        interp = interp.rechunk({2: -1})
        output_xds = output_xds.assign(
            {data_field: (term_xds.GAIN_AXES, interp)})

    return output_xds


def interp_gpr(gain, jhj, t, f, tp, fp, lt, lf,
               noise_inflation, term_type):

    jhj = jhj.real

    f = f/1e6  # convert to MHz
    fp = fp/1e6
    K = (matern52(t, t, 1.0, lt), matern52(f, f, 1.0, lf))
    Kp = (matern52(tp, t, 1.0, lt), matern52(fp, f, 1.0, lf))

    tt, ff = np.meshgrid(t, f, indexing='ij')
    x = np.vstack((tt.flatten(), ff.flatten()))
    ttp, ffp = np.meshgrid(tp, fp, indexing='ij')
    xp = np.vstack((ttp.flatten(), ffp.flatten()))

    D, N = x.shape
    X = np.vstack((x, np.ones((1,N))))
    D, Np = xp.shape
    Xp = np.vstack((xp, np.ones((1,Np))))
    ntp = tp.size
    nfp = fp.size

    # operator that needs to be inverted
    def Kyop(x, K, Sigma, sigmafsq):
        # sigmafsq is actually part of K but we want to set it
        # per antenna and correlation so we cheat this way
        return kron_matvec(K, sigmafsq*x) + Sigma*x

    nt, nf, nant, ndir, ncorr = jhj.shape
    sol = np.zeros((ntp, nfp, nant, ndir, ncorr), dtype=gain.dtype)
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                g = gain[:, :, p, d, c]
                # unwrap if phase
                if term_type == 'phase':
                    g = np.unwrap(np.unwrap(g, axis=0), axis=1)
                jhjflat = jhj[:, :, p, d, c].ravel()
                # assumes gain flags correspond to where jhj is zero
                ival = np.where(jhjflat>0)[0]
                xval = x[:, ival]
                yval = g.ravel()[ival]
                theta = fit_hyperplane(xval, yval)
                # subtract plane approx
                plane_approx = X.T.dot(theta).reshape(nt, nf)
                y = gain[:, :, p, d, c] - plane_approx
                sigmafsq = np.var(y)
                # this gives small weight to flaged data
                # need to check that it's sensible
                Sigma = np.where(jhj[:, :, p, d, c]>0,
                                 noise_inflation/jhj[:, :, p, d, c], 1e10)
                Ky = partial(Kyop, K=K, Sigma=Sigma, sigmafsq=sigmafsq)
                Kyinv, success, eps = pcg(Ky,
                                          y,
                                          np.zeros((nt, nf), dtype=gain.dtype),
                                          tol=1e-6,
                                          maxit=500)
                sol[:, :, p, d, c] = (kron_tensorvec(Kp, sigmafsq * Kyinv) +
                                      Xp.T.dot(theta).reshape(ntp, nfp))
                if not success:
                    print(f"                                    {term_type} failed at antenna {p} with eps of {eps}")
                else:
                    print(f"{term_type} succeeded at antenna {p}")

    return sol

