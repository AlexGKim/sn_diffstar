"""
"""
from functools import partial
from collections import OrderedDict
from jax import random as jran
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from diffstar.utils import _jax_get_dt_array
from diffstar.main_sequence import get_ms_sfh_from_mah_kern

UH = -0.7671
LGM_X0, LGM_K = 13.0, 0.5
SFR_MIN = 1e-12
DEFAULT_SFH_PDF_MAINSEQ_PDICT = OrderedDict(
    mean_ulgm_mainseq_ylo=10.04,
    mean_ulgm_mainseq_yhi=14.98,
    mean_ulgy_mainseq_ylo=-2.69,
    mean_ulgy_mainseq_yhi=3.83,
    mean_ul_mainseq_ylo=-23.74,
    mean_ul_mainseq_yhi=33.59,
    mean_utau_mainseq_ylo=37.79,
    mean_utau_mainseq_yhi=-34.69,
)
DEFAULT_SFH_PDF_MAINSEQ_PARAMS = jnp.array(
    tuple(DEFAULT_SFH_PDF_MAINSEQ_PDICT.values())
)

get_ms_sfh_scan_tobs_lgm0 = get_ms_sfh_from_mah_kern(
    tobs_loop="scan", galpop_loop="vmap"
)


@jjit
def _ms_means_and_covs_lgmpop(lgmpop):
    means_ms_pop = jnp.array(_get_ms_means(lgmpop)).T

    cov_ms = _get_covs_lgmpop(lgmpop)

    return means_ms_pop, cov_ms


@jjit
def mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, tarr):
    lgmpop = mah_params_pop[:, 0]
    means_ms_pop, cov_ms_pop = _ms_means_and_covs_lgmpop(lgmpop)

    ms_u_params_pop = _get_ms_means(lgmpop)

    ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(tarr, mah_params_pop, ms_u_params_pop)
    ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)

    dtarr = _jax_get_dt_array(tarr)
    ms_smh_pop = _integrate_sfrpop(ms_sfh_pop, dtarr)
    ms_logsmh_pop = jnp.log10(ms_smh_pop)

    return ms_u_params_pop, ms_sfh_pop, ms_logsmh_pop


@jjit
def _get_covs_lgmpop(lgmpop):
    arr = jnp.eye(4) * 0.01
    covs = jnp.repeat(arr[None, ...], lgmpop.size, axis=0)
    return covs


@partial(jjit, static_argnames=["n_gals"])
def _mc_ms_u_params_lgm0(ran_key, means_ms, cov_ms, n_gals):
    ms_params = jran.multivariate_normal(ran_key, means_ms, cov_ms, shape=(n_gals,))
    ulgm = ms_params[:, 0]
    ulgy = ms_params[:, 1]
    ul = ms_params[:, 2]
    utau = ms_params[:, 3]

    uh = jnp.zeros(n_gals) + UH

    ms_u_params = jnp.array((ulgm, ulgy, ul, uh, utau)).T

    return ms_u_params


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _fun(x, ymin, ymax):
    return _sigmoid(x, LGM_X0, LGM_K, ymin, ymax)


@jjit
def _get_ms_means(lgm0, ms_params=DEFAULT_SFH_PDF_MAINSEQ_PARAMS):
    mean_ulgm_mainseq_ylo, mean_ulgm_mainseq_yhi = ms_params[:2]
    mean_ulgy_mainseq_ylo, mean_ulgy_mainseq_yhi = ms_params[2:4]
    mean_ul_mainseq_ylo, mean_ul_mainseq_yhi = ms_params[4:6]
    mean_utau_mainseq_ylo, mean_utau_mainseq_yhi = ms_params[6:8]
    ulgm = _fun(lgm0, mean_ulgm_mainseq_ylo, mean_ulgm_mainseq_yhi)
    ulgy = _fun(lgm0, mean_ulgy_mainseq_ylo, mean_ulgy_mainseq_yhi)
    ul = _fun(lgm0, mean_ul_mainseq_ylo, mean_ul_mainseq_yhi)
    utau = _fun(lgm0, mean_utau_mainseq_ylo, mean_utau_mainseq_yhi)
    uh = jnp.zeros_like(ul) + UH
    return ulgm, ulgy, ul, uh, utau


@jjit
def _integrate_sfr(sfr, dt):
    """Calculate the cumulative stellar mass history."""
    return jnp.cumsum(sfr * dt) * 1e9


_integrate_sfrpop = jjit(vmap(_integrate_sfr, in_axes=[0, None]))
