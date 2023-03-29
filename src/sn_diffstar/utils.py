import jax
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp
from diffstar.utils import _jax_get_dt_array

from sn_diffstar.mspop import _ms_means_and_covs_lgmpop, _get_ms_means, get_ms_sfh_scan_tobs_lgm0, _integrate_sfrpop, SFR_MIN

@jjit
def SNR(t0, A, beta, tp, mah_params_one):
    mah_params_pop=mah_params_one[None,:]
    
    taus_p = jnp.linspace(jnp.power(tp, beta+1), jnp.power(t0, beta+1), 1000)
    taus = jnp.power(taus_p, -(beta+1))
    
    lgmpop = mah_params_pop[:,0]
#     means_ms_pop, cov_ms_pop = _ms_means_and_covs_lgmpop(lgmpop)

    ms_u_params_pop = _get_ms_means(lgmpop)

    ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(t0-taus, mah_params_pop, ms_u_params_pop)
    ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)

#     _, ms_sfh_pop, _ = mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, taus+t0)
    return 1./(1+beta)*A*jnp.trapz(ms_sfh_pop[0,:],taus_p)

SNR_t0  = jjit(vmap(SNR,  in_axes=(0, None, None, None, None)))
SNR_mah  = jjit(vmap(SNR,  in_axes=(None, None, None, None, 0)))
SNR_gal  = jjit(vmap(SNR,  in_axes=(0, None, None, None, 0)))
SNR_t0_mah  = jjit(vmap(SNR_mah,  in_axes=(0, None, None, None, None)))
SNR_beta  = jjit(vmap(SNR,  in_axes=(None, None, 0, None, None)))
SNR_tp  = jjit(vmap(SNR,  in_axes=(None, None, None, 0, None)))
SNR_beta_tp  = jjit(vmap(SNR_tp,  in_axes=(None, None, 0, None, None)))

@jjit
def logSM(t0, mah_params_one):
    mah_params_pop=mah_params_one[None,:]
    tarr = jnp.linspace(0.1, t0, 200)
    
    lgmpop = mah_params_pop[:,0]
#     means_ms_pop, cov_ms_pop = _ms_means_and_covs_lgmpop(lgmpop)

    ms_u_params_pop = _get_ms_means(lgmpop)

    ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(tarr, mah_params_pop, ms_u_params_pop)
    ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)
    
    dtarr = _jax_get_dt_array(tarr)
    ms_smh_pop = _integrate_sfrpop(ms_sfh_pop, dtarr)
    ms_logsmh_pop = jnp.log10(ms_smh_pop)
#     _, ms_sfh_pop, _ = mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, taus+t0)
    return ms_logsmh_pop[0,-1]

logSM_t0  = jjit(vmap(logSM,  in_axes=(0, None)))
logSM_mah  = jjit(vmap(logSM,  in_axes=(None, 0)))
logSM_gal  = jjit(vmap(logSM,  in_axes=(0, 0)))
logSM_t0_mah  = jjit(vmap(logSM_mah,  in_axes=(0, None)))