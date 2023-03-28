import jax
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp

from sn_diffstar.mspop import _ms_means_and_covs_lgmpop, _get_ms_means, get_ms_sfh_scan_tobs_lgm0, SFR_MIN

@jjit
def SNR(t0, A, beta, tp, mah_params_one, Tmax):
    mah_params_pop=mah_params_one[None,:]
    
    taus_p = jnp.linspace(jnp.power(tp, beta+1), jnp.power(Tmax-t0, beta+1), 1000)
    taus = jnp.power(taus_p, -(beta+1))
    
    lgmpop = mah_params_pop[:,0]
#     means_ms_pop, cov_ms_pop = _ms_means_and_covs_lgmpop(lgmpop)

    ms_u_params_pop = _get_ms_means(lgmpop)

    ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(taus+t0, mah_params_pop, ms_u_params_pop)
    ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)

#     _, ms_sfh_pop, _ = mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, taus+t0)
    return 1./(1+beta)*A*jnp.trapz(ms_sfh_pop[0,:],taus_p)

SNR_t0  = jjit(vmap(SNR,  in_axes=(0, None, None, None, None, None)))
SNR_mah  = jjit(vmap(SNR,  in_axes=(None, None, None, None, 0, None)))
SNR_gal  = jjit(vmap(SNR,  in_axes=(0, None, None, None, 0, None)))
SNR_t0_mah  = jjit(vmap(SNR_mah,  in_axes=(0, None, None, None, None, None)))
SNR_beta  = jjit(vmap(SNR,  in_axes=(None, None, 0, None, None, None)))
SNR_tp  = jjit(vmap(SNR,  in_axes=(None, None, None, 0, None, None)))
SNR_beta_tp  = jjit(vmap(SNR_tp,  in_axes=(None, None, 0, None, None, None)))

