import jax
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp
from diffstar.utils import _jax_get_dt_array

# from sn_diffstar.mspop import _ms_means_and_covs_lgmpop, _get_ms_means, get_ms_sfh_scan_tobs_lgm0, _integrate_sfrpop, SFR_MIN

from diffstar.sfh import get_sfh_from_mah_kern

get_sfh_from_mah = get_sfh_from_mah_kern(tobs_loop='vmap', galpop_loop='vmap')

## DTD Model from Eq. 6 Wiseman et al. 2021 https://academic.oup.com/mnras/article/506/3/3330/6318383?login=false
## Their nominal values are

tp=0.04 # Gyr
A = 2.11e-13 # / M_star / yr
beta = -1.13

## SFH history units pf M_star/yr

## SNR returns 1/yr

@jjit
def SNR(t0, A, beta, tp, mah_params_one, ms_u_params_one, q_u_params_one):
    mah_params_pop=mah_params_one[None,:]
    ms_u_params_pop=ms_u_params_one[None,:]
    q_u_params_pop=q_u_params_one[None,:]
    
    taus_p = jnp.linspace(jnp.power(tp, beta+1), jnp.power(t0, beta+1), 200)
    taus = jnp.power(taus_p, -(beta+1))
    
    # lgmpop = mah_params_pop[:,0]
#     means_ms_pop, cov_ms_pop = _ms_means_and_covs_lgmpop(lgmpop)

    # ms_u_params_pop = _get_ms_means(lgmpop)

    # ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(t0-taus, mah_params_pop, ms_u_params_pop)
    
    # ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(t0-taus, mah_params_pop, ms_u_params_pop)
    
    # ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)

#     _, ms_sfh_pop, _ = mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, taus+t0)
    
    ms_sfh_pop = get_sfh_from_mah(t0-taus, mah_params_pop, ms_u_params_pop, q_u_params_pop)
    
    # 1e9 since t0 units in Gyr, SFH units in yr
    return 1e9/(1+beta)*A*jnp.trapz(ms_sfh_pop[0,:],taus_p)
    # return ms_sfh_pop[0,:].sum()
    
def temp(t0, A, beta, tp, mah_params_one, ms_u_params_one):
    mah_params_pop=mah_params_one[None,:]
    ms_u_params_pop=ms_u_params_one[None,:]

    from diffstar.stars import calculate_sm_sfr_fstar_history_from_mah

    dmhdt_fit, log_mah_fit = loss_data[2:4]
    lgt = np.log10(tarr)
    index_select, index_high, fstar_tdelay = loss_data[8:11]

    _histories = calculate_sm_sfr_fstar_history_from_mah(
        lgt,
        dt,
        dmhdt_fit,
        log_mah_fit,
        u_sfr_fit_params,
        u_q_fit_params,
        index_select,
        index_high,
        fstar_tdelay,
    )
    smh_fit, sfh_fit, fstar_fit = _histories

SNR_t0  = jjit(vmap(SNR,  in_axes=(0, None, None, None, None, None, None)))
SNR_mah  = jjit(vmap(SNR,  in_axes=(None, None, None, None, 0, 0, 0 )))
SNR_gal  = jjit(vmap(SNR,  in_axes=(0, None, None, None, 0, 0, 0)))
SNR_t0_mah  = jjit(vmap(SNR_mah,  in_axes=(0, None, None, None, None, None, None)))
SNR_beta  = jjit(vmap(SNR,  in_axes=(None, None, 0, None, None, None, None)))
SNR_tp  = jjit(vmap(SNR,  in_axes=(None, None, None, 0, None, None, None)))
SNR_beta_tp  = jjit(vmap(SNR_tp,  in_axes=(None, None, 0, None, None, None, None)))

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