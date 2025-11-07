from .lens_solver import solve_single_lens
from .lens_model import kpc_to_arcsec
from ..sl_cosmology import Dang
from ..config import SCATTER
import numpy as np

# SPS PARAMETER
# M_star = alpha_sps * M_sps
# logM_star = log_alpha_sps + logM_sps



def lens_properties(model, beta_unit):
   xA, xB = solve_single_lens(model, beta_unit)
   kappaA = model.kappa(xA)
   kappaB = model.kappa(xB)
   gammaA = model.gamma(xA)
   gammaB = model.gamma(xB)
   magnificationA = model.mu_from_rt(xA)
   magnificationB = model.mu_from_rt(xB)
   kappa_starA = model.kappa_star(xA)
   kappa_starB = model.kappa_star(xB)
   alphaA = model.alpha(xA)
   alphaB = model.alpha(xB)  
       
   sA = 1- kappa_starA/kappaA
   sB = 1- kappa_starB/kappaB  

#    Mh5 = model.Mh5()

   einstein_radius = model.einstein_radius()  # [kpc]
   einstein_radius_arcsec = kpc_to_arcsec(einstein_radius, model.zl, Dang)  # [arcsec]

   return {
       'xA': xA, 'xB': xB, 'beta': beta_unit,
       'kappaA': kappaA, 'kappaB': kappaB,
       'gammaA': gammaA, 'gammaB': gammaB,
       'magnificationA': magnificationA, 'magnificationB': magnificationB,
       'kappa_starA': kappa_starA, 'kappa_starB': kappa_starB,
       'alphaA': alphaA, 'alphaB': alphaB,
       'sA': sA, 'sB': sB,
       'einstein_radius_kpc': einstein_radius,
       'einstein_radius_arcsec': einstein_radius_arcsec,
        # 'logMh5': Mh5,
   }

   
   
# add source properties

def observed_data(
   model,
   beta_unit: float,
   m_s: float,
   maximum_magnitude: float,
   logalpha_sps: float,
   logM_star: float,
   logM_star_sps: float,
   logM_halo: float,
   logRe: float,
   zl: float,
   zs: float,
   gamma_in: float,
   c_halo: float,
   *,
   caustic: bool = False,
   scatter_Mstar=None,
):
   """
   计算 lens 的属性，并返回包含源属性的字典（标量参数版本）。
   """
   if model is None:
       raise ValueError("observed_data requires a LensModel instance; do not construct inside observed_data.")

   properties = lens_properties(model, beta_unit)

   scatter_mag = SCATTER.mag  # [mag] 源光度的散射
   properties['scatter_mag'] = scatter_mag
   magnitude_observedA = m_s - 2.5 * np.log10(properties['magnificationA']) + np.random.normal(loc=0.0, scale=scatter_mag)
   magnitude_observedB = m_s - 2.5 * np.log10(properties['magnificationB']) + np.random.normal(loc=0.0, scale=scatter_mag)

   if scatter_Mstar is None:
       scatter_Mstar = SCATTER.star  # [Msun] 源质量的散射

   logMsps_observed = logM_star_sps + np.random.normal(loc=0.0, scale=scatter_Mstar)  # 添加噪声

   # no observed error
   if magnitude_observedA > maximum_magnitude or magnitude_observedB > maximum_magnitude:
       properties['is_lensed'] = False
   else:
       properties['is_lensed'] = True

   # 添加源属性
   properties['magnitude_observedA'] = magnitude_observedA  # [mag]
   properties['magnitude_observedB'] = magnitude_observedB  # [mag]
   properties['m_s'] = m_s  # [mag]
   properties['maximum_magnitude'] = maximum_magnitude  # [mag]
   properties['beta_unit'] = beta_unit  # [kpc]
   properties['logalpha_sps'] = logalpha_sps  # [Msun]
   properties['logM_star'] = logM_star
   properties['logM_star_sps'] = logM_star_sps  # [Msun]
   properties['logM_star_sps_observed'] = logMsps_observed
   properties['logM_halo'] = logM_halo
   properties['logRe'] = logRe
   properties['zl'] = zl
   properties['zs'] = zs
   properties['gamma_in'] = gamma_in
   properties['c_halo'] = c_halo

   # Add halo-level physical quantities for completeness
   try:
       r200_kpc = (model.M_halo * 3.0 / (4 * np.pi * 200 * model.rhoc_z))**(1/3) * 1000.0
   except Exception:
       r200_kpc = None
   properties['r200'] = r200_kpc
   properties['rs'] = getattr(model, 'rs', None)
   # Einstein radius (duplicate of existing fields, added as theta_E)
   try:
       _thetaE_kpc = properties.get('einstein_radius_kpc', None)
       if _thetaE_kpc is None:
           _thetaE_kpc = model.einstein_radius()
       thetaE_arcsec = kpc_to_arcsec(_thetaE_kpc, zl, Dang) if _thetaE_kpc is not None else None
   except Exception:
       thetaE_arcsec = None
   properties['theta_E'] = thetaE_arcsec
   # Normalization and critical density
   properties['gnfw_norm'] = getattr(model, 'gnfw_norm', None)
   properties['sigma_crit'] = getattr(model, 's_cr', None)

   if caustic:
       properties['ycaustic_kpc'] = model.solve_ycaustic()
       properties['ycaustic_arcsec'] = model.solve_ycaustic_arcsec()
       properties['xradcrit_kpc'] = model.solve_xradcrit()
       properties['xradcrit_arcsec'] = kpc_to_arcsec(properties['xradcrit_kpc'], zl, Dang)
   
   return properties


def empty_lens_data(
   model,
   logM_star_sps,
   logM_star,
   logM_halo,
   logRe,
   maximum_magnitude,
   logalpha_sps,
   zl,
   zs,
   gamma_in,
   c_halo,
   *,
   beta_unit=None,
   ycaustic_kpc=None,
):
   """
   Build a placeholder lens record with the same fields as observed_data
   but representing a non-lensed system. Numeric lensing outputs are set
   to None and is_lensed is False.

   Parameters
   ----------
   input_df : pandas.DataFrame
       Single-row DataFrame containing the same inputs expected by
       observed_data (logM_star_sps, logM_star, logM_halo, logRe,
       beta_unit, m_s, maximum_magnitude, logalpha_sps, zl, zs). Values
       for beta_unit and m_s may be None for a pure placeholder.
   ycaustic_kpc : float | None
       Optional caustic size to carry through to the result.

   Returns
   -------
   dict
       A dictionary mirroring observed_data's keys with placeholder
       values for lensing-specific quantities.
   """
   m_s = None

   # Placeholder for lensing properties
   result = {
       'xA': None, 'xB': None, 'beta': None,
       'kappaA': None, 'kappaB': None,
       'gammaA': None, 'gammaB': None,
       'magnificationA': None, 'magnificationB': None,
       'kappa_starA': None, 'kappa_starB': None,
       'alphaA': None, 'alphaB': None,
       'sA': None, 'sB': None,
       'einstein_radius_kpc': None,
       'einstein_radius_arcsec': None,
       'scatter_mag': None,
       'magnitude_observedA': None,
       'magnitude_observedB': None,
       'is_lensed': False,
       # source and galaxy inputs
       'm_s': m_s,
       'maximum_magnitude': maximum_magnitude,
       'beta_unit': beta_unit,
       'logalpha_sps': logalpha_sps,
       'logM_star': logM_star,
       'logM_star_sps': logM_star_sps,
       'logM_star_sps_observed': None,
       'logM_halo': logM_halo,
       'logRe': logRe,
       'zl': zl,
       'zs': zs,
       'gamma_in': gamma_in,
       'c_halo': c_halo,
   }
   # Include caustic information if available
   result['ycaustic_kpc'] = ycaustic_kpc
   result['ycaustic_arcsec'] = (
       kpc_to_arcsec(ycaustic_kpc, zl, Dang) if ycaustic_kpc is not None else None
   )
   # Add halo-level quantities available without image formation
   try:
       r200_kpc = (model.M_halo * 3.0 / (4 * np.pi * 200 * model.rhoc_z))**(1/3) * 1000.0
   except Exception:
       r200_kpc = None
   result['r200'] = r200_kpc
   result['rs'] = getattr(model, 'rs', None)
   # Einstein radius convenience (also available separately as *_kpc/arcsec for lensed cases)
   try:
       thetaE_kpc = model.einstein_radius()
       result['einstein_radius_kpc'] = thetaE_kpc
       result['einstein_radius_arcsec'] = kpc_to_arcsec(thetaE_kpc, zl, Dang)
       result['theta_E'] = result['einstein_radius_arcsec']
   except Exception:
       result['theta_E'] = None
   # Normalization and critical surface density
   result['gnfw_norm'] = getattr(model, 'gnfw_norm', None)
   result['sigma_crit'] = getattr(model, 's_cr', None)
   return result
