from scipy.special import erf
import numpy as np
from scipy.stats import norm

# def selection_function(mu, m_lim, m_source, sigma_m): ...
# def mag_likelihood(m_obs, mu, m_source, sigma_m): ...


def selection_function(mu, m_lim, ms, sigma_m):
    """选择函数（单个图像）"""
    return 0.5 * (1 + erf((m_lim - ms + 2.5 * np.log10(np.abs(mu))) / (np.sqrt(2) * sigma_m)))

def mag_likelihood(m_obs, mu, ms, sigma_m):
    """星等的似然函数"""
    mag_model = ms - 2.5 * np.log10(np.abs(mu))
    return norm.pdf(m_obs, loc=mag_model, scale=sigma_m)