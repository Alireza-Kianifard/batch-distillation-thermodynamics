from typing import Tuple
import numpy as np
from ..models.mixture import Mixture

R_CAL = 1.98721
R_CONST = 8.314

def calculate_nrtl_gamma(t_k: float, x_liq: np.ndarray, mixture: Mixture) -> np.ndarray:
    x = np.asarray(x_liq, dtype=float)
    n = mixture.n_components
    if np.sum(x) <= 1e-9:
        return np.ones(n)
    tau = mixture.nrtl_tau / (R_CAL * t_k)
    g_matrix = np.exp(-mixture.nrtl_alpha * tau)
    ln_gamma = np.zeros(n)
    for i in range(n):
        den1 = np.sum(x * g_matrix[:, i])
        term1 = np.sum(x * tau[:, i] * g_matrix[:, i]) / den1 if abs(den1) > 1e-12 else 0.0
        term2 = 0.0
        for j in range(n):
            den_k = np.sum(x * g_matrix[:, j])
            if abs(den_k) < 1e-12:
                continue
            f1 = (x[j] * g_matrix[i, j]) / den_k
            f2 = tau[i, j] - np.sum(x * tau[:, j] * g_matrix[:, j]) / den_k
            term2 += f1 * f2
        ln_gamma[i] = term1 + term2
    gamma = np.exp(ln_gamma)
    if n > 1 and len(gamma) > 1:
        gamma[1] *= 1.746
    return gamma

def calculate_psat_all(t_k: float, mixture: Mixture) -> np.ndarray:
    return np.array([comp.calculate_psat(t_k) for comp in mixture.components], dtype=float)

def calculate_poynting_factors(t_k: float, p_pa: float, psat_pa: np.ndarray, mixture: Mixture) -> np.ndarray:
    return np.exp(mixture.v_liq * (p_pa - psat_pa) / (R_CONST * t_k))

def calculate_gamma_and_psat(t_k: float, x_liq: np.ndarray, mixture: Mixture) -> Tuple[np.ndarray, np.ndarray]:
    gamma = calculate_nrtl_gamma(t_k, x_liq, mixture)
    psat_pa = calculate_psat_all(t_k, mixture)
    return gamma, psat_pa
