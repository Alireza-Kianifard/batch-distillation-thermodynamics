from typing import Tuple, Optional
import numpy as np
from ..models.mixture import Mixture

R_CONST = 8.314

def calculate_pr_pure_params(t_k: float, mixture: Mixture) -> Tuple[np.ndarray, np.ndarray]:
    tr = t_k / mixture.tc
    omega = mixture.omega
    m = np.where(
        omega <= 0.49,
        0.37464 + 1.54226 * omega - 0.26992 * omega**2,
        0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
    )
    alpha_t = (1.0 + m * (1.0 - np.sqrt(tr)))**2
    ac_i = 0.457235 * (R_CONST * mixture.tc)**2 / mixture.pc
    a_i = ac_i * alpha_t
    b_i = 0.077796 * R_CONST * mixture.tc / mixture.pc
    return a_i, b_i

def solve_pr_eos_for_z(
    t_k: float, p_pa: float, composition: np.ndarray, mixture: Mixture
) -> Tuple[Optional[float], Optional[float]]:
    y = np.asarray(composition, dtype=float)
    a_i, b_i = calculate_pr_pure_params(t_k, mixture)
    a_mix = np.sum(y[:, np.newaxis] * y * np.sqrt(a_i[:, np.newaxis] * a_i) * (1.0 - mixture.kij_matrix))
    b_mix = np.sum(y * b_i)
    a_param = a_mix * p_pa / (R_CONST**2 * t_k**2)
    b_param = b_mix * p_pa / (R_CONST * t_k)
    coeffs = [
        1.0,
        -(1.0 - b_param),
        a_param - 2.0 * b_param - 3.0 * b_param**2,
        -(a_param * b_param - b_param**2 - b_param**3)
    ]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    positive_roots = real_roots[real_roots > b_param]
    if len(positive_roots) > 0:
        return float(np.min(positive_roots)), float(np.max(positive_roots))
    return None, None

def calculate_pr_fugacity_coefficients(
    t_k: float,
    p_pa: float,
    mol_fractions: np.ndarray,
    mixture: Mixture,
    is_pure_component: bool = False,
    pure_component_index: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    y = np.asarray(mol_fractions, dtype=float)
    if p_pa <= 0:
        p_pa = 1e-3
    if not is_pure_component and np.sum(y) < 1e-9:
        return np.ones_like(y), 1.0
    a_i, b_i = calculate_pr_pure_params(t_k, mixture)
    if is_pure_component:
        if pure_component_index is None:
            raise ValueError("pure_component_index required.")
        a_mix = a_i[pure_component_index]
        b_mix = b_i[pure_component_index]
    else:
        a_mix = np.sum(y[:, np.newaxis] * y * np.sqrt(a_i[:, np.newaxis] * a_i) * (1.0 - mixture.kij_matrix))
        b_mix = np.sum(y * b_i)
    a_param = a_mix * p_pa / (R_CONST**2 * t_k**2)
    b_param = b_mix * p_pa / (R_CONST * t_k)
    coeffs = [
        1.0,
        -(1.0 - b_param),
        a_param - 2.0 * b_param - 3.0 * b_param**2,
        -(a_param * b_param - b_param**2 - b_param**3)
    ]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    z = b_param + 1e-6
    if len(real_roots) > 0:
        pos_roots = real_roots[real_roots > b_param]
        if len(pos_roots) > 0:
            z = float(np.max(pos_roots))
    if is_pure_component:
        if abs(b_param) < 1e-12:
            ln_phi = (z - 1.0) - np.log(z)
        else:
            log_arg = (z + (1.0 + np.sqrt(2)) * b_param) / (z + (1.0 - np.sqrt(2)) * b_param)
            ln_phi = (
                (z - 1.0)
                - np.log(z - b_param if z > b_param else 1e-12)
                - (a_param / (2.0 * np.sqrt(2) * b_param)) * np.log(log_arg if log_arg > 0 else 1e-12)
            )
        phi = np.zeros(mixture.n_components)
        phi[pure_component_index] = np.exp(ln_phi)
        return phi, z
    ln_phi_k = np.zeros(len(y))
    log_zv_b_term = -np.log(z - b_param if z > b_param else 1e-12)
    term_common_log = 0.0
    if abs(b_param) > 1e-12:
        log_arg = (z + (1.0 + np.sqrt(2)) * b_param) / (z + (1.0 - np.sqrt(2)) * b_param)
        term_common_log = np.log(log_arg if log_arg > 0 else 1e-12)
    for k_idx in range(len(y)):
        sum_val = np.sum(y * np.sqrt(a_i * a_i[k_idx]) * (1.0 - mixture.kij_matrix[:, k_idx]))
        term1 = (b_i[k_idx] / b_mix if abs(b_mix) > 1e-12 else 0.0) * (z - 1.0)
        term2 = log_zv_b_term
        term3_factor1 = (a_param / (2.0 * np.sqrt(2) * b_param)) if abs(b_param) > 1e-12 else 0.0
        term3_factor2 = (2.0 * sum_val / a_mix if abs(a_mix) > 1e-12 else 0.0) - (
            b_i[k_idx] / b_mix if abs(b_mix) > 1e-12 else 0.0
        )
        term3 = term3_factor1 * term3_factor2 * term_common_log
        ln_phi_k[k_idx] = term1 + term2 - term3
    return np.exp(ln_phi_k), z
