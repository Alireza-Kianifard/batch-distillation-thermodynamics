from typing import Tuple
import numpy as np
from ..models.mixture import Mixture
from ..core.eos_pr import calculate_pr_fugacity_coefficients
from ..core.activity_nrtl import calculate_gamma_and_psat, calculate_poynting_factors
from .wilson import calculate_wilson_k

def rachford_rice_objective(nv: float, z_feed: np.ndarray, k_values: np.ndarray) -> float:
    denominators = 1.0 + nv * (k_values - 1.0)
    denominators[np.abs(denominators) < 1e-12] = 1e-12 * np.sign(
        denominators[np.abs(denominators) < 1e-12] + 1e-20
    )
    return float(np.sum(z_feed * (k_values - 1.0) / denominators))

def solve_nv_rachford_rice(
    z_feed: np.ndarray, k_values: np.ndarray, tol: float = 1e-9, max_iter: int = 100
) -> float:
    z = np.asarray(z_feed, dtype=float)
    k = np.asarray(k_values, dtype=float)
    sum_zk = np.sum(z * k)
    if sum_zk <= 1.0:
        return 0.0
    sum_z_div_k = np.sum(z / np.where(k == 0.0, 1e-12, k))
    if sum_z_div_k <= 1.0:
        return 1.0
    nv_low, nv_high = 0.0, 1.0
    f_low = rachford_rice_objective(nv_low, z, k)
    if abs(f_low) < tol:
        return nv_low
    for _ in range(max_iter):
        nv_mid = (nv_low + nv_high) / 2.0
        f_mid = rachford_rice_objective(nv_mid, z, k)
        if abs(f_mid) < tol or (nv_high - nv_low) / 2.0 < tol:
            return nv_mid
        if f_mid > 0.0:
            nv_low = nv_mid
        else:
            nv_high = nv_mid
    return (nv_low + nv_high) / 2.0

def perform_rachford_rice_flash(
    z_feed: np.ndarray, k_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    z = np.asarray(z_feed, dtype=float)
    k = np.asarray(k_values, dtype=float)
    nv = solve_nv_rachford_rice(z, k)
    nv = float(np.clip(nv, 0.0, 1.0))
    denominators = 1.0 + nv * (k - 1.0)
    denominators[np.abs(denominators) < 1e-12] = 1e-12 * np.sign(
        denominators[np.abs(denominators) < 1e-12] + 1e-20
    )
    x = z / denominators
    y = k * x
    if nv <= 1e-9:
        nv = 0.0
        x = z / np.sum(z) if np.sum(z) > 0 else np.copy(z)
        y = k * x
        if np.sum(y) > 0:
            y = y / np.sum(y)
    elif nv >= (1.0 - 1e-9):
        nv = 1.0
        y = z / np.sum(z) if np.sum(z) > 0 else np.copy(z)
        x = y / np.where(k == 0.0, 1e-12, k)
        if np.sum(x) > 0:
            x = x / np.sum(x)
    else:
        if abs(np.sum(x)) > 1e-9:
            x = x / np.sum(x)
        if abs(np.sum(y)) > 1e-9:
            y = y / np.sum(y)
    return x, y, nv

def flash(
    t_f: float,
    p_psi: float,
    feed: np.ndarray,
    mixture: Mixture,
    max_iter: int = 100,
    tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t_k = ((t_f - 32.0) * 5.0 / 9.0) + 273.15
    p_pa = p_psi * 6894.76
    n = mixture.n_components
    feed_arr = np.asarray(feed, dtype=float)
    k_values = calculate_wilson_k(p_pa, t_k, mixture)
    _, psat_pa_const_t = calculate_gamma_and_psat(t_k, np.ones_like(feed_arr), mixture)
    phi_sat_v = np.zeros(n)
    for i in range(n):
        phi_pure, _ = calculate_pr_fugacity_coefficients(
            t_k, psat_pa_const_t[i], np.eye(n)[i], mixture,
            is_pure_component=True, pure_component_index=i
        )
        phi_sat_v[i] = phi_pure[i] if phi_pure[i] > 1e-9 else 1.0
    x, y, nv = np.copy(feed_arr), np.copy(feed_arr), 0.0
    for _ in range(max_iter):
        x, y, nv = perform_rachford_rice_flash(feed_arr, k_values)
        gamma, psat_pa = calculate_gamma_and_psat(t_k, x, mixture)
        phi_hat_v, _ = calculate_pr_fugacity_coefficients(t_k, p_pa, y, mixture)
        poynting = calculate_poynting_factors(t_k, p_pa, psat_pa, mixture)
        safe_phi_hat_v = np.where(phi_hat_v < 1e-9, 1.0, phi_hat_v)
        numerators = gamma * phi_sat_v * psat_pa * poynting
        denominators = safe_phi_hat_v * p_pa
        k_new = numerators / np.where(np.abs(denominators) < 1e-20, 1e-20, denominators)
        error_sq = ((k_new - k_values) / k_values)**2
        if np.sqrt(np.mean(error_sq)) < tol:
            break
        k_values = np.clip(k_new, 1e-10, 1e10)
    return k_values, x, y, nv
