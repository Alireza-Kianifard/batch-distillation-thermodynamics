from typing import Tuple, Optional
import numpy as np
from ..models.mixture import Mixture
from ..core.eos_pr import calculate_pr_fugacity_coefficients
from ..core.activity_nrtl import calculate_gamma_and_psat, calculate_poynting_factors
from .wilson import calculate_wilson_k

def _calculate_objective_f_and_k_at_t(
    t_k: float,
    p_pa: float,
    x_comp: np.ndarray,
    mixture: Mixture,
    k_guess: Optional[np.ndarray] = None,
    max_iter_k: int = 50,
    tol_k: float = 1e-9
) -> Tuple[float, np.ndarray]:
    n = mixture.n_components
    if k_guess is None:
        k_values = calculate_wilson_k(p_pa, t_k, mixture)
    else:
        k_values = np.copy(k_guess)
    for _ in range(max_iter_k):
        y = k_values * x_comp
        if np.sum(y) > 1e-9:
            y = y / np.sum(y)
        phi_hat_v, _ = calculate_pr_fugacity_coefficients(t_k, p_pa, y, mixture)
        gamma, psat_pa = calculate_gamma_and_psat(t_k, x_comp, mixture)
        phi_sat_v = np.zeros(n)
        for i in range(n):
            phi_pure, _ = calculate_pr_fugacity_coefficients(
                t_k, psat_pa[i], np.eye(n)[i], mixture,
                is_pure_component=True, pure_component_index=i
            )
            phi_sat_v[i] = phi_pure[i] if phi_pure[i] > 1e-9 else 1.0
        poynting = calculate_poynting_factors(t_k, p_pa, psat_pa, mixture)
        safe_phi_hat_v = np.where(phi_hat_v < 1e-9, 1.0, phi_hat_v)
        numerators = gamma * phi_sat_v * psat_pa * poynting
        denominators = safe_phi_hat_v * p_pa
        k_new = numerators / np.where(np.abs(denominators) < 1e-20, 1e-20, denominators)
        if np.sqrt(np.mean(((k_new - k_values) / k_values)**2)) < tol_k:
            break
        k_values = np.clip(k_new, 1e-10, 1e10)
    f_obj = float(np.sum(k_values * x_comp) - 1.0)
    return f_obj, k_values

def calculate_bubble_temperature(
    p_psi: float,
    x_comp: np.ndarray,
    initial_t_f_guess: float,
    mixture: Mixture,
    max_iter_t: int = 50,
    tol_f_t: float = 1e-6
) -> Tuple[float, np.ndarray, np.ndarray]:
    p_pa = p_psi * 6894.76
    t_k = ((initial_t_f_guess - 32.0) * 5.0 / 9.0) + 273.15
    k_vals = None
    x = np.asarray(x_comp, dtype=float)
    for _ in range(max_iter_t):
        f_t, k_vals = _calculate_objective_f_and_k_at_t(t_k, p_pa, x, mixture, k_guess=k_vals)
        if abs(f_t) < tol_f_t:
            y = k_vals * x
            if np.sum(y) > 1e-9:
                y = y / np.sum(y)
            t_f = ((t_k - 273.15) * 9.0 / 5.0) + 32.0
            return t_f, y, k_vals
        delta_t = t_k * 0.001 or 1e-3
        f_t_plus, _ = _calculate_objective_f_and_k_at_t(t_k + delta_t, p_pa, x, mixture, k_vals)
        df_dt = (f_t_plus - f_t) / delta_t
        if abs(df_dt) < 1e-12:
            t_k -= 0.1
            continue
        t_k_next = t_k - f_t / df_dt
        if abs(t_k_next - t_k) > t_k * 0.1:
            t_k_next = t_k - np.sign(f_t / df_dt) * t_k * 0.1
        t_k = t_k_next
    y = k_vals * x if k_vals is not None else np.copy(x)
    if np.sum(y) > 1e-9:
        y = y / np.sum(y)
    t_f = ((t_k - 273.15) * 9.0 / 5.0) + 32.0
    return t_f, y, k_vals
