import numpy as np
from ..models.mixture import Mixture

R_CONST = 8.314

def calculate_ideal_gas_enthalpy(
    t_k: float, composition: np.ndarray, mixture: Mixture, t_ref: float = 298.15
) -> float:
    y = np.asarray(composition, dtype=float)
    h_pure = np.array([
        comp.calculate_pure_ideal_gas_enthalpy(t_k, t_ref=t_ref, r_const=R_CONST)
        for comp in mixture.components
    ])
    return float(np.sum(y * h_pure))

def calculate_residual_enthalpy(
    t_k: float, p_pa: float, composition: np.ndarray, z: float, mixture: Mixture
) -> float:
    y = np.asarray(composition, dtype=float)
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
    a_mix = np.sum(y[:, np.newaxis] * y * np.sqrt(a_i[:, np.newaxis] * a_i) * (1.0 - mixture.kij_matrix))
    b_mix = np.sum(y * b_i)
    b_param = b_mix * p_pa / (R_CONST * t_k)
    d_alpha_dt = -m * np.sqrt(alpha_t / (t_k * tr))
    d_ai_dt = ac_i * d_alpha_dt
    d_amix_dt = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            term_ij = y[i] * y[j] * (1.0 - mixture.kij_matrix[i, j])
            d_sqrt_ai_aj_dt = 0.5 * (a_i[i] * a_i[j])**(-0.5) * (
                d_ai_dt[i] * a_i[j] + a_i[i] * d_ai_dt[j]
            )
            d_amix_dt += term_ij * d_sqrt_ai_aj_dt
    term_a_derivative = t_k * d_amix_dt - a_mix
    log_arg_num = z + (1.0 + np.sqrt(2)) * b_param
    log_arg_den = z + (1.0 - np.sqrt(2)) * b_param
    log_term = np.log(log_arg_num / log_arg_den if log_arg_den > 1e-12 else 1e-12)
    h_r = R_CONST * t_k * (z - 1.0) + (term_a_derivative / (2.0 * np.sqrt(2) * b_mix)) * log_term
    return float(h_r)
