from typing import Dict, Any
import numpy as np
from ..models.mixture import Mixture
from ..core.eos_pr import solve_pr_eos_for_z
from ..core.enthalpy import calculate_residual_enthalpy
from ..solvers.bubble_point import calculate_bubble_temperature

def simulate_batch_distillation(
    initial_moles: float,
    initial_comp: np.ndarray,
    pressure_psi: float,
    heat_duty_watt: float,
    total_time_sec: float,
    dt_sec: float,
    mixture: Mixture
) -> Dict[str, Any]:
    current_moles_l = float(initial_moles)
    current_comp_x = np.asarray(initial_comp, dtype=float)
    current_time_sec = 0.0
    total_vapor_out = 0.0

    history: Dict[str, list] = {
        "time_sec": [],
        "temp_C": [],
        "moles_in_still": [],
        "total_vapor_out": [],
        "liquid_comp": [],
        "vapor_comp": [],
        "vapor_rate_mols": [],
    }

    last_temp_f = 208.0

    while current_time_sec <= total_time_sec:
        t_bubble_f, y_vapor, _ = calculate_bubble_temperature(
            pressure_psi, current_comp_x, last_temp_f, mixture
        )
        last_temp_f = t_bubble_f
        t_k = ((t_bubble_f - 32.0) * 5.0 / 9.0) + 273.15
        p_pa = pressure_psi * 6894.76

        z_l, _ = solve_pr_eos_for_z(t_k, p_pa, current_comp_x, mixture)
        _, z_v = solve_pr_eos_for_z(t_k, p_pa, y_vapor, mixture)

        if z_l is None or z_v is None:
            break

        h_r_liquid = calculate_residual_enthalpy(t_k, p_pa, current_comp_x, z_l, mixture)
        h_r_vapor = calculate_residual_enthalpy(t_k, p_pa, y_vapor, z_v, mixture)

        delta_h_vap = h_r_vapor - h_r_liquid
        if delta_h_vap <= 0.0:
            break

        vapor_rate_per_sec = heat_duty_watt / delta_h_vap

        history["time_sec"].append(current_time_sec)
        history["temp_C"].append((t_bubble_f - 32.0) * 5.0 / 9.0)
        history["moles_in_still"].append(current_moles_l)
        history["liquid_comp"].append(np.copy(current_comp_x))
        history["vapor_comp"].append(np.copy(y_vapor))
        history["total_vapor_out"].append(total_vapor_out)
        history["vapor_rate_mols"].append(vapor_rate_per_sec * 3600.0)

        moles_vaporized_in_step = vapor_rate_per_sec * dt_sec
        if moles_vaporized_in_step > current_moles_l:
            moles_vaporized_in_step = current_moles_l

        current_moles_l -= moles_vaporized_in_step
        total_vapor_out += moles_vaporized_in_step

        if current_moles_l < 1e-9:
            break

        moles_comp_old = current_comp_x * (current_moles_l + moles_vaporized_in_step)
        moles_comp_new = moles_comp_old - (y_vapor * moles_vaporized_in_step)
        current_comp_x = moles_comp_new / current_moles_l

        current_time_sec += dt_sec

    return history
