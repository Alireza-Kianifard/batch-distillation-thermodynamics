import numpy as np
from ..models.mixture import Mixture

def calculate_wilson_k(p_pa: float, t_k: float, mixture: Mixture) -> np.ndarray:
    if p_pa <= 0.0:
        p_pa = 1e-3
    pc = mixture.pc
    tc = mixture.tc
    omega = mixture.omega
    k_values = (pc / p_pa) * np.exp(5.37 * (1.0 + omega) * (1.0 - (tc / t_k)))
    return np.clip(k_values, 1e-12, 1e12)
