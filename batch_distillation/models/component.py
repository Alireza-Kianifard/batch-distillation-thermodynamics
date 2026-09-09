from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class Component:
    name: str
    mw: float
    tc: float
    pc: float
    omega: float
    v_liq: float
    antoine_coeffs: Dict[str, float]
    cpig_coeffs: List[float]
    hf_298: float = 0.0

    def calculate_psat(self, t_k: float) -> float:
        if t_k <= 0:
            raise ValueError(f"Temperature must be positive. Received: {t_k}")
        a = self.antoine_coeffs['a']
        b = self.antoine_coeffs['b']
        c = self.antoine_coeffs.get('c', 0.0)
        d = self.antoine_coeffs['d']
        e = self.antoine_coeffs['e']
        f = self.antoine_coeffs.get('f', 2.0)
        ln_psat_kpa = a + (b / (t_k + c)) + (d * np.log(t_k)) + (e * (t_k ** f))
        return float(np.exp(ln_psat_kpa) * 1000.0)

    def calculate_pure_ideal_gas_enthalpy(
        self, t_k: float, t_ref: float = 298.15, r_const: float = 8.314
    ) -> float:
        a, b, c, d, e = self.cpig_coeffs
        delta_h_ig = r_const * (
            a * (t_k - t_ref)
            + (b / 2.0) * (t_k**2 - t_ref**2)
            + (c / 3.0) * (t_k**3 - t_ref**3)
            + (d / 4.0) * (t_k**4 - t_ref**4)
            - e * (1.0 / t_k - 1.0 / t_ref)
        )
        return float(self.hf_298 + delta_h_ig)
