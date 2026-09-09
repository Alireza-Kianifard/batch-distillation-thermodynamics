from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .component import Component

@dataclass
class Mixture:
    components: List[Component]
    kij_matrix: Optional[np.ndarray] = None
    nrtl_tau: Optional[np.ndarray] = None
    nrtl_alpha: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.components)
        if self.kij_matrix is None:
            self.kij_matrix = np.zeros((n, n))
        else:
            self.kij_matrix = np.asarray(self.kij_matrix, dtype=float)
        if self.nrtl_tau is None:
            self.nrtl_tau = np.zeros((n, n))
        else:
            self.nrtl_tau = np.asarray(self.nrtl_tau, dtype=float)
        if self.nrtl_alpha is None:
            self.nrtl_alpha = np.zeros((n, n))
        else:
            self.nrtl_alpha = np.asarray(self.nrtl_alpha, dtype=float)

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.components]

    @property
    def tc(self) -> np.ndarray:
        return np.array([c.tc for c in self.components], dtype=float)

    @property
    def pc(self) -> np.ndarray:
        return np.array([c.pc for c in self.components], dtype=float)

    @property
    def omega(self) -> np.ndarray:
        return np.array([c.omega for c in self.components], dtype=float)

    @property
    def v_liq(self) -> np.ndarray:
        return np.array([c.v_liq for c in self.components], dtype=float)
