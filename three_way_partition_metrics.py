import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ThreeWayPartition:
    core_indices    : List[List[int]]
    fringe_indices  : List[List[int]]
    trivial_indices : List[int]
    labels          : np.ndarray
    n_samples       : int
    n_clusters      : int
    algorithm       : str = "unknown"

    _core_set   : Optional[set]        = field(default=None, repr=False, compare=False)
    _fringe_set : Optional[set]        = field(default=None, repr=False, compare=False)
    _fringe_count: Optional[np.ndarray]= field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=int)
        self.trivial_indices = list(self.trivial_indices)

    @property
    def core_set(self) -> set:
        """Unique set of all core samples (disjoint across clusters)."""
        if self._core_set is None:
            s = set()
            for lst in self.core_indices:
                s.update(lst)
            self._core_set = s
        return self._core_set

    @property
    def fringe_set(self) -> set:
        """
        Unique set of all fringe samples, excluding core members.
        Ensures phi_core + phi_fringe + phi_trivial = 1.
        """
        if self._fringe_set is None:
            s = set()
            for lst in self.fringe_indices:
                s.update(lst)
            self._fringe_set = s - self.core_set
        return self._fringe_set

    @property
    def fringe_membership_count(self) -> np.ndarray:
        """
        shape (n_samples,): number of clusters whose fringe region contains each sample.
        """
        if self._fringe_count is None:
            cnt = np.zeros(self.n_samples, dtype=int)
            for lst in self.fringe_indices:
                for i in lst:
                    if 0 <= i < self.n_samples:
                        cnt[i] += 1
            self._fringe_count = cnt
        return self._fringe_count

def compute_threeway_metrics(
    partition : ThreeWayPartition,
    y_true    : np.ndarray,
) -> dict:
    """
    Compute CFSI and TPQI for a three-way clustering partition.

    Parameters
    ----------
    partition : ThreeWayPartition
    y_true    : np.ndarray, shape (n_samples,), ground-truth labels

    Returns
    -------
    dict with keys 'CFSI' and 'TPQI', both in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    N  = partition.n_samples
    K  = partition.n_clusters

    core_set   = partition.core_set
    fringe_set = partition.fringe_set

    n_core   = len(core_set)
    n_fringe = len(fringe_set)

    # ── CP: core purity (Eq. 36) ─────────────────────────────────────────
    # CP = (1 / n_core) * sum_i max_z |{x in Core(C_i) : l(x) = z}|
    if n_core == 0:
        CP = float('nan')
    else:
        cp_sum = 0
        for k in range(K):
            c_idx = partition.core_indices[k]
            if not c_idx:
                continue
            _, counts = np.unique(y_true[np.array(c_idx)], return_counts=True)
            cp_sum += int(counts.max())
        CP = cp_sum / n_core

    # ── FA: fringe ambiguity (Eq. 37) ────────────────────────────────────
    # FA = 1 - (1 / n_fringe) * sum_i max_z |{x in C_i^+/-: l(x) = z}|
    # n_fringe counts fringe assignments with multiplicity (Sigma_i |C_i^+/-|).
    n_fringe_mult = sum(len(lst) for lst in partition.fringe_indices)

    if n_fringe_mult == 0:
        FA = float('nan')
    else:
        fp_sum = 0
        for k in range(K):
            f_idx = partition.fringe_indices[k]
            if not f_idx:
                continue
            _, counts = np.unique(y_true[np.array(f_idx)], return_counts=True)
            fp_sum += int(counts.max())
        FP = fp_sum / n_fringe_mult
        FA = 1.0 - FP

    # ── CFSI: core-fringe stratification index (Eq. 38) ──────────────────
    # CFSI = (CP + FA) / 2
    # Degenerate case (n_fringe_mult == 0): no fringe region, FA undefined.
    # CFSI is set to 0 as a sentinel value, strictly below the neutral
    if np.isnan(CP):
        CFSI = float('nan')
    elif n_fringe_mult == 0:
        CFSI = 0.0
    else:
        CFSI = (CP + FA) / 2.0

    # ── TPQI: three-way partition quality index (Eq. 39) ─────────────────
    # TPQI = CP * CR + FA * (1 - CR),  CR = n_core / |U|
    # Degenerate case (CR = 1, no fringe): TPQI reduces to CP.
    CR = n_core / N

    if np.isnan(CP):
        TPQI = float('nan')
    elif n_fringe_mult == 0:
        # FA undefined; FA * (1 - CR) term vanishes only when CR = 1.
        # If trivial samples exist (CR < 1), TPQI is not well-defined.
        if abs(1.0 - CR) < 1e-9:
            TPQI = CP
        else:
            warnings.warn(
                f"[three_way_metrics | {partition.algorithm}] "
                f"No fringe region but CR = {CR:.4f} < 1 (trivial samples exist). "
                f"TPQI is set to CP as an approximation.",
                stacklevel=2,
            )
            TPQI = CP
    elif np.isnan(FA):
        TPQI = float('nan')
    else:
        TPQI = CP * CR + FA * (1.0 - CR)

    return {
        'CFSI': CFSI,
        'TPQI': TPQI,
    }
