"""
stl_robustness.py
@author: Kasidit Muenprasitivej
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import yaml


# --------------------
# STL Operator class
# --------------------

class STLNode(ABC):
    #Abstract base class for all STL formula nodes

    @abstractmethod
    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute robustness signal rho(Phi, x, t_i) for every time index i.

        Parameters
        ----------
        t : np.ndarray, shape (T,)          : time stamps.
        x : np.ndarray, shape (T, state_dim): State trajectory sampled at times t.

        Returns
        -------
        rho : np.ndarray, shape (T,) :Robustness value at each time step.
            
        where rho[i] = rho(Phi, x, t[i])
        """

    @abstractmethod
    def horizon(self) -> float:
        """
        Compute the time horizon ||Phi|| of this STL formula node.

        Propagates recursively from children to parent:

            Predicate              : 0
            Negation (NOT phi)     : ||phi||
            Conjunction/Disjunction: max(||phi1||, ||phi2||)
            Globally/Eventually    : b + ||child||
            Until (phi U[a,b] psi) : b + max(||phi||, ||psi||)

        Returns
        -------
        float : The time horizon of this node.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


class Predicate(STLNode):
    """
    Leaf node: a real-valued predicate Mu(x(t)) > 0.

    Examples
    --------
    expression for 1D state           = "x[0] - 1.0"       -->  x0(t) - 1.0  > 0
    expression for (atleast) 3D state = "x[1] + x[2] - 3"  -->  x1(t) + x2(t) - 3  > 0
    """

    def __init__(
        self,
        expression: str,
        c_x: float | None = None,
        c_y: float | None = None,
        r:   float | None = None,
    ) -> None:
        self.expression = expression
        self.c_x = c_x
        self.c_y = c_y
        self.r   = r

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.array([
            eval(self.expression, {"np": np, "x": x[i]}) #Here we replace character 'x' in the expression string with value x[i] for given signal x at time index i
            for i in range(len(t))
        ])

    def horizon(self) -> float:
        # ||mu|| = 0  (a predicate needs only the value at the current instant)
        return 0.0

    def __repr__(self) -> str:
        return f"Predicate({self.expression!r})"


class Negation(STLNode):
    """Not Phi  -->  rho = -rho_child"""

    def __init__(self, child: STLNode) -> None:
        self.child = child

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return -self.child.robustness(t, x)

    def horizon(self) -> float:
        # ||NOT phi|| = ||phi||
        return self.child.horizon()

    def __repr__(self) -> str:
        return f"NOT({self.child!r})"


class Conjunction(STLNode):
    """Phi AND Psi  -->  rho = min(rho_left, rho_right)"""

    def __init__(self, left: STLNode, right: STLNode) -> None:
        self.left = left
        self.right = right

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.minimum(
            self.left.robustness(t, x),
            self.right.robustness(t, x),
        )

    def horizon(self) -> float:
        # ||phi1 AND phi2|| = max(||phi1||, ||phi2||)
        return max(self.left.horizon(), self.right.horizon())

    def __repr__(self) -> str:
        return f"AND({self.left!r}, {self.right!r})"


class Disjunction(STLNode):
    """Phi OR Psi  -->  rho = max(rho_left, rho_right)"""

    def __init__(self, left: STLNode, right: STLNode) -> None:
        self.left = left
        self.right = right

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.maximum(
            self.left.robustness(t, x),
            self.right.robustness(t, x),
        )

    def horizon(self) -> float:
        # ||phi1 OR phi2|| = max(||phi1||, ||phi2||)
        return max(self.left.horizon(), self.right.horizon())

    def __repr__(self) -> str:
        return f"OR({self.left!r}, {self.right!r})"


class TemporalOperator(STLNode):
    """
    TemporalOperator for Globally and Eventually.

    Given current time t[i] and interval [a,b],
    We  find robustness by...
     
        For each time step t[i] we find all indices j such that
            t[i] + a < t[j] < t[i] + b
        and aggregate robustness with min (G) or max (F) over the time window
    """

    def __init__(self, a: float, b: float, child: STLNode) -> None:
        if a < 0 or b < a:
            raise ValueError(f"Invalid interval [{a}, {b}]: need 0 <= a <= b.")
        self.a = a
        self.b = b
        self.child = child

    def window_indices(self, t: np.ndarray, i: int) -> np.ndarray:
        """Return sorted array of indices j s.t. t[i]+a < t[j] < t[i]+b."""
        
        lo = t[i] + self.a
        hi = t[i] + self.b
        
        return np.where((t >= lo - 1e-12) & (t <= hi + 1e-12))[0]

    @abstractmethod
    def aggregate(self, values: np.ndarray) -> float:
        """min (G) or max (F) aggregation."""

    @property
    @abstractmethod
    def _empty_value(self) -> float:
        """Value returned when the window is empty."""

    def horizon(self) -> float:
        # ||G[a,b] phi|| = ||F[a,b] phi|| = b + ||child||
        return self.b + self.child.horizon()

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        
        #Robustness value of the entire trajectory based on the children operator
        child_rho = self.child.robustness(t, x)
        
        rho = np.empty(len(t))
        for i in range(len(t)):
            
            #convert t[i] into interval [t[i] + a, t[i] + b] with correct indices
            idx = self.window_indices(t, i)
            
            if len(idx) == 0:
                
                # If the window contains no valid indices (e.g. near the end of the signal), then return inf
                rho[i] = self._empty_value
            else:
                
                #apply min (G) or max (F) over interval [a,b]
                rho[i] = self.aggregate(child_rho[idx])
        return rho


class Globally(TemporalOperator):
    """G[a,b]Phi  -->  rho(t) = min_{t' in [t+a, t+b]} rho_child(t')"""

    def aggregate(self, values: np.ndarray) -> float:
        return float(np.min(values))

    @property
    def _empty_value(self) -> float:
        return math.inf  # vacuously true

    def __repr__(self) -> str:
        return f"G[{self.a},{self.b}]({self.child!r})"


class Eventually(TemporalOperator):
    """F[a,b]Phi  -->  rho(t) = max_{t' in [t+a, t+b]} rho_child(t')"""

    def aggregate(self, values: np.ndarray) -> float:
        return float(np.max(values))

    @property
    def _empty_value(self) -> float:
        return -math.inf  # vacuously false

    def __repr__(self) -> str:
        return f"F[{self.a},{self.b}]({self.child!r})"


class Release(STLNode):
    """
    Phi R[a,b] Psi  —  dual of Until, arises when negation is pushed through U.

    Semantics: Psi must hold throughout [a,b] UNLESS Phi becomes true first,
               in which case both must hold at that moment.

    Robustness at time t[i]:

        rho(t[i]) = min over j in window(i) of:
                        max( rho_phi(t[j]),
                             max over k in [i, j] of rho_psi(t[k]) )

    This is the exact min/max dual of Until.

    Empty outer window  --> +inf  (vacuously true, dual of Until's -inf)
    """

    def __init__(self, a: float, b: float, phi: STLNode, psi: STLNode) -> None:
        if a < 0 or b < a:
            raise ValueError(f"Invalid interval [{a}, {b}]: need 0 <= a <= b.")
        self.a   = a
        self.b   = b
        self.phi = phi   # release trigger  (if it holds, both must hold)
        self.psi = psi   # release consequent (must hold until trigger or end)

    def horizon(self) -> float:
        # ||phi R[a,b] psi|| = b + max(||phi||, ||psi||)  (same structure as Until)
        return self.b + max(self.phi.horizon(), self.psi.horizon())

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        phi_rho = self.phi.robustness(t, x)   # shape (T,)
        psi_rho = self.psi.robustness(t, x)   # shape (T,)

        rho = np.empty(len(t))

        for i in range(len(t)):
            lo = t[i] + self.a
            hi = t[i] + self.b

            witness_idx = np.where(
                (t >= lo - 1e-12) & (t <= hi + 1e-12)
            )[0]

            if len(witness_idx) == 0:
                rho[i] = math.inf    # vacuously true (dual of Until's -inf)
                continue

            worst = math.inf
            for j in witness_idx:
                # psi must hold on [t[i], t[j]]  -->  indices i..j inclusive
                psi_hold = float(np.max(psi_rho[i : j + 1]))
                candidate = max(phi_rho[j], psi_hold)
                if candidate < worst:
                    worst = candidate

            rho[i] = worst

        return rho

    def __repr__(self) -> str:
        return f"R[{self.a},{self.b}]({self.phi!r}, {self.psi!r})"


class Until(STLNode):
    """
    Phi U[a,b] Psi

    Robustness at time t[i]:

        rho(t[i]) = max over j in window(i) of:
                        min( rho_psi(t[j]),
                             min over k in [i, j] of rho_phi(t[k]) )

    Reading this in plain English:
      - Psi must become true at some time t[j] within [t[i]+a, t[i]+b].
      - Phi must hold continuously from t[i] up to (and including) t[j].
      - We pick the j that maximises the resulting robustness (best-case witness).

    Empty outer window  --> -inf  (no witness, vacuously false)
    """

    def __init__(self, a: float, b: float, phi: STLNode, psi: STLNode) -> None:
        if a < 0 or b < a:
            raise ValueError(f"Invalid interval [{a}, {b}]: need 0 <= a <= b.")
        self.a   = a
        self.b   = b
        self.phi = phi   # the "until" antecedent  (must hold up to witness)
        self.psi = psi   # the "until" consequent  (must hold at witness)

    def horizon(self) -> float:
        # ||phi U[a,b] psi|| = b + max(||phi||, ||psi||)
        return self.b + max(self.phi.horizon(), self.psi.horizon())

    def robustness(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        phi_rho = self.phi.robustness(t, x)   # shape (T,)
        psi_rho = self.psi.robustness(t, x)   # shape (T,)

        rho = np.empty(len(t))

        for i in range(len(t)):
            lo = t[i] + self.a
            hi = t[i] + self.b

            # All candidate witness indices j: t[j] in [t[i]+a, t[i]+b]
            witness_idx = np.where(
                (t >= lo - 1e-12) & (t <= hi + 1e-12)
            )[0]

            if len(witness_idx) == 0:
                rho[i] = -math.inf   # no witness available --> false
                continue

            best = -math.inf
            for j in witness_idx:
                # phi must hold on [t[i], t[j]]  -->  indices i..j inclusive
                phi_hold = float(np.min(phi_rho[i : j + 1]))
                # witness robustness: how well does psi hold at j,
                # constrained by how well phi held the whole way there
                candidate = min(psi_rho[j], phi_hold)
                if candidate > best:
                    best = candidate

            rho[i] = best

        return rho

    def __repr__(self) -> str:
        return f"U[{self.a},{self.b}]({self.phi!r}, {self.psi!r})"


# --------------------
# YAML Parser
# --------------------

def parse_stl_yaml(path: str, pnf: bool = False) -> STLNode:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return _parse_node(cfg, pnf=pnf)


def _parse_node(cfg: dict, pnf: bool = False) -> STLNode:
    """
    Recursively convert a config dict into an STLNode.

    Parameters
    ----------
    cfg : YAML node dictionary.
    pnf : If True, return the formula in Positive Normal Form.
          When a NOT node is encountered, we check its child:
            - Child is PREDICATE  → negation already sits on a leaf, nothing to fix.
                                    Return Negation(Predicate) as-is.
            - Child is NOT        → double negation, strip both and recurse into grandchild.
            - Child is AND        → De Morgan: NOT(φ AND ψ) → (NOT φ) OR  (NOT ψ)
            - Child is OR         → De Morgan: NOT(φ OR  ψ) → (NOT φ) AND (NOT ψ)
            - Child is G          → temporal duality: NOT G[a,b] φ → F[a,b] (NOT φ)
            - Child is F          → temporal duality: NOT F[a,b] φ → G[a,b] (NOT φ)
            - Child is U          → Until/Release duality:
                                    NOT (φ U[a,b] ψ) → (NOT ψ) R[a,b] (NOT φ)
            - Child is R          → Release/Until duality:
                                    NOT (φ R[a,b] ψ) → (NOT ψ) U[a,b] (NOT φ)
    """

    node_type = cfg.get("type", "").upper()

    if node_type == "PREDICATE":
        expr = cfg.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("Predicate node requires a non-empty 'expression' string.")
        return Predicate(
            expression = expr,
            c_x        = cfg.get("c_x"),
            c_y        = cfg.get("c_y"),
            r          = cfg.get("r"),
        )

    if node_type == "NOT":
        child_cfg   = cfg["child"]
        child_type  = child_cfg.get("type", "").upper()

        # Negation already on a predicate leaf — nothing to fix, return as-is
        if not pnf or child_type == "PREDICATE":
            return Negation(_parse_node(child_cfg, pnf=pnf))

        # Double negation: NOT NOT φ  →  φ
        if child_type == "NOT":
            return _parse_node(child_cfg["child"], pnf=True)

        # De Morgan: NOT (φ AND ψ)  →  (NOT φ) OR (NOT ψ)
        if child_type == "AND":
            return Disjunction(
                _parse_node({"type": "NOT", "child": child_cfg["left"]},  pnf=True),
                _parse_node({"type": "NOT", "child": child_cfg["right"]}, pnf=True),
            )

        # De Morgan: NOT (φ OR ψ)  →  (NOT φ) AND (NOT ψ)
        if child_type == "OR":
            return Conjunction(
                _parse_node({"type": "NOT", "child": child_cfg["left"]},  pnf=True),
                _parse_node({"type": "NOT", "child": child_cfg["right"]}, pnf=True),
            )

        # Temporal duality: NOT G[a,b] φ  →  F[a,b] (NOT φ)
        if child_type == "G":
            a, b = require_interval(child_cfg)
            return Eventually(
                a, b,
                _parse_node({"type": "NOT", "child": child_cfg["child"]}, pnf=True),
            )

        # Temporal duality: NOT F[a,b] φ  →  G[a,b] (NOT φ)
        if child_type == "F":
            a, b = require_interval(child_cfg)
            return Globally(
                a, b,
                _parse_node({"type": "NOT", "child": child_cfg["child"]}, pnf=True),
            )

        # Until/Release duality: NOT (φ U[a,b] ψ)  →  (NOT ψ) R[a,b] (NOT φ)
        if child_type == "U":
            a, b = require_interval(child_cfg)
            return Release(
                a, b,
                phi=_parse_node({"type": "NOT", "child": child_cfg["psi"]}, pnf=True),
                psi=_parse_node({"type": "NOT", "child": child_cfg["phi"]}, pnf=True),
            )

        # Release/Until duality: NOT (φ R[a,b] ψ)  →  (NOT ψ) U[a,b] (NOT φ)
        if child_type == "R":
            a, b = require_interval(child_cfg)
            return Until(
                a, b,
                phi=_parse_node({"type": "NOT", "child": child_cfg["psi"]}, pnf=True),
                psi=_parse_node({"type": "NOT", "child": child_cfg["phi"]}, pnf=True),
            )

    if node_type == "AND":
        return Conjunction(
            require_child(cfg, "left",  pnf=pnf),
            require_child(cfg, "right", pnf=pnf),
        )

    if node_type == "OR":
        return Disjunction(
            require_child(cfg, "left",  pnf=pnf),
            require_child(cfg, "right", pnf=pnf),
        )

    if node_type == "G":
        a, b = require_interval(cfg)
        return Globally(a, b, require_child(cfg, "child", pnf=pnf))

    if node_type == "F":
        a, b = require_interval(cfg)
        return Eventually(a, b, require_child(cfg, "child", pnf=pnf))

    if node_type == "U":
        a, b = require_interval(cfg)
        return Until(
            a, b,
            phi=require_child(cfg, "phi", pnf=pnf),
            psi=require_child(cfg, "psi", pnf=pnf),
        )

    if node_type == "R":
        a, b = require_interval(cfg)
        return Release(
            a, b,
            phi=require_child(cfg, "phi", pnf=pnf),
            psi=require_child(cfg, "psi", pnf=pnf),
        )

    raise ValueError(
        f"Unknown node type: {cfg.get('type')!r}.  "
        f"Supported: predicate, NOT, AND, OR, G, F, U, R."
    )


def require_child(cfg: dict, key: str, pnf: bool = False) -> STLNode:
    if key not in cfg:
        raise ValueError(f"Node of type '{cfg.get('type')}' requires a '{key}' sub-node.")
    return _parse_node(cfg[key], pnf=pnf)


def require_interval(cfg: dict) -> tuple[float, float]:
    iv = cfg.get("interval")
    if iv is None or len(iv) != 2:
        raise ValueError(
            f"Node of type '{cfg.get('type')}' requires an 'interval: [a, b]' key."
        )
    a, b = float(iv[0]), float(iv[1])
    return a, b


# --------------------
# Horizon evaluator
# --------------------

class STLHorizonEvaluator:
    """Returns the time horizon ||Phi|| of a parsed STL formula."""

    def __init__(self, path: str) -> None:
        self.formula = parse_stl_yaml(path)

    def evaluate(self) -> float:
        return self.formula.horizon()

    def __repr__(self) -> str:
        return f"STLHorizonEvaluator(formula={self.formula!r}, horizon={self.evaluate():.4f})"


# --------------------
# Robustness eval
# --------------------

class STLRobustnessEvaluator:
    """
    a parsed STL formula and evaluates robustness for arbitrary (t, x).
    """

    def __init__(self, path: str, pnf: bool = False) -> None:
        self.formula = parse_stl_yaml(path, pnf=pnf)

    # --------------------
    # Evaluation
    # --------------------

    def evaluate(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute the robustness signal rho(Phi, x, t_i) for every time index i.
        """
        t, x = self.validate(t, x) # Make sure the dimension is formatted correctly
        return self.formula.robustness(t, x)

    def evaluate_at(self, t: np.ndarray, x: np.ndarray, t0: float = 0.0) -> float:
        """
        Return robustness at a single time instant t0.
        """
        t, x = self.validate(t, x)
        idx = int(np.argmin(np.abs(t - t0)))
        return float(self.formula.robustness(t, x)[idx])

    def horizon(self) -> float:
        """
        Return the time horizon ||Phi|| of the loaded STL formula.
        This is the minimum signal duration required to evaluate the formula.
        """
        return self.formula.horizon()

    #For  Making sure the dimension is formatted correctly
    @staticmethod
    def validate(t: np.ndarray, x: np.ndarray):

        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if t.ndim != 1:
            raise ValueError("'t' must be a 1-D array.")
        if x.ndim != 2:
            raise ValueError("'x' must be a 2-D array of shape (T, state_dim).")
        if len(t) != x.shape[0]:
            raise ValueError(
                f"Length mismatch: t has {len(t)} steps, x has {x.shape[0]}."
            )
        return t, x

    def __repr__(self) -> str:
        return f"STLRobustnessEvaluator(formula={self.formula!r})"


# --------------------
# Debug
# --------------------

if __name__ == "__main__":

    # Time array
    tf = 10
    t0 = 0
    dt = 0.1
    t    = t = np.linspace(t0, tf, int((tf - t0) / dt + 1))
        
    # 1D Signal
    x  = np.sin(t) + 0.5 * np.cos(2*t)
    
    # -------------------------------
    # Spec: G[0,10] F[0,5] NOT (x[0] >= 1.0 ∧ x[0] <=  2.5)                     
    # -------------------------------

    evaluator1 = STLRobustnessEvaluator("stl_README/stl_nested_spec_example.yaml", pnf=True)
    rho1 = evaluator1.evaluate(t, x)

    print("=" * 60)
    print(f"  Formula  : {evaluator1.formula}")
    print(f"  Horizon  : {evaluator1.horizon()}")
    print(f"  rho(x,Phi1,0) : {rho1[0]} ")
    print(f"  rho(x,Phi1,t) for t=0,1,2...,10 : {rho1[::int(1/dt)]}")
    