"""
main.py
Online control synthesis under STL specifications using tTLT + HJ Reachability.

Reference:
    Yu et al., 2023 — Online control synthesis for uncertain systems under
    signal temporal logic specifications
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
import io
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")          # must be set before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

from scipy.interpolate import RegularGridInterpolator

# ── HJ Reachability (optimized_dp) ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "optimized_dp"))

from odp.Grid    import Grid
from odp.solver  import computeSpatDerivArray
from odp.dynamics import DubinsCar2, Plane2D

# ── STL robustness / horizon ──────────────────────────────────────────────────
from stl_robustness import STLHorizonEvaluator, STLNode, Predicate

# ── tTLT node types ───────────────────────────────────────────────────────────
from tTLT_nodes import (
    TubeNode,
    AndOperatorNode,
    OrOperatorNode,
    GloballyOperatorNode,
    EventuallyOperatorNode,
    UntilOperatorNode,
    plot_ttlt_tree,
    _ttlt_collect_layout,
    Post,
    Pre,
    Children,
)

# ── tTLT HJ solver ────────────────────────────────────────────────────────────
from stl_hj import (
    HJtTLT,
    tTLT,
)

# ── tTLT synthesis algorithms ─────────────────────────────────────────────────
from tTLT_synthesis import (
    get_MTS,
    get_all_tubes,
    Compression,
    Initialization,
    trackingSetNode,
    updatetTLT,
    buildControlTree,
    Backtracking,
    postSet,
    get_spat_deriv_at_state,
    print_MTS,
    print_compressed_tree,
    plot_compressed_reachable_sets,
)



from HJR_FNO.HJR_FNO2d import FNO2d, SpectralConv2d


# ==============================================================================
# LIDAR detection
# ==============================================================================

def collect_predicates(stl_root: STLNode, ttlt_root: TubeNode) -> list[Predicate]:
    """
    DFS traversal of an STLNode formula tree.
    Returns only Predicate leaf nodes whose corresponding TubeNode in the
    tTLT tree is a leaf (tube.operator is None).

    A phi-of-Until predicate owns a UntilOperatorNode after solve_until, so
    it is excluded.  A psi-of-Until or a G/F child predicate has no operator
    below it, so it is included.

    Parameters
    ----------
    stl_root  : root of the STLNode formula tree  (solver.formula)
    ttlt_root : root of the wired tTLT TubeNode tree  (solver.root)
    """
    found: list[Predicate] = []
    stack: list[STLNode] = [stl_root]
    while stack:
        node = stack.pop()
        if isinstance(node, Predicate):
            tube = find_tube_in_tree(ttlt_root, repr(node))
            if tube is not None and tube.operator is None:
                found.append(node)
        for attr in ("child", "left", "right", "phi", "psi"):
            child = getattr(node, attr, None)
            if child is not None:
                stack.append(child)
    return found


def plot_predicate_offsets(
    ax,
    predicates: list[Predicate],
    r_2: float,
    N: int = 12,
    true_centers: dict | None = None,
    detected: set | None = None,
    color: str = "green",
    alpha: float = 0.25,
    linewidth: float = 1.0,
) -> None:
    """
    For each Predicate with (c_x, c_y, r):

    - Undetected: draw N dashed low-opacity circles of radius r whose centers
      lie on a ring of radius r_2 around the nominal predicate center, showing
      the uncertainty region.

    - Detected (index in `detected`): draw a single solid magenta circle of
      radius r at the true center stored in `true_centers[i]`.

    Parameters
    ----------
    ax           : Matplotlib Axes to draw on.
    predicates   : Output of collect_predicates(solver.formula, solver.root).
    r_2          : Offset ring radius (uncertainty radius).
    N            : Number of offset circles per undetected predicate.
    true_centers : Dict mapping predicate index → (true_cx, true_cy).
    detected     : Set of predicate indices that have been LIDAR-detected.
    color        : Line color for undetected offset circles.
    alpha        : Opacity for undetected offset circles.
    linewidth    : Line width for all circles.
    """
    true_centers = true_centers or {}
    detected     = detected     or set()

    theta = np.linspace(0, 2 * np.pi, 360)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for i, pred in enumerate(predicates):
        if pred.c_x is None or pred.c_y is None or pred.r is None:
            continue

        if i in detected and i in true_centers:
            # ── Detected: draw true circle in solid magenta ──────────────────
            tc_x, tc_y = true_centers[i]
            ax.plot(
                tc_x + pred.r * cos_t,
                tc_y + pred.r * sin_t,
                color="magenta", alpha=1.0,
                linewidth=linewidth + 0.5, linestyle="-",
            )
        else:
            # ── Undetected: draw N dashed low-opacity offset circles ─────────
            for k in range(N):
                phi_k = 2 * np.pi * k / N
                cx_k  = pred.c_x + r_2 * np.cos(phi_k)
                cy_k  = pred.c_y + r_2 * np.sin(phi_k)
                ax.plot(
                    cx_k + pred.r * cos_t,
                    cy_k + pred.r * sin_t,
                    color=color, alpha=alpha,
                    linewidth=linewidth, linestyle="--",
                )


def find_tube_in_tree(root: TubeNode, formula_repr: str) -> TubeNode | None:
    """
    DFS through the tTLT tree (TubeNode → OperatorNode → TubeNode …) and
    return the first TubeNode whose .formula matches formula_repr, or None.

    Uses the wired tree structure (TubeNode.operator / OperatorNode children),
    NOT _predicate_cache, so the returned node has correct .parent pointers.

    Naming conventions handled
    --------------------------
    G / F / psi-of-Until  : formula == repr(pred)                (exact)
    phi-of-Until          : formula == "phi(U[a,b])(repr(pred))" (startswith "phi(")
    psi fallback          : formula == "copy_N(repr(pred))"      (startswith "copy_")

    The endswith check is intentionally restricted to "phi(" and "copy_" prefixes
    so that operator-tube formulas like "G[a,b](repr(pred))" are NOT matched here —
    those would be matched prematurely and prevent the DFS from reaching the true
    predicate leaf below.
    """
    wrapped = f"({formula_repr})"   # suffix pattern for Until naming conventions

    def _matches(formula: str) -> bool:
        if formula == formula_repr:
            return True
        if formula.endswith(wrapped) and (
            formula.startswith("phi(") or formula.startswith("copy_")
        ):
            return True
        return False

    stack:   list[TubeNode] = [root]
    visited: set[int]       = set()
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if _matches(node.formula):
            return node
        op = node.operator
        if op is None:
            continue
        if isinstance(op, (AndOperatorNode, OrOperatorNode)):
            stack.extend([op.left, op.right])
        elif isinstance(op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
            stack.append(op.child)
    return None


def lidar_detect(
    state: np.ndarray,
    obstacles: list[dict],
    lidar_range: float,
) -> list[dict]:
    """
    Detect circular obstacles whose center falls within the LIDAR sensing circle.

    Detection criterion:
        ||robot_pos - obstacle_center||_2  <=  lidar_range

    Parameters
    ----------
    state : np.ndarray, shape (2,) or larger
        Current robot position; only the first two elements [x, y] are used.
    obstacles : list of dict
        Each entry describes one unknown circular obstacle and must contain:
            'center' : array-like (cx, cy)  — center of the circle
            'radius' : float                — radius of the circle
    lidar_range : float
        Sensing radius of the LIDAR.

    Returns
    -------
    list of dict
        Subset of `obstacles` whose centers lie within `lidar_range` of the robot.
    """
    robot_pos = np.asarray(state[:2])
    detected = []
    for obs in obstacles:
        center = np.asarray(obs["center"])
        if np.linalg.norm(robot_pos - center) <= lidar_range:
            detected.append(obs)
    return detected


def _draw_ttlt_tree(ax_t, active_labels: set):
        ax_t.clear()
        positions, labels, kinds, edges, _ = _ttlt_collect_layout(solver.root)

        # ── bounding box + adaptive sizing ───────────────────────────────────
        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        x_rng = max(all_x) - min(all_x) or 1.0
        y_rng = max(all_y) - min(all_y) or 1.0
        x_pad = max(1.5, x_rng * 0.12)
        y_pad = max(1.5, y_rng * 0.12)          # extra top/bottom room
        n     = len(positions)
        s     = max(750, min(900, 5000 // n))   # node marker area
        fst   = max(10,   min(9,   70   // n))   # tube font size
        fso   = max(6,   min(7,   60   // n))   # operator font size

        # ── edges ─────────────────────────────────────────────────────────────
        for src_id, dst_id in edges:
            x0, y0 = positions[src_id]; x1, y1 = positions[dst_id]
            ax_t.plot([x0, x1], [y0, y1], color="#888888", lw=1.0, zorder=1)

        # ── nodes ─────────────────────────────────────────────────────────────
        for nid, k in kinds.items():
            px, py = positions[nid]; lbl = labels[nid]
            if k == "tube":
                fc = "orange" if lbl in active_labels else "#E8E8E8"
                ax_t.scatter(px, py, s=s, c=fc, edgecolors="#555555", lw=1.2, zorder=2, marker="o")
                ax_t.text(px, py, lbl, ha="center", va="center", fontsize=fst, fontweight="bold", zorder=3)
            else:
                ax_t.scatter(px, py, s=s, c="#C8D8F0", edgecolors="#335599", lw=1.2, zorder=2, marker="D")
                ax_t.text(px, py, lbl, ha="center", va="center", fontsize=fso, color="#223366", zorder=3)

        ax_t.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        ax_t.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
        ax_t.axis("off")
        ax_t.set_title("tTLT Tree  (orange = active B(x,t))", fontsize=9)
        plt.tight_layout()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    # ── Parameters ────────────────────────────────────────────────────────────
    t  = 0.0              # initial time
    x  = np.array([-1, 0.8])#np.array([0.5, 0.8])   # initial state
    dt = 0.5              # time step for reachable set computation

    use_uncertainty = False  # False → predicates known exactly (no LIDAR, no offset plot)

    lidar_range = 2.0     # LIDAR sensing radius
    r_2         = 0.7     # offset ring radius for predicate visualisation
    N_offset    = 6       # number of offset circles per predicate

    # ── Step 1: Build tTLT and solve all reachable sets ───────────────────────

    # Plane2D  :  F[5,10](G[0,10] mu_1)  AND  (mu_2 U[0,8] mu_3)
    stl_yaml_path  = "configs/stl_spec_hj_3.yaml"
    hj_config_path = "configs/hj_config.yaml"

    result = tTLT(
        stl_yaml_path  = stl_yaml_path,
        hj_config_path = hj_config_path,
    )
    solver = HJtTLT(result, hj_config_path)
    solver.FNO_enable = False
    solver.solve_all(dt=dt, accuracy="medium", plot=False, verbose=False)
    stl_horizon  = STLHorizonEvaluator(stl_yaml_path).evaluate()

    print(stl_horizon)
    predicates   = collect_predicates(solver.formula, solver.root)

    # # DubinsCar  :  (mu_1 AND NOT mu_2) U[0,12] G[0,8](mu_3)
    # result = tTLT(
    #     stl_yaml_path  = "configs/stl_spec_dubins.yaml",
    #     hj_config_path = "configs/hj_config_dubins.yaml",
    # )
    # solver = HJtTLT(result, "configs/hj_config_dubins.yaml")
    # solver.solve_all(dt=1, accuracy="medium", plot=False, verbose=False)
    # stl_horizon = STLHorizonEvaluator("configs/stl_spec_dubins.yaml").evaluate()

    # ── Step 2: MTS decomposition ─────────────────────────────────────────────
    maxTempSeg = get_MTS(solver.root)
    print_MTS(maxTempSeg)

    # ── Step 3: GIF setup ─────────────────────────────────────────────────────
    gif_frames: list[Image.Image] = []
    gif_path   = "tube_traversal.gif"
    save_gif   = True

    # ── Step 4: Plot setup ────────────────────────────────────────────────────
    fig, (ax, ax_tree) = plt.subplots(1, 2, figsize=(16, 7))
    state_history = [tuple(x)]
    plt.ion()

    g        = solver.grid
    x_coords = g.grid_points[0]
    y_coords = g.grid_points[1]
    extent   = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
    X, Y     = np.meshgrid(x_coords, y_coords)

    brs_colors  = ["lightblue", "gold", "lightgreen", "lightsalmon", "plum"]
    line_colors = ["blue", "goldenrod", "darkgreen", "darkorange", "purple"]

    # ── Assign one random offset center per predicate as its "true" center ────
    pred_true_centers: dict[int, tuple[float, float]] = {}
    pred_detected:     set[int]                        = set()
    if use_uncertainty:
        rng = np.random.default_rng()
        for i, pred in enumerate(predicates):
            if pred.c_x is None or pred.c_y is None or pred.r is None:
                continue
            k     = int(rng.integers(0, N_offset))
            phi_k = 2 * np.pi * k / N_offset
            pred_true_centers[i] = (
                pred.c_x + r_2 * np.cos(phi_k),
                pred.c_y + r_2 * np.sin(phi_k),
            )

    # ── Initial plot (t = 0, before the loop) ─────────────────────────────────
    brs      = solver.root.sdf
    z        = brs[:, :, 0] if brs.ndim == 3 else brs
    z_plot   = z.T
    z_masked = np.where(z_plot <= 0, 0.0, np.nan)

    ax.imshow(
        z_masked,
        extent=extent, origin="lower", aspect="auto",
        cmap=ListedColormap([brs_colors[-1]]),
        vmin=-0.5, vmax=0.5,
        alpha=0.4, interpolation="nearest",
    )
    ax.contour(X, Y, z_plot, levels=[0.0],
               colors=line_colors[-1], linewidths=2, label="BRS t=0")

    # predicate boundaries
    for entry in solver._predicate_cache.values():
        sdf_2d = entry["sdf"][:, :, 0] if entry["sdf"].ndim == 3 else entry["sdf"]
        ax.contour(X, Y, sdf_2d.T, levels=[0.0],
                   colors="green", linewidths=1.5, linestyles="--")

    # debug: BRS contours for specific tubes
    for tube in get_all_tubes(solver.root):
        if tube.label in {"X7", "X8"}:
            z = (tube.sdf[:, :, 0] if tube.sdf.ndim == 3 else tube.sdf).T
            ax.contour(X, Y, z, levels=[0.0], linewidths=2.0, label=tube.label)

    # predicate offset circles
    if use_uncertainty:
        plot_predicate_offsets(ax, predicates, r_2=r_2, N=N_offset,
                               true_centers=pred_true_centers, detected=pred_detected)

    ax.plot(x[0], x[1], "ro", markersize=8, label=f"state t={t:.3g}", zorder=6)
    if use_uncertainty:
        _theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(x[0] + lidar_range * np.cos(_theta),
                x[1] + lidar_range * np.sin(_theta),
                color="cyan", linewidth=1.5, linestyle="-", alpha=0.8,
                label=f"LIDAR r={lidar_range}", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(g.min[0], g.max[0])
    ax.set_ylim(g.min[1], g.max[1])
    ax.set_title(
        f"Reachable Set at t={t:.3g}\n"
        f"state = {np.round(x, 3)}\n"
        f"Root Node = X0"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    _draw_ttlt_tree(ax_tree, active_labels=set())

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    gif_frames.append(Image.open(buf).copy())
    buf.close()

    plt.pause(1)


    # ── Step 5: Online control synthesis ──────────────────────────────────────
    # tk_array = np.arange(0, stl_horizon + 1e-8, dt)
    tk_array = np.arange(0, stl_horizon, dt)

    # Algorithm 6 — Initialization
    PostSet = Initialization(solver.root, tk_arr=tk_array, t0=t)
    print("Initialization")
    print("Post(B_xt-):", PostSet)

    all_tubes = get_all_tubes(root=solver.root)
    print(f"\n  {'Label':<6}  {'ta':<10}  {'th':<10}")
    print("  " + "-" * 30)
    for tube in all_tubes:
        label  = tube.label if tube.label is not None else "—"
        ta_str = f"{tube.ta:.4f}" if tube.ta is not None else "None"
        th_str = f"{tube.th:.4f}" if tube.th != np.inf  else "inf"
        print(f"  {label:<6} {ta_str:<10}  {th_str:<10}")

    print("\n" + "-" * 60)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in tk_array:

        print(f"\nt = {t}")

        # Algorithm 7 — trackingSetNode
        B_xt = trackingSetNode(PostSet=PostSet, state=x, grid=solver.grid, t=t, dt=dt)
        print("B_xt (trackingSetNode):", B_xt)

        # Assign activation time to newly entered tubes
        for S_t in B_xt:
            if S_t.ta is None:
                S_t.ta = t

        # Algorithm 8 — updatetTLT
        updatetTLT(B_xt=B_xt, root=solver.root, tk_arr=tk_array, t=t, dt=dt)

        # Algorithm 9 — buildControlTree
        buildControlTree(B_xt=B_xt, root=solver.root, state=x, g=solver.grid,
                         dyn=solver.dynamics, dt=dt)
        print("B_xt (after updatetTLT & buildControlTree):", B_xt)

        # Algorithm 3 — Compression
        compressed_root = Compression(maxTempSeg, solver.grid)
        print("Compressed Tree:", get_all_tubes(root=compressed_root))

        # Algorithm 10 — Backtracking
        admissCtrlSet = Backtracking(
            root            = solver.root,
            compressed_root = compressed_root,
            state           = x,
            g               = solver.grid,
            dyn             = solver.dynamics,
        )
        print("Admissible Control Set:", admissCtrlSet)

        # Safety check
        if admissCtrlSet.size == 0 or admissCtrlSet.shape[-1] == 0:
            plt.ioff()
            plt.show(block=True)
            raise ValueError("No admissible control found — check STL spec and reachable sets.")

        # Apply first (optimal) control
        u_opt     = admissCtrlSet[0]
        print("Control applied:", u_opt)

        # Compute worst-case disturbance from active tube's value function gradient
        _sdf = B_xt[0].sdf if B_xt else None
        if _sdf is not None and _sdf.ndim == solver.grid.dims:
            _derivs  = [
                        computeSpatDerivArray(solver.grid, _sdf, deriv_dim=dim, accuracy="medium")
                        for dim in range(1, solver.grid.dims + 1)
                        ]
            spat_deriv = get_spat_deriv_at_state(solver.grid, _derivs, x)
            opt_dstb   = solver.dynamics.optDstb_inPython(x, spat_deriv)
        else:
            opt_dstb = None

        # Integrate state one step forward
        state_dot = solver.dynamics.dynamics_inPython(x, u_opt, disturbance=opt_dstb)
        x         = x + np.array(state_dot) * dt
        state_history.append(tuple(x))
        print("Updated state:", x)

        # Algorithm 11 — postSet
        PostSet = postSet(B_xt=B_xt, t=t, root=solver.root)
        print("PostSet:", PostSet)

        # Print tube status table
        all_tubes = get_all_tubes(root=solver.root)
        print(f"\n  {'Label':<6}  {'ta':<10}  {'th':<10}")
        print("  " + "-" * 30)
        for tube in all_tubes:
            label  = tube.label if tube.label is not None else "—"
            ta_str = f"{tube.ta:.4f}" if tube.ta is not None else "None"
            th_str = f"{tube.th:.4f}" if tube.th != np.inf  else "inf"
            print(f"  {label:<6}  {ta_str:<10}  {th_str:<10}")

        print("-" * 60)

        # ── LIDAR detection update ────────────────────────────────────────────
        if use_uncertainty:
            robot_pos = x[:2]

            print("predicate true centers", pred_true_centers)
            for i, tc in pred_true_centers.items():
                if i not in pred_detected:
                    if np.linalg.norm(robot_pos - np.array(tc)) <= lidar_range:
                        pred_detected.add(i)

                        predicates[i].c_x = tc[0]
                        predicates[i].c_y = tc[1]

                        pred        = predicates[i]
                        tube        = find_tube_in_tree(solver.root, repr(pred))
                        parent_op   = tube.parent   if tube else None
                        parent_tube = Pre(tube)     if tube else None

                        if isinstance(parent_op, UntilOperatorNode):
                            solver.update_until(op=parent_op, psi_center=tc, dt=dt)
                        elif isinstance(parent_op, EventuallyOperatorNode):
                            solver.update_eventually(op=parent_op, new_center=tc)
                        elif isinstance(parent_op, GloballyOperatorNode):
                            solver.update_globally(op=parent_op, new_center=tc)

                            if isinstance(parent_tube.parent, EventuallyOperatorNode):
                                solver.update_eventually(op=parent_tube.parent, new_center=tc)

                            #TODO Check if parent of Globally is Until

                        print(f"  [LIDAR] Predicate {i} detected at true center {np.round(tc, 3)}")
                        print(f"          tube        = {tube}")
                        print(f"          parent_op   = {parent_op}")
                        print(f"          parent_tube = {parent_tube}")

        # ── Plot update ───────────────────────────────────────────────────────
        ax.clear()

        for idx, S_t in enumerate(B_xt):
            if not (hasattr(S_t, "sdf") and S_t.sdf is not None):
                continue
            brs      = S_t.sdf
            z        = brs[:, :, 0] if brs.ndim == 3 else brs
            z_plot   = z.T
            z_masked = np.where(z_plot <= 0, 0.0, np.nan)

            ax.imshow(
                z_masked,
                extent=extent, origin="lower", aspect="auto",
                cmap=ListedColormap([brs_colors[idx % len(brs_colors)]]),
                vmin=-0.5, vmax=0.5,
                alpha=0.4, interpolation="nearest",
            )
            ax.contour(X, Y, z_plot, levels=[0.0],
                       colors=line_colors[idx % len(line_colors)],
                       linewidths=2, label=f"BRS {idx}")

        # predicate boundaries
        for entry in solver._predicate_cache.values():
            sdf_2d = entry["sdf"][:, :, 0] if entry["sdf"].ndim == 3 else entry["sdf"]
            ax.contour(X, Y, sdf_2d.T, levels=[0.0],
                       colors="green", linewidths=1.5, linestyles="--")

        # predicate offset circles
        if use_uncertainty:
            plot_predicate_offsets(ax, predicates, r_2=r_2, N=N_offset,
                                   true_centers=pred_true_centers, detected=pred_detected)

        # trajectory history
        if len(state_history) > 1:
            xs = [s[0] for s in state_history]
            ys = [s[1] for s in state_history]
            ax.plot(xs, ys, "r--o", markersize=4, linewidth=1,
                    label="trajectory", zorder=5)

        # current state
        ax.plot(x[0], x[1], "ro", markersize=8,
                label=f"state t={t:.3g}", zorder=6)
        if use_uncertainty:
            _theta = np.linspace(0, 2 * np.pi, 360)
            ax.plot(x[0] + lidar_range * np.cos(_theta),
                    x[1] + lidar_range * np.sin(_theta),
                    color="cyan", linewidth=1.5, linestyle="-", alpha=0.8,
                    label=f"LIDAR r={lidar_range}", zorder=5)

        bxt_labels  = ", ".join(S_t.label for S_t in B_xt   if hasattr(S_t, "label"))
        post_labels = ", ".join(S_t.label for S_t in PostSet if hasattr(S_t, "label"))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(g.min[0], g.max[0])
        ax.set_ylim(g.min[1], g.max[1])
        ax.set_title(
            f"Reachable Set at t={t:.3g}\n"
            f"state = {np.round(x, 3)}\n"
            f"B(x,t) = [{bxt_labels}]\n"
            f"Post(B(x,t)) = [{post_labels}]",
            fontsize=9,
        )
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
        _draw_ttlt_tree(ax_tree, active_labels={S_t.label for S_t in B_xt if S_t.label})

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())
        buf.close()

        plt.pause(0.01)

    # ── Post-loop ─────────────────────────────────────────────────────────────
    plt.ioff()

    if gif_frames and save_gif:
        gif_frames[0].save(
            gif_path,
            save_all       = True,
            append_images  = gif_frames[1:],
            duration       = 500,   # ms per frame
            loop           = 0,     # loop forever
        )
        print(f"\nGIF saved to {os.path.abspath(gif_path)}")

    plt.show(block=True)