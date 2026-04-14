"""
stl_hj.py
Converts an STLNode tree into an equivalent HJ Reachability setup,
and provides HJtTLT class for solving reachable sets from STL operators.

Predicate -> SDF conversion
---------------------------
The SDF is derived directly from the predicate expression string by
evaluating it over every grid point.  No cx/cy/r parameters are needed.
The 'hj' block in the YAML only requires a 'region' key:

    hj:
      region: target    # reach set   (l(x) <= 0 = safe, absorbing)
      region: obstacle  # avoid set   (l(x) <= 0 = unsafe, always excluded)

Sign convention: predicate expressions are written as f(x) > 0 (satisfied
when positive).  We store l(x) = -f(x) so that l(x) <= 0 means safe,
matching the HJ reachability convention throughout.

Dimensions not referenced by x[i] in the expression are automatically
treated as ignored (constant mid-value during evaluation).

Currently supported STL operators
----------------------------------
    Predicate    -->  expression-based SDF on grid
    Negation     -->  negated predicate SDF  (PNF: child always a Predicate)
    Globally     -->  Finite-time invariance set  (BRT)
    Eventually   -->  Finite-time BRS
    Until        -->  Finite-time reach-avoid BRS

Usage
-----
    from stl_hj import tTLT, HJtTLT

    result = tTLT(
        stl_yaml_path  = "configs/stl_spec_hj.yaml",
        hj_config_path = "configs/hj_config.yaml",
    )

    solver = HJtTLT(result, "configs/hj_config.yaml")
    solver.solve_all(dt=1, accuracy="medium", plot=True)      # Plotly
    solver.solve_all(dt=1, accuracy="medium", matplot=True)   # Matplotlib
"""

from __future__ import annotations

import re, hashlib
import sys
import math
import io
from typing import List
from PIL import Image


import numpy as np
import scipy.io as sio
import yaml
import plotly.graph_objects as go
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # use Tk instead of Qt to avoid segfault
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.interpolate import RegularGridInterpolator


sys.path.insert(0, str(Path(__file__).parent / "optimized_dp"))

from odp.Grid                     import Grid
from odp.Plots.plotting_utilities import slider_define
from odp.solver                   import HJSolver, computeSpatDerivArray
import odp.dynamics as _dyn_module

from stl_robustness import (
    STLNode,
    Predicate,
    Negation,
    Conjunction,
    Disjunction,
    Globally,
    Eventually,
    Until,
    parse_stl_yaml,
    _parse_node,
)

from tTLT_nodes import (
    TubeNode,
    OperatorNode,
    AndOperatorNode,
    OrOperatorNode,
    GloballyOperatorNode,
    EventuallyOperatorNode,
    UntilOperatorNode,
    wire,
    print_ttlt_tree,
    plot_ttlt_tree,
    _ttlt_collect_layout,
)

import torch
from HJR_FNO.HJR_FNO2d import FNO2d, SpectralConv2d


# --------------------
# HJ Config loader  (Grid + Dynamics)
# --------------------

def load_hj_config(hj_config_path: str) -> tuple[Grid, object]:
    with open(hj_config_path) as fh:
        cfg = yaml.safe_load(fh)

    gcfg         = cfg["grid"]
    dims         = int(gcfg["dims"])
    min_bounds   = np.array(gcfg["min_bounds"],   dtype=float)
    max_bounds   = np.array(gcfg["max_bounds"],   dtype=float)
    pts_each_dim = np.array(gcfg["pts_each_dim"], dtype=int)
    periodic     = gcfg.get("periodic_dims", [])

    g = Grid(
        minBounds    = min_bounds,
        maxBounds    = max_bounds,
        dims         = dims,
        pts_each_dim = pts_each_dim,
        periodicDims = periodic,
    )

    dcfg       = cfg["dynamics"]
    class_name = dcfg["class"]
    params     = dcfg.get("params", {})

    if not hasattr(_dyn_module, class_name):
        raise ValueError(f"Unknown dynamics class '{class_name}'.")

    dyn_class = getattr(_dyn_module, class_name)
    dynamics  = dyn_class(**params)

    dyn_dims = len(dynamics.x)
    if dyn_dims != g.dims:
        raise ValueError(
            f"Dynamics '{class_name}' has state dimension {dyn_dims} "
            f"but grid has {g.dims} dims."
        )

    return g, dynamics


# --------------------
# SDF helpers
# --------------------

def _referenced_dims(expression: str) -> set[int]:
    return {int(m) for m in re.findall(r"x\[(\d+)\]", expression)}


def _eval_expression_on_grid(expression: str, g: Grid) -> np.ndarray:
    dims            = g.dims
    pts             = g.pts_each_dim
    ref_dims        = _referenced_dims(expression)
    broadcast_shape = tuple(pts)

    axes  = [np.linspace(g.min[d], g.max[d], pts[d]) for d in range(dims)]
    grids = np.meshgrid(*axes, indexing="ij", sparse=True)

    x_grid = np.empty((dims, *broadcast_shape))
    for d in range(dims):
        if d in ref_dims:
            x_grid[d] = np.broadcast_to(grids[d], broadcast_shape)
        else:
            x_grid[d] = (g.min[d] + g.max[d]) / 2.0

    flat_size  = int(np.prod(pts))
    sdf_flat   = np.empty(flat_size, dtype=float)
    x_grid_2d  = x_grid.reshape(dims, flat_size)

    for i in range(flat_size):
        x = x_grid_2d[:, i]
        sdf_flat[i] = eval(expression, {"np": np, "x": x})

    return sdf_flat.reshape(broadcast_shape)


def predicate_to_hj(node: Predicate, g: Grid) -> dict:
    expression = node.expression
    hj         = getattr(node, "hj_params", None)
    region     = str(hj.get("region", "target")).lower() if hj else "target"

    if region not in ("target", "obstacle"):
        raise ValueError(f"hj.region must be 'target' or 'obstacle', got '{region}'.")

    raw_sdf     = _eval_expression_on_grid(expression, g)
    sdf         = -raw_sdf
    ref_dims    = _referenced_dims(expression)
    ignore_dims = [d for d in range(g.dims) if d not in ref_dims]

    formula = (
        f"Expr({expression!r})  ignore_dims={ignore_dims}  "
        f"— {'target/reach' if region == 'target' else 'obstacle/avoid'}"
    )

    print(f"  [predicate_to_hj]  {formula}")
    print(f"    raw_sdf : min={raw_sdf.min():.4f}  max={raw_sdf.max():.4f}")
    print(f"    sdf(=-f): min={sdf.min():.4f}  max={sdf.max():.4f}")

    return {"sdf": sdf, "region": region, "formula": formula}


# --------------------
# Tree walker
# --------------------

def _collect_predicates(node: STLNode, results: list) -> None:
    if isinstance(node, Predicate):
        results.append(node)
        return
    if isinstance(node, Negation):
        _collect_predicates(node.child, results)
    elif isinstance(node, (Conjunction, Disjunction)):
        _collect_predicates(node.left,  results)
        _collect_predicates(node.right, results)
    elif isinstance(node, (Globally, Eventually)):
        _collect_predicates(node.child, results)
    elif isinstance(node, Until):
        _collect_predicates(node.phi, results)
        _collect_predicates(node.psi, results)


# --------------------
# Extended YAML parser
# --------------------

def _attach_hj_params(cfg: dict, node: STLNode) -> None:
    node_type = cfg.get("type", "").upper()

    if node_type == "PREDICATE":
        node.hj_params = cfg.get("hj", None)
    elif node_type == "NOT":
        node.hj_params = None
        _attach_hj_params(cfg["child"], node.child)
    elif node_type in ("AND", "OR"):
        node.hj_params = None
        _attach_hj_params(cfg["left"],  node.left)
        _attach_hj_params(cfg["right"], node.right)
    elif node_type in ("G", "F"):
        node.hj_params = None
        _attach_hj_params(cfg["child"], node.child)
    elif node_type == "U":
        node.hj_params = None
        _attach_hj_params(cfg["phi"], node.phi)
        _attach_hj_params(cfg["psi"], node.psi)
    else:
        node.hj_params = None


def load_stl_with_hj(stl_yaml_path: str) -> tuple[STLNode, dict]:
    with open(stl_yaml_path) as fh:
        cfg = yaml.safe_load(fh)
    root = _parse_node(cfg, pnf=True)
    _attach_hj_params(cfg, root)
    return root, cfg


# --------------------
# tTLT  — top-level entry point
# --------------------

def tTLT(stl_yaml_path: str, hj_config_path: str) -> dict:
    g, dynamics = load_hj_config(hj_config_path)
    root, _     = load_stl_with_hj(stl_yaml_path)

    predicate_nodes: list[Predicate] = []
    _collect_predicates(root, predicate_nodes)

    target_sets   = []
    obstacle_sets = []
    labels        = []

    for pred in predicate_nodes:
        hj_result = predicate_to_hj(pred, g)
        labels.append(hj_result["formula"])
        if hj_result["region"] == "target":
            target_sets.append(hj_result["sdf"])
        else:
            obstacle_sets.append(hj_result["sdf"])

    print("=" * 60)
    print(f"  STL formula  : {root}")
    print(f"  Grid dims    : {g.dims}  pts: {g.pts_each_dim}")
    print(f"  Dynamics     : {dynamics.__class__.__name__}")
    print(f"  Target sets  : {len(target_sets)}")
    print(f"  Obstacle sets: {len(obstacle_sets)}")
    for lbl in labels:
        print(f"    {lbl}")
    print("=" * 60)

    return {
        "formula"       : root,
        "grid"          : g,
        "dynamics"      : dynamics,
        "target_sets"   : target_sets,
        "obstacle_sets" : obstacle_sets,
        "labels"        : labels,
    }


# --------------------
# Plot helpers
# --------------------

def _make_2d_meshgrid(g: Grid):
    cx = complex(0, g.pts_each_dim[0])
    cy = complex(0, g.pts_each_dim[1])
    mg_X, mg_Y = np.mgrid[g.min[0]:g.max[0]:cx, g.min[1]:g.max[1]:cy]
    return mg_X[:, 0], mg_Y[0, :]


def _make_3d_meshgrid(g: Grid):
    cx = complex(0, g.pts_each_dim[0])
    cy = complex(0, g.pts_each_dim[1])
    cz = complex(0, g.pts_each_dim[2])
    return np.mgrid[
        g.min[0]:g.max[0]:cx,
        g.min[1]:g.max[1]:cy,
        g.min[2]:g.max[2]:cz,
    ]


def _contour_boundary(x_axis, y_axis, z, color, name, width=2):
    return go.Contour(
        x=x_axis, y=y_axis, z=z.T,
        contours=dict(start=0, end=0, size=1, coloring="none"),
        contours_coloring="none",
        line=dict(color=color, width=width),
        showscale=False, name=name,
    )


def _iso_trace(mg_X, mg_Y, mg_Z, sdf, color, name, opacity):
    return go.Isosurface(
        x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
        value=sdf.flatten(),
        caps=dict(x_show=True, y_show=True),
        isomin=-0.1, isomax=0.1, surface_count=1,
        colorscale=[[0, color], [1, color]],
        opacity=opacity, showscale=False, name=name,
    )




# --------------------
# HJ - FNO
# --------------------

def HJR_FNO2d_SuperResQuery(g, sdf_input, time_hyparam, model):
    """
    Query the FNO2d model for a single SDF input across all time slices.
    Output resolution matches sdf_input shape — upsample sdf_input beforehand
    if super-resolution is desired.

    Args:
        g           : grid object with g.min, g.max  (physical domain bounds)
        sdf_input   : np.ndarray (Nx, Ny)  — SDF input, determines output resolution
        time_hyparam: np.ndarray (T,)      — time values to query
        model       : trained FNO2d model

    Returns:
        xx_query      : torch.Tensor (T, Nx, Ny, 4)
        pred_reshaped : torch.Tensor (Nx, Ny, T)
    """
    Nx_out, Ny_out = sdf_input.shape

    T = len(time_hyparam)

    # ---- Coordinate grids ------------------------------------------------
    # Use [-1, 1] to match training normalization.
    # g.min and g.max define the physical domain but coordinates were
    # normalized to [-1, 1] during training — must stay consistent.
    x_out = np.linspace(g.min[0], g.max[0], Nx_out)
    y_out = np.linspace(g.min[1], g.max[1], Ny_out)
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing='ij')   # (Nx_out, Ny_out)

    # SDF shape drives everything — no separate Nx_in/Ny_in needed
    assert sdf_input.ndim == 2, f"sdf_input must be 2D, got shape {sdf_input.shape}"

    sdf_torch = torch.tensor(sdf_input, dtype=torch.float32)   # (Nx_in, Ny_in)
    X_torch   = torch.tensor(X_out,    dtype=torch.float32)    # (Nx_out, Ny_out)
    Y_torch   = torch.tensor(Y_out,    dtype=torch.float32)

    # ---- Build query tensor: (T, Nx_out, Ny_out, 4) ----------------------
    xx_query = torch.empty(T, Nx_out, Ny_out, 4, dtype=torch.float32)
    for t_i in range(T):
        xx_query[t_i, :, :, 0] = sdf_torch                    # SDF(x,y)
        xx_query[t_i, :, :, 1] = X_torch                      # x coord
        xx_query[t_i, :, :, 2] = Y_torch                      # y coord
        xx_query[t_i, :, :, 3] = float(time_hyparam[t_i])     # t broadcast

    # ---- Run inference ---------------------------------------------------
    query_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xx_query),
        batch_size=8,
        shuffle=False
    )

    pred  = torch.zeros(T, Nx_out, Ny_out, dtype=torch.float32)
    index = 0

    model.eval()
    with torch.no_grad():
        for (x,) in query_loader:
            x   = x.cuda()
            out = model(x)                         # (B, Nx_out, Ny_out, 1)
            B   = out.size(0)
            pred[index:index+B] = out.squeeze(-1).cpu()
            index += B

    pred_reshaped = pred.permute(1, 2, 0)          # (Nx_out, Ny_out, T)
    return xx_query, pred_reshaped


# --------------------
# HJtTLT
# --------------------

class HJtTLT:
    """
    Wraps a tTLT result dict and provides HJ reachable set solvers
    for each STL temporal operator.

    The tTLT tree is built bottom-up via solve_all() or individual solve_*().
    self.root holds the final combined root TubeNode after solving.

    Plotting flags (can be combined):
        plot=True     -> Plotly interactive HTML viewer  (_plot_result)
        matplot=True  -> Matplotlib TkAgg window         (plot_brs_matplotlib)
    """

    def __init__(self, tTLT_result: dict, hj_config_path: str) -> None:
        self.formula         = tTLT_result["formula"]
        self.grid            = tTLT_result["grid"]
        self.dynamics        = tTLT_result["dynamics"]
        self.target_sets     = tTLT_result["target_sets"]
        self.obstacle_sets   = tTLT_result["obstacle_sets"]
        self.labels          = tTLT_result["labels"]
        self._hj_config_path = hj_config_path
        self._combined_sdfs: dict[str, np.ndarray] = {}
        self._tube_cache:    dict[str, TubeNode]   = {}
        self.root:           TubeNode | None        = None
        self.verbose         = True
        
        

        self._predicate_cache: dict[str, dict] = {}



        self.FNO_enable      = True
        save_path       = '/home/kmuenpra/git/tTLT/training_HJFNO/model/02_hjrno_Plane2D_Until_u[neg1_1]_d[neg01_01].pt'
        # ---- Load full model --------------------------------------------------------
        self.model_until = torch.load(save_path, weights_only=False, map_location='cuda')
        self.model_until.eval()
        print(f"Model loaded from:      {save_path}")
        print(f"Model type: {type(self.model_until)}")

        # ---- Load model from checkpoint ---------------------------------------------
        # ft_save_path = '/home/kmuenpra/git/tTLT/training_HJFNO/model/02_finetune_t8_best.pt'
        # self.model_until = FNO2d(modes1=16, modes2=16, width=64).to('cuda')
        # ft_ckpt = torch.load(ft_save_path, map_location='cuda', weights_only=False)
        # self.model_until.load_state_dict(ft_ckpt['model_state_dict'])
        # self.model_until.eval()
        # print(f"Model loaded from:      {ft_save_path}")


        # ---- Load precomputed BRS for Eventually and Globally ---------------
        _fg_mat = sio.loadmat(
            '/home/kmuenpra/git/tTLT/training_HJFNO/HJB_training_mat/Plane2D_FG_50x50.mat'
        )
        self.brs_full_eventually = _fg_mat['brs_F'].astype(np.float64)  # (Nx, Ny, T)
        self.brs_full_globally   = _fg_mat['brs_G'].astype(np.float64)  # (Nx, Ny, T)
        print(f"BRS loaded — F shape: {self.brs_full_eventually.shape}  G shape: {self.brs_full_globally.shape}")

        # # ---- Debug plot: F/G BRS at t=0 -------------------------------------
        # _x  = _fg_mat['x_axis'].squeeze()
        # _y  = _fg_mat['y_axis'].squeeze()
        # _X, _Y = np.meshgrid(_x, _y)
        # _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
        # for _ax, _brs, _title, _col in zip(
        #     _axes,
        #     [self.brs_full_eventually, self.brs_full_globally],
        #     ["Eventually (F) BRS  t=0", "Globally (G) BRS  t=0"],
        #     ["blue", "red"],
        # ):
        #     _z = _brs[:, :, 0].T
        #     _ax.contourf(_X, _Y, np.where(_z <= 0, _z, np.nan),
        #                  levels=[-1, 0], colors=[_col], alpha=0.3)
        #     _ax.contour(_X, _Y, _z, levels=[0], colors=[_col], linewidths=2)
        #     _ax.set_title(_title); _ax.set_aspect("equal")
        #     _ax.set_xlabel("x"); _ax.set_ylabel("y")
        # plt.tight_layout()
        # plt.show(block=True)           # pause here until the window is closed
        # # ---------------------------------------------------------------------

        

    # --------------------
    # Dynamics factory
    # --------------------

    def _make_dynamics(self, uMode: str, dMode: str) -> object:
        with open(self._hj_config_path) as fh:
            cfg = yaml.safe_load(fh)
        dcfg       = cfg["dynamics"]
        class_name = dcfg["class"]
        params     = dict(dcfg.get("params", {}))
        params["uMode"] = uMode
        params["dMode"] = dMode
        dyn_class = getattr(_dyn_module, class_name)
        return dyn_class(**params)

    # --------------------
    # Internal dispatch helper
    # --------------------

    def _do_plot(
        self,
        brs:         np.ndarray,
        static_sdfs: list,
        title:       str,
        brs_color:   str = "lightblue",
        brs_name:    str = "BRS",
        plot:        bool = False,
        matplot:     bool = False,
    ) -> None:
        """Call the appropriate plotter(s) based on flags."""
        if plot:
            self._plot_result(
                brs         = brs,
                static_sdfs = static_sdfs,
                title       = title,
                brs_color   = brs_color,
                brs_name    = brs_name,
            )
        if matplot:
            self.plot_brs_matplotlib(
                brs         = brs,
                grid        = self.grid,
                static_sdfs = static_sdfs,
                title       = title,
                brs_name    = brs_name,
            )

    # --------------------
    # Node-to-SDF resolver
    # --------------------

    def _resolve_node_to_sdf(
        self,
        node:        STLNode,
        dt:          float = 1,
        accuracy:    str   = "medium",
        plot:        bool  = False,
        matplot:     bool  = False,
        all_results: dict  = None,
    ) -> np.ndarray:
        if isinstance(node, Predicate):
            hj_result = predicate_to_hj(node, self.grid)
            sdf       = hj_result["sdf"]
            region    = hj_result["region"]
            tube      = TubeNode(sdf=sdf, formula=repr(node))

            # store in predicate cache
            self._predicate_cache[repr(node)] = {
                "node"   : node,
                "sdf"    : sdf,
                "region" : region,
                "tube"   : tube,
            }
            self._tube_cache[repr(node)] = tube
            return sdf

        elif isinstance(node, Negation):
            res = self.solve_negation(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
            if all_results is not None:
                all_results.update(res)
            return res[repr(node)]["sdf"]

        elif isinstance(node, Conjunction):
            res = self.solve_and(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
            if all_results is not None:
                all_results.update(res)
            return res[repr(node)]["sdf"]

        elif isinstance(node, Disjunction):
            res = self.solve_or(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
            if all_results is not None:
                all_results.update(res)
            return res[repr(node)]["sdf"]

        elif isinstance(node, Globally):
            res = self.solve_globally(node, dt=dt, accuracy=accuracy, save_all=True, plot=plot, matplot=matplot)
            if all_results is not None:
                all_results.update(res)
            return res[repr(node)]["sdf"]

        elif isinstance(node, Eventually):
            res = self.solve_eventually(node, dt=dt, accuracy=accuracy, save_all=True, plot=plot, matplot=matplot)
            if all_results is not None:
                all_results.update(res)
            return res[repr(node)]["sdf"]

        elif isinstance(node, Until):
            res = self.solve_until(node, dt=dt, accuracy=accuracy, save_all=True, plot=plot, matplot=matplot)
            if all_results is not None:
                all_results.update(res)
            return res[repr(node)]["sdf"]

        else:
            raise ValueError(f"_resolve_node_to_sdf: unsupported node type '{type(node).__name__}'.")

    # --------------------
    # Negation
    # --------------------

    def solve_negation(
        self,
        node:     Negation,
        dt:       float = 1,
        accuracy: str   = "medium",
        plot:     bool  = False,
        matplot:  bool  = False,
    ) -> dict:
        if not isinstance(node, Negation):
            raise ValueError(f"solve_negation: expected Negation, got {type(node).__name__}.")
        if not isinstance(node.child, Predicate):
            raise ValueError(f"solve_negation expects PNF: child must be Predicate, got {type(node.child).__name__}.")

        pred         = node.child
        hj_result    = predicate_to_hj(pred, self.grid)
        child_sdf    = hj_result["sdf"]
        child_region = hj_result["region"]
        neg_sdf      = -child_sdf
        neg_region   = "obstacle" if child_region == "target" else "target"
        formula      = f"NOT({repr(pred)})  region: {child_region} -> {neg_region}"

        print("=" * 60)
        print(f"  Solving Negation: NOT({pred.expression!r})")
        print(f"  child_sdf : min={child_sdf.min():.4f}  max={child_sdf.max():.4f}")
        print(f"  neg_sdf   : min={neg_sdf.min():.4f}  max={neg_sdf.max():.4f}")
        print(f"  region flip: {child_region} --> {neg_region}")
        print("=" * 60)

        neg_tube = TubeNode(sdf=neg_sdf, formula=formula, brs_full=None)
        brs_full = neg_sdf[:, :, np.newaxis] if neg_sdf.ndim == 2 else neg_sdf
        key      = repr(node)
        result   = {
            "tube"        : neg_tube,
            "sdf"         : neg_sdf,
            "brs_full"    : brs_full,
            "static_sdfs" : [(child_sdf,
                               "green" if neg_region == "target" else "red",
                               f"child SDF ({child_region})", 2)],
            "region"      : neg_region,
        }
        self._tube_cache[key] = neg_tube
        if node is self.formula:
            self.root = neg_tube

        self._do_plot(
            brs         = brs_full,
            static_sdfs = result["static_sdfs"],
            title       = f"NOT: {repr(node)}",
            brs_name    = f"NOT({child_region}->{neg_region})",
            plot        = plot,
            matplot     = matplot,
        )

        return {key: result}

    # --------------------
    # AND
    # --------------------

    def solve_and(
        self,
        node:     Conjunction,
        dt:       float = 1,
        accuracy: str   = "medium",
        plot:     bool  = False,
        matplot:  bool  = False,
    ) -> dict:
        if not isinstance(node, Conjunction):
            raise ValueError(f"solve_and: expected Conjunction, got {type(node).__name__}.")

        print("=" * 60)
        print(f"  Solving AND:  left={node.left!r}  right={node.right!r}")

        child_results: dict = {}
        l_sdf = self._resolve_node_to_sdf(node.left,  dt, accuracy, plot, matplot, child_results)
        r_sdf = self._resolve_node_to_sdf(node.right, dt, accuracy, plot, matplot, child_results)

        l_sdf_t0 = l_sdf[..., 0] if l_sdf.ndim == self.grid.dims + 1 else l_sdf
        r_sdf_t0 = r_sdf[..., 0] if r_sdf.ndim == self.grid.dims + 1 else r_sdf
        combined_sdf = np.maximum(l_sdf_t0, r_sdf_t0)

        print(f"  left_sdf      : min={l_sdf_t0.min():.4f}  max={l_sdf_t0.max():.4f}")
        print(f"  right_sdf     : min={r_sdf_t0.min():.4f}  max={r_sdf_t0.max():.4f}")
        print(f"  combined (AND): min={combined_sdf.min():.4f}  max={combined_sdf.max():.4f}")
        print(f"  Fraction where AND holds: {(combined_sdf <= 0).mean():.4f}")

        _LEAF_TYPES = (Predicate, Negation)
        both_static = isinstance(node.left, _LEAF_TYPES) and isinstance(node.right, _LEAF_TYPES)

        left_root  = self._tube_cache.get(repr(node.left),
                                          TubeNode(sdf=l_sdf_t0, formula=repr(node.left)))
        right_root = self._tube_cache.get(repr(node.right),
                                          TubeNode(sdf=r_sdf_t0, formula=repr(node.right)))

        combined_root = TubeNode(
            sdf     = combined_sdf,
            formula = f"AND({node.left!r},{node.right!r})",
            brs_full= None,
        )

        if both_static:
            print("  Both children static — collapsing to flat TubeNode (no operator child).")
        else:
            and_op = AndOperatorNode(left=left_root, right=right_root, stl_node=node)
            wire(combined_root, and_op, [left_root, right_root])
            print(f"  Wired AND: left={type(node.left).__name__}  right={type(node.right).__name__}")

        print("=" * 60)

        brs_full = combined_sdf[:, :, np.newaxis] if combined_sdf.ndim == 2 else combined_sdf
        key = repr(node)
        self._combined_sdfs[key] = combined_sdf
        result = {
            "tube"        : combined_root,
            "sdf"         : combined_sdf,
            "brs_full"    : brs_full,
            "static_sdfs" : [
                (l_sdf_t0, "green",  f"left:  {node.left!r}",  2),
                (r_sdf_t0, "orange", f"right: {node.right!r}", 2),
            ],
        }
        self._tube_cache[key] = combined_root
        if node is self.formula:
            self.root = combined_root

        self._do_plot(
            brs         = brs_full,
            static_sdfs = result["static_sdfs"],
            title       = f"AND: {repr(node)}",
            brs_name    = "AND region",
            plot        = plot,
            matplot     = matplot,
        )

        return {**child_results, key: result}

    # --------------------
    # OR
    # --------------------

    def solve_or(
        self,
        node:     Disjunction,
        dt:       float = 1,
        accuracy: str   = "medium",
        plot:     bool  = False,
        matplot:  bool  = False,
    ) -> dict:
        if not isinstance(node, Disjunction):
            raise ValueError(f"solve_or: expected Disjunction, got {type(node).__name__}.")

        print("=" * 60)
        print(f"  Solving OR:  left={node.left!r}  right={node.right!r}")

        child_results: dict = {}
        l_sdf = self._resolve_node_to_sdf(node.left,  dt, accuracy, plot, matplot, child_results)
        r_sdf = self._resolve_node_to_sdf(node.right, dt, accuracy, plot, matplot, child_results)

        l_sdf_t0 = l_sdf[..., 0] if l_sdf.ndim == self.grid.dims + 1 else l_sdf
        r_sdf_t0 = r_sdf[..., 0] if r_sdf.ndim == self.grid.dims + 1 else r_sdf
        combined_sdf = np.minimum(l_sdf_t0, r_sdf_t0)

        print(f"  left_sdf     : min={l_sdf_t0.min():.4f}  max={l_sdf_t0.max():.4f}")
        print(f"  right_sdf    : min={r_sdf_t0.min():.4f}  max={r_sdf_t0.max():.4f}")
        print(f"  combined (OR): min={combined_sdf.min():.4f}  max={combined_sdf.max():.4f}")
        print(f"  Fraction where OR holds: {(combined_sdf <= 0).mean():.4f}")

        _LEAF_TYPES = (Predicate, Negation)
        both_static = isinstance(node.left, _LEAF_TYPES) and isinstance(node.right, _LEAF_TYPES)

        left_root  = self._tube_cache.get(repr(node.left),
                                          TubeNode(sdf=l_sdf_t0, formula=repr(node.left)))
        right_root = self._tube_cache.get(repr(node.right),
                                          TubeNode(sdf=r_sdf_t0, formula=repr(node.right)))

        combined_root = TubeNode(
            sdf     = combined_sdf,
            formula = f"OR({node.left!r},{node.right!r})",
            brs_full= None,
        )

        if both_static:
            print("  Both children static — collapsing to flat TubeNode (no operator child).")
        else:
            or_op = OrOperatorNode(left=left_root, right=right_root, stl_node=node)
            wire(combined_root, or_op, [left_root, right_root])
            print(f"  Wired OR: left={type(node.left).__name__}  right={type(node.right).__name__}")

        print("=" * 60)

        brs_full = combined_sdf[:, :, np.newaxis] if combined_sdf.ndim == 2 else combined_sdf
        key = repr(node)
        self._combined_sdfs[key] = combined_sdf
        result = {
            "tube"        : combined_root,
            "sdf"         : combined_sdf,
            "brs_full"    : brs_full,
            "static_sdfs" : [
                (l_sdf_t0, "green",  f"left:  {node.left!r}",  2),
                (r_sdf_t0, "orange", f"right: {node.right!r}", 2),
            ],
        }
        self._tube_cache[key] = combined_root
        if node is self.formula:
            self.root = combined_root

        self._do_plot(
            brs         = brs_full,
            static_sdfs = result["static_sdfs"],
            title       = f"OR: {repr(node)}",
            brs_name    = "OR region",
            plot        = plot,
            matplot     = matplot,
        )

        return {**child_results, key: result}

    # --------------------
    # Globally
    # --------------------

    def solve_globally(
        self,
        node:     Globally,
        dt:       float = 1,
        accuracy: str   = "medium",
        save_all: bool  = True,
        plot:     bool  = False,
        matplot:  bool  = False,
    ) -> dict:
        if not isinstance(node, Globally):
            raise ValueError(f"solve_globally: expected Globally, got {type(node).__name__}.")

        a, b    = node.a, node.b
        horizon = b #NOTE choose reachable set horizon to be [0,b]
        tau     = np.arange(0, horizon + 1e-8, dt)

        child_results: dict = {}
        child_sdf  = self._resolve_node_to_sdf(node.child, dt, accuracy, plot, matplot, child_results)
        child_tube = TubeNode(sdf=child_sdf, formula=repr(node.child))

        inv_dynamics = self._make_dynamics(uMode="max", dMode="min")
        comp_methods = {"TargetSetMode": "minVOverTime"}

        print("=" * 60)
        print(f"  Solving invariance set for  G[{a},{b}]({node.child!r})")
        print(f"  uMode='max', dMode='min', minVOverTime")
        print(f"  Horizon: [0,{horizon}]  steps:{len(tau)}  dt={dt}  accuracy={accuracy}")
        print("=" * 60)


        if self.FNO_enable:

            tau_G  = np.arange(0, 10 + 1e-8, 0.25)
            t_hrz_index = np.argmin(np.abs(tau_G - horizon))

            if (isinstance(node.child, Predicate)
                    and node.child.c_x is not None and node.child.c_y is not None):
                brs_full = self.translate_brs(
                    brs_full = self.brs_full_globally[...,:(t_hrz_index+1)],
                    g        = self.grid,
                    dx       = node.child.c_x,
                    dy       = node.child.c_y,
                )
                print(f"  [FNO-G] Translated globally BRS by "
                      f"(+{node.child.c_x:.3f}, +{node.child.c_y:.3f})")
            else:
                brs_full = self.brs_full_globally[...,:(t_hrz_index+1)]
                print("  [FNO-G] Using raw precomputed globally BRS (no translation).")
        else:
            brs_full = HJSolver(
                dynamics_obj     = inv_dynamics,
                grid             = self.grid,
                multiple_value   = -child_sdf,
                tau              = tau,
                compMethod       = comp_methods,
                saveAllTimeSteps = save_all,
                accuracy         = accuracy,
                verbose          = self.verbose,
            )
            brs_full = -1 * brs_full
        brs_t0   = brs_full[..., 0]

        print(f"  child_sdf : min={child_sdf.min():.4f}  max={child_sdf.max():.4f}")
        print(f"  brs(t=0)  : min={brs_t0.min():.4f}  max={brs_t0.max():.4f}")
        print(f"  Fraction in invariance set (t=0): {(brs_t0 <= 0).mean():.4f}")
        print("=" * 60)

        g_op     = GloballyOperatorNode(a=a, b=b, child=child_tube, stl_node=node)
        brs_tube = TubeNode(sdf=brs_t0, formula=f"G[{a},{b}]({node.child!r})", brs_full=brs_full)
        wire(brs_tube, g_op, [child_tube])

        key    = repr(node)
        result = {
            "tube"        : brs_tube,
            "sdf"         : brs_t0,
            "brs_full"    : brs_full,
            "static_sdfs" : [(child_sdf, "green", "child SDF (stay inside)", 3)],
        }
        self._tube_cache[key] = brs_tube
        if node is self.formula:
            self.root = brs_tube

        self._do_plot(
            brs         = brs_full,
            static_sdfs = result["static_sdfs"],
            title       = f"G: {repr(node)}",
            brs_name    = "Invariance Set",
            plot        = plot,
            matplot     = matplot,
        )

        return {**child_results, key: result}

    # --------------------
    # Eventually
    # --------------------

    def solve_eventually(
        self,
        node:     Eventually,
        dt:       float = 1,
        accuracy: str   = "medium",
        save_all: bool  = True,
        plot:     bool  = False,
        matplot:  bool  = False,
    ) -> dict:
        if not isinstance(node, Eventually):
            raise ValueError(f"solve_eventually: expected Eventually, got {type(node).__name__}.")

        a, b    = node.a, node.b
        horizon = b #NOTE choose reachable set horizon to be [0,b]
        tau     = np.arange(0, horizon + 1e-8, dt)

        child_results: dict = {}
        # if isinstance(node.child, Globally):
        #     g_res      = self.solve_globally(node.child, dt=dt, accuracy=accuracy,
        #                                       save_all=save_all, plot=plot, matplot=matplot)
        #     child_results.update(g_res)
        #     child_tube = g_res[repr(node.child)]["tube"]
        #     child_sdf  = g_res[repr(node.child)]["sdf"]
        # else:
        child_sdf  = self._resolve_node_to_sdf(node.child, dt, accuracy, plot, matplot, child_results)
        child_tube = self._tube_cache.get(
            repr(node.child), TubeNode(sdf=child_sdf, formula=repr(node.child)))

        # print("Evnetually child_results", child_results)

        comp_methods = {"TargetSetMode": "minVWithV0"}

        print("=" * 60)
        print(f"  Solving BRS for  F[{a},{b}]({node.child!r})")
        print(f"  Horizon: [0,{horizon}]  steps:{len(tau)}  dt={dt}  accuracy={accuracy}")
        print("=" * 60)

        if self.FNO_enable:

            tau_F  = np.arange(0, 10 + 1e-8, 0.25)
            t_hrz_index = np.argmin(np.abs(tau_F - horizon))

            # Identify the predicate whose center shifts the BRS.
            # Supported child forms:
            #   F[a,b]( Predicate )            → predicate = node.child
            #   F[a,b]( G[a,b]( Predicate ) )  → predicate = node.child.child
            _pred = None
            if isinstance(node.child, Predicate):
                _pred = node.child
            elif isinstance(node.child, Globally) and isinstance(node.child.child, Predicate):
                _pred = node.child.child

            if _pred is not None and _pred.c_x is not None and _pred.c_y is not None:
                brs_full = self.translate_brs(
                    brs_full = self.brs_full_eventually[...,:(t_hrz_index + 1)],
                    g        = self.grid,
                    dx       = _pred.c_x,
                    dy       = _pred.c_y,
                )
                print(f"  [FNO-F] Translated eventually BRS by "
                      f"(+{_pred.c_x:.3f}, +{_pred.c_y:.3f})")
            else:
                brs_full = self.brs_full_eventually[...,:(t_hrz_index + 1)]
                print("  [FNO-F] Using raw precomputed eventually BRS (no translation).")
        else:
            brs_full = HJSolver(
                dynamics_obj     = self.dynamics,
                grid             = self.grid,
                multiple_value   = child_sdf,
                tau              = tau,
                compMethod       = comp_methods,
                saveAllTimeSteps = save_all,
                accuracy         = accuracy,
                verbose          = self.verbose,
            )
        brs_t0 = brs_full[..., 0]

        print(f"  child_sdf : min={child_sdf.min():.4f}  max={child_sdf.max():.4f}")
        print(f"  brs(t=0)  : min={brs_t0.min():.4f}  max={brs_t0.max():.4f}")
        print(f"  Fraction in BRS (t=0): {(brs_t0 <= 0).mean():.4f}")
        print("=" * 60)

        f_op     = EventuallyOperatorNode(a=a, b=b, child=child_tube, stl_node=node)
        brs_tube = TubeNode(sdf=brs_t0, formula=f"F[{a},{b}]({node.child!r})", brs_full=brs_full)
        wire(brs_tube, f_op, [child_tube])

        key    = repr(node)
        result = {
            "tube"        : brs_tube,
            "sdf"         : brs_t0,
            "brs_full"    : brs_full,
            "static_sdfs" : [(child_sdf, "green", "Target", 3)],
        }
        self._tube_cache[key] = brs_tube
        if node is self.formula:
            self.root = brs_tube

        self._do_plot(
            brs         = brs_full,
            static_sdfs = result["static_sdfs"],
            title       = f"F: {repr(node)}",
            brs_name    = "BRS",
            plot        = plot,
            matplot     = matplot,
        )

        return {**child_results, key: result}

    # --------------------
    # Until
    # --------------------

    @staticmethod
    def _collect_end_leaves(tube: TubeNode) -> list[TubeNode]:
        if tube.operator is None:
            return [tube]
        leaves = []
        op     = tube.operator
        if isinstance(op, (AndOperatorNode, OrOperatorNode)):
            leaves += HJtTLT._collect_end_leaves(op.left)
            leaves += HJtTLT._collect_end_leaves(op.right)
        elif isinstance(op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
            leaves += HJtTLT._collect_end_leaves(op.child)
        return leaves

    @staticmethod
    def _replace_leaf(old_leaf: TubeNode, new_tube: TubeNode) -> None:
        parent_op = old_leaf.parent
        if parent_op is None:
            return
        if isinstance(parent_op, (AndOperatorNode, OrOperatorNode)):
            if parent_op.left is old_leaf:
                parent_op.left  = new_tube
            elif parent_op.right is old_leaf:
                parent_op.right = new_tube
        elif isinstance(parent_op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
            if parent_op.child is old_leaf:
                parent_op.child = new_tube
        new_tube.parent = parent_op


    def translated_predicate_sdf_until(self, phi: Predicate, psi: Predicate, g: Grid) -> np.ndarray:

        if any(v is None for v in [phi.c_x, phi.c_y, phi.r, psi.c_x, psi.c_y]):
            raise ValueError("phi must have c_x, c_y, r set and psi must have c_x, c_y set.")

        new_cx = phi.c_x - psi.c_x
        new_cy = phi.c_y - psi.c_y

        translated_expression = (
            f"{phi.r} - ((x[0]-{new_cx})**2 + (x[1]-{new_cy})**2)**0.5"
        )

        translated_node            = Predicate(
            expression = translated_expression,
            c_x        = new_cx,
            c_y        = new_cy,
            r          = phi.r,
        )
        translated_node.hj_params  = {"region": "target"}   # <-- added

        hj_result = predicate_to_hj(translated_node, g)
        return hj_result["sdf"]
    

    def translate_brs(
        self,
        brs_full: np.ndarray,
        g:        Grid,
        dx:       float,
        dy:       float,
        fill:     float = 1.0,
        ) -> np.ndarray:
        """
        Translate brs_full by (dx, dy) over the same grid bounds.

        For each output grid point (x, y), the value is looked up at
        (x - dx, y - dy) in the original brs_full.  This is equivalent
        to shifting the reachable set origin by (+dx, +dy).

        Points that map outside the original grid are filled with `fill`.

        Parameters
        ----------
        brs_full : np.ndarray, shape (Nx, Ny, T)
        g        : Grid   — defines axis coordinates
        dx       : float  — translation in x  (e.g. psi.c_x)
        dy       : float  — translation in y  (e.g. psi.c_y)
        fill     : float  — value for points outside original grid (default +1.0)

        Returns
        -------
        brs_translated : np.ndarray, shape (Nx, Ny, T)
        """

        x_coords = np.linspace(g.min[0], g.max[0], g.pts_each_dim[0])  # (Nx,)
        y_coords = np.linspace(g.min[1], g.max[1], g.pts_each_dim[1])  # (Ny,)
        T        = brs_full.shape[2]

        brs_translated = np.full_like(brs_full, fill_value=fill)

        for k in range(T):

            interp = RegularGridInterpolator(
                (x_coords, y_coords),
                brs_full[:, :, k],
                method       = "linear",
                bounds_error = False,   # don't raise on out-of-bounds
                fill_value   = fill,    # out-of-bounds → fill
            )

            # For each output point (x, y), query the original at (x - dx, y - dy)
            xx, yy        = np.meshgrid(x_coords, y_coords, indexing="ij")  # (Nx, Ny)
            query_points  = np.stack([xx - dx, yy - dy], axis=-1)           # (Nx, Ny, 2)

            brs_translated[:, :, k] = interp(query_points)

        return brs_translated
    

    def update_until(self, op:UntilOperatorNode, psi_center:List, dt):


        # Phi unitl Psi
        phi = op.stl_node.phi
        psi = op.stl_node.psi

        psi.c_x = psi_center[0]
        psi.c_y = psi_center[1]

        horizon = op.b #NOTE choose reachable set horizon to be [0,b]
        tau     = np.arange(0, horizon + 1e-8, dt)


        #Origin centered Phi SDF (compatible with FNO)
        sdf_phi_translated = self.translated_predicate_sdf_until(phi, psi, self.grid)

        # ---- FNO prediction -------------------------------------------------
        _, FNO_pred = HJR_FNO2d_SuperResQuery(
            g            = self.grid,
            sdf_input    = -sdf_phi_translated,
            time_hyparam = tau, #(10-horizon) + tau,
            model        = self.model_until,
        )


        brs_full = FNO_pred.detach().cpu().numpy()

        # Suffix minimum: brs_full[..., t_i] = min(brs_full[..., t_i:])
        brs_full = np.minimum.accumulate(brs_full[..., ::-1], axis=-1)[..., ::-1]

        brs_full = self.translate_brs(
                brs_full = brs_full,
                g        = self.grid,
                dx       = psi_center[0],
                dy       = psi_center[1],
            )

        #===============================================================
        # ---- Update phi TubeNode with the recomputed BRS --------------------
        op.tube.sdf      = brs_full[..., op.tube.t_index]
        op.tube.brs_full = brs_full


        # ---- Update psi TubeNode with translated SDF --------------------
        # NOTE psi is always Predicate Node, so there's no changes to it

        translated_expression = (
            f"{op.stl_node.psi.r} - ((x[0]-{psi_center[0]})**2 + (x[1]-{psi_center[1]})**2)**0.5"
        )
        translated_node            = Predicate(
            expression = translated_expression,
            c_x        = psi_center[0],
            c_y        = psi_center[1],
            r          = op.stl_node.psi.r,
        )
        translated_node.hj_params  = {"region": "target"}   # <-- added
        hj_result = predicate_to_hj(translated_node, self.grid)

        op.child.sdf = hj_result["sdf"]
        op.child.brs_full = None
        #===============================================================

        print(f"  [update_until] psi BRS updated  "
              f"| true center = ({op.stl_node.psi.c_x:.3f}, {op.stl_node.psi.c_y:.3f})"
              f"  t_index = {op.child.t_index}"
              f"  sdf shape = {op.child.sdf.shape}")


    def update_eventually(
        self,
        op:       EventuallyOperatorNode,
        new_center: List,
    ) -> None:
        """
        Re-translate the precomputed eventually BRS to a new predicate center
        and update the child TubeNode in-place.

        Parameters
        ----------
        node       : EventuallyOperatorNode whose .child is the target TubeNode.
        new_center : [c_x, c_y] — true center of the detected predicate.
        """
        c_x, c_y = new_center[0], new_center[1]

        brs_full = self.translate_brs(
            brs_full = self.brs_full_eventually,
            g        = self.grid,
            dx       = c_x,
            dy       = c_y,
        )

        op.tube.sdf      = brs_full[..., op.tube.t_index]
        op.tube.brs_full = brs_full

        print(f"  [update_eventually] BRS updated"
              f"  | true center = ({c_x:.3f}, {c_y:.3f})"
              f"  t_index = {op.tube.t_index}"
              f"  sdf shape = {op.tube.sdf.shape}")


    def update_globally(
        self,
        op:       GloballyOperatorNode,
        new_center: List,
    ) -> None:
        """
        Re-translate the precomputed globally BRS to a new predicate center
        and update the child TubeNode in-place.

        Parameters
        ----------
        node       : GloballyOperatorNode whose .child is the target TubeNode.
        new_center : [c_x, c_y] — true center of the detected predicate.
        """
        c_x, c_y = new_center[0], new_center[1]

        brs_full = self.translate_brs(
            brs_full = self.brs_full_globally,
            g        = self.grid,
            dx       = c_x,
            dy       = c_y,
        )

        op.tube.sdf      = brs_full[..., op.tube.t_index]
        op.tube.brs_full = brs_full

        print(f"  [update_globally] BRS updated"
              f"  | true center = ({c_x:.3f}, {c_y:.3f})"
              f"  t_index = {op.tube.t_index}"
              f"  sdf shape = {op.tube.sdf.shape}")


    def solve_until(
        self,
        node:     Until,
        dt:       float = 1,
        accuracy: str   = "medium",
        save_all: bool  = True,
        plot:     bool  = False,
        matplot:  bool  = False,
    ) -> dict:
        if not isinstance(node, Until):
            raise ValueError(f"solve_until: expected Until, got {type(node).__name__}.")

        a, b    = node.a, node.b
        horizon = b #NOTE choose reachable set horizon to be [0,b]
        tau     = np.arange(0, horizon + 1e-8, dt)

        comp_methods = {
            "TargetSetMode":   "minVWithV0",
            "ObstacleSetMode": "maxVWithObstacle",
        }



        #===================
        # NOTE Current implementation, we limit our formula to mu_i Until mu_j (where mu are predicates)
        #==================


        child_results: dict = {}
        phi_sdf_root = self._resolve_node_to_sdf(node.phi, dt, accuracy, plot, matplot, child_results)

        _phi_u_label = f"phi(U[{a},{b}])({node.phi!r})"
        if repr(node.phi) in self._tube_cache:
            phi_tube = self._tube_cache[repr(node.phi)]
            phi_tube.formula = _phi_u_label
            self._tube_cache[_phi_u_label] = phi_tube
        else:
            phi_tube = TubeNode(sdf=phi_sdf_root, formula=_phi_u_label)
            self._tube_cache[_phi_u_label] = phi_tube

        end_leaves = self._collect_end_leaves(phi_tube)
        

        psi_sdf_root = self._resolve_node_to_sdf(node.psi, dt, accuracy, plot, matplot, child_results)


        print("=" * 60)
        print(f"  Solving Until:  {node.phi!r} U[{a},{b}] {node.psi!r}")
        print(f"  Horizon: [0,{horizon}]  steps:{len(tau)}  dt={dt}  accuracy={accuracy}")
        print(f"  Found {len(end_leaves)} end leaf(ves) in phi subtree")
        print("=" * 60)

        first_brs_t0   = None
        first_brs_full = None
        u_op_i         = None

        for i, leaf in enumerate(end_leaves):
            phi_sdf_i = leaf.sdf
            print(f"  [{i+1}/{len(end_leaves)}] HJ solve: phi='{leaf.formula}'")


            if self.FNO_enable:

                #Origin centered Phi SDF (compatible with FNO)
                sdf_phi_translated = self.translated_predicate_sdf_until(node.phi, node.psi, self.grid)

                # ---- FNO prediction -------------------------------------------------
                _, FNO_pred = HJR_FNO2d_SuperResQuery(
                    g            = self.grid,
                    sdf_input    = -sdf_phi_translated,
                    time_hyparam = tau, #(10-horizon) + tau,
                    model        = self.model_until,
                )


                brs_full = FNO_pred.detach().cpu().numpy()

                # Suffix minimum: brs_full[..., t_i] = min(brs_full[..., t_i:])
                brs_full = np.minimum.accumulate(brs_full[..., ::-1], axis=-1)[..., ::-1]

                brs_full = self.translate_brs(
                        brs_full = brs_full,
                        g        = self.grid,
                        dx       = node.psi.c_x,
                        dy       = node.psi.c_y,
                    )
                
                # x_coords = np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0])
                # y_coords = np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1])
                # X, Y = np.meshgrid(x_coords, y_coords)
                # for k in range(brs_full.shape[2]):
                #     z = brs_full[:, :, k].T
                #     plt.figure()
                #     plt.contourf(X, Y, z, levels=[z.min(), 0], colors=["lightblue"])
                #     plt.contour(X, Y, z, levels=[0], colors="blue", linewidths=2)
                #     plt.title(f"t_index={k}")
                #     plt.axis("equal")
                #     plt.show()
                

                



            else:

                brs_full = HJSolver(
                    dynamics_obj     = self.dynamics,
                    grid             = self.grid,
                    multiple_value   = [psi_sdf_root, -phi_sdf_i],
                    tau              = tau,
                    compMethod       = comp_methods,
                    saveAllTimeSteps = save_all,
                    accuracy         = accuracy,
                    verbose          = self.verbose,
                )

            brs_t0 = brs_full[..., 0]
            if first_brs_t0   is None: first_brs_t0   = brs_t0
            if first_brs_full is None: first_brs_full = brs_full

            print(f"    phi_sdf : min={phi_sdf_i.min():.4f}  max={phi_sdf_i.max():.4f}")
            print(f"    psi_sdf : min={psi_sdf_root.min():.4f}  max={psi_sdf_root.max():.4f}")
            print(f"    brs_t0  : min={brs_t0.min():.4f}  max={brs_t0.max():.4f}")
            print(f"    Fraction in BRS (t=0): {(brs_t0 <= 0).mean():.4f}")

            psi_tube = self._tube_cache.get(
                repr(node.psi),
                TubeNode(sdf=psi_sdf_root, formula=f"copy_{i+1}({repr(node.psi)})", brs_full=None),
            )
            u_op_i = UntilOperatorNode(a=a, b=b, child=psi_tube, stl_node=node)

            if leaf is phi_tube:
                # Simple case: phi is a direct predicate leaf.
                # Wire phi_tube itself as the UntilOperatorNode owner so it stays
                # reachable in the tTLT tree with correct .parent pointers.
                leaf.sdf      = brs_t0
                leaf.brs_full = brs_full
                wire(leaf, u_op_i, [psi_tube])
                # _replace_leaf is a no-op here (leaf.parent is None), so skip it.
            else:
                # Complex case: phi has sub-operators; replace the end leaf with
                # a dedicated BRS tube and graft it into the phi subtree.
                brs_tube_i = TubeNode(sdf=brs_t0, formula=f"copy_{i+1}({repr(node)})", brs_full=brs_full)
                wire(brs_tube_i, u_op_i, [psi_tube])
                self._replace_leaf(leaf, brs_tube_i)

            self._do_plot(
                brs         = brs_full,
                static_sdfs = [
                    (phi_sdf_i,    "green",  f"phi leaf {i+1} (stay inside)", 3),
                    (psi_sdf_root, "orange", "psi (reach target)",            3),
                ],
                title   = f"U leaf {i+1}/{len(end_leaves)}: {repr(node)}",
                brs_name= "BRS",
                plot    = plot,
                matplot = matplot,
            )

        combined_root = phi_tube

        key    = repr(node)
        result = {
            "tube"        : combined_root,
            "sdf"         : first_brs_t0,
            "brs_full"    : first_brs_full,
            "static_sdfs" : [
                (phi_sdf_root, "green",  "phi (stay inside)",  3),
                (psi_sdf_root, "orange", "psi (reach target)", 3),
            ],
        }
        self._tube_cache[key] = combined_root
        if node is self.formula:
            self.root = combined_root

        print(f"  Combined root: {combined_root.formula!r}")
        print("=" * 60)

        return {**child_results, key: result}

    # --------------------
    # Solve all (single entry point)
    # --------------------

    def solve_all(
        self,
        dt:       float = 1,
        accuracy: str   = "medium",
        plot:     bool  = False,
        matplot:  bool  = False,
        verbose:  bool  = True,
    ) -> None:
        self.verbose = verbose

        print("\n" + "=" * 60)
        print("  Solving STL formula (bottom-up, single dispatch)")
        print("=" * 60)

        node = self.formula

        if isinstance(node, Predicate):
            sdf = predicate_to_hj(node, self.grid)["sdf"]
            self.root = TubeNode(sdf=sdf, formula=repr(node), brs_full=None)
        elif isinstance(node, Negation):
            self.solve_negation(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
        elif isinstance(node, Conjunction):
            self.solve_and(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
        elif isinstance(node, Disjunction):
            self.solve_or(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
        elif isinstance(node, Globally):
            self.solve_globally(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
        elif isinstance(node, Eventually):
            self.solve_eventually(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
        elif isinstance(node, Until):
            self.solve_until(node, dt=dt, accuracy=accuracy, plot=plot, matplot=matplot)
        else:
            raise ValueError(f"solve_all: unsupported root node type '{type(node).__name__}'.")

        if self.root is None:
            print("  Warning: solver.root not set after solve_all.")
            return

        _ = _ttlt_collect_layout(self.root)

        print("\n" + "=" * 60)
        print("  tTLT TREE STRUCTURE")
        print("=" * 60)
        print_ttlt_tree(self.root)
        plot_ttlt_tree(self.root, save_path="ttlt_tree.html")
        print("=" * 60)

    # --------------------
    # Plotly plotter  (plot=True)
    # --------------------

    def _plot_result(
            self,
            brs:         np.ndarray,
            static_sdfs: list,
            title:       str,
            brs_color:   str = "lightblue",
            brs_name:    str = "BRS",
        ) -> None:
        g    = self.grid
        dims = g.dims

        if dims == 2:
            x_axis, y_axis = _make_2d_meshgrid(g)
            if brs.ndim == 2:
                brs = brs[:, :, np.newaxis]
            N = brs.shape[2]

            def _fill(z):
                z_m = np.where(z <= 0, z, np.nan)
                return go.Heatmap(
                    x=x_axis, y=y_axis, z=z_m.T,
                    colorscale=[[0, brs_color], [1, brs_color]],
                    zmin=-0.1, zmax=0.0,
                    opacity=0.5, showscale=False,
                    name=f"{brs_name} (fill)",
                    hoverinfo="skip",
                )

            def _hover(z):
                z_m    = np.where(z <= 0, z, np.nan)
                xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
                mask   = ~np.isnan(z_m)
                return go.Scatter(
                    x=xx[mask].ravel(),
                    y=yy[mask].ravel(),
                    mode="markers",
                    marker=dict(size=6, color="rgba(0,0,0,0)"),
                    customdata=z_m[mask].ravel(),
                    hovertemplate=(
                        "x: %{x:.3f}<br>"
                        "y: %{y:.3f}<br>"
                        "V(x,y): %{customdata:.4f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name="hover",
                )

            def _boundary(z):
                return go.Contour(
                    x=x_axis, y=y_axis, z=z.T,
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    contours_coloring="none",
                    line=dict(color="blue", width=2),
                    showscale=False, name=brs_name,
                )

            static_traces = [
                _contour_boundary(x_axis, y_axis, sdf, color, name, width)
                for sdf, color, name, width in static_sdfs
            ]

            frames = [go.Frame(
                data=[_fill(brs[:, :, k]), _boundary(brs[:, :, k])],
                traces=[0, 1], name=str(k),
            ) for k in range(N)]

            fig = go.Figure(
                data=[_fill(brs[:, :, 0]), _boundary(brs[:, :, 0]), *static_traces],
                frames=frames,
            )
            fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")

        elif dims == 3:
            mg_X, mg_Y, mg_Z = _make_3d_meshgrid(g)
            if brs.ndim == 3:
                brs = brs[:, :, :, np.newaxis]
            N = brs.shape[3]

            static_traces = [
                _iso_trace(mg_X, mg_Y, mg_Z, sdf, color, name, 0.2)
                for sdf, color, name, _ in static_sdfs
            ]
            frames = [go.Frame(
                data=[_iso_trace(mg_X, mg_Y, mg_Z, brs[:, :, :, k], brs_color, brs_name, 0.9)],
                traces=[0], name=str(k),
            ) for k in range(N)]
            fig = go.Figure(
                data=[
                    _iso_trace(mg_X, mg_Y, mg_Z, brs[:, :, :, 0], brs_color, brs_name, 0.9),
                    *static_traces,
                ],
                frames=frames,
            )
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis={"nticks": 20}, zaxis={"nticks": 20},
                    camera_eye={"x": 0, "y": -1, "z": 0.5},
                    aspectratio={"x": 1, "y": 1, "z": 0.6},
                ),
            )
        else:
            raise ValueError(f"Plotting not supported for grid dims={dims}.")

        fig = slider_define(fig)
        fig.show()


        #=== EXTRA: save gif ====
        # gif_frames = []
        # for frame in fig.frames:
        #     # Overlay this frame's data on top of the static traces
        #     frame_fig = go.Figure(
        #         data=[*frame.data, *static_traces],  # animated layers + static
        #         layout=fig.layout,
        #     )
        #     png_bytes = frame_fig.to_image(format="png", width=800, height=600, scale=1)
        #     gif_frames.append(Image.open(io.BytesIO(png_bytes)).copy())


        # safe_title = re.sub(r'[\\/*?:"<>|()\']', "_", title).strip()
        # if len(safe_title) > 60:
        #     h = hashlib.md5(title.encode()).hexdigest()[:8]
        #     safe_title = f"{safe_title[:60]}_{h}"
        # gif_frames[0].save(
        #     f"{safe_title}.gif",
        #     save_all=True,
        #     append_images=gif_frames[1:],
        #     duration=100,   # ms per frame
        #     loop=0,
        # )
        # print("GIF saved to output.gif")

    # --------------------
    # Matplotlib plotter  (matplot=True)
    # --------------------

    def plot_brs_matplotlib(
        self,
        brs: np.ndarray,
        grid,
        title: str = "Backward Reachable Set",
        brs_color: str = "lightblue",
        zero_contour_color: str = "blue",
        zero_contour_width: float = 2.0,
        brs_alpha: float = 0.5,
        time_arr: np.ndarray | None = None,
        static_sdfs: list | None = None,
        brs_name: str = "BRS",
    ) -> None:
        dims = grid.dims

        if dims == 2:
            x_coords = grid.grid_points[0]               # (Nx,)
            y_coords = grid.grid_points[1]               # (Ny,)
            z = brs[:, :, 0] if brs.ndim == 3 else brs  # (Nx, Ny)

        elif dims == 3:
            x_coords = grid.grid_points[0]               # (Nx,)
            y_coords = grid.grid_points[1]               # (Ny,)
            z_coords = grid.grid_points[2]               # (Nz,)
            z = brs[:, :, 0, 0] if brs.ndim == 4 else brs[:, :, 0]  # (Nx, Ny)

        else:
            raise ValueError(f"Plotting not supported for grid dims={dims}.")

        z_plot   = z.T                                   # (Ny, Nx) row-major
        z_masked = np.where(z_plot <= 0, 0.0, np.nan)

        extent    = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
        X, Y      = np.meshgrid(x_coords, y_coords)     # (Ny, Nx)
        fill_cmap = ListedColormap([brs_color])

        _z_suffix = ""
        if dims == 3:
            _z_suffix = f"  [z = {z_coords[0]:.3g}]"

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.imshow(
            z_masked,
            extent=extent, origin="lower", aspect="auto",
            cmap=fill_cmap, vmin=-0.5, vmax=0.5,
            alpha=brs_alpha, interpolation="nearest",
        )
        ax.contour(X, Y, z_plot, levels=[0.0],
                   colors=zero_contour_color,
                   linewidths=zero_contour_width)

        if static_sdfs:
            for sdf, color, name, width in static_sdfs:
                sdf_2d = sdf[:, :, 0] if sdf.ndim == 3 else sdf  # (Nx, Ny)
                ax.contour(X, Y, sdf_2d.T, levels=[0.0],
                           colors=color, linewidths=width, linestyles="--")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title + _z_suffix)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", linestyle="--", alpha=0.4)

        legend_elements = [
            Patch(facecolor=brs_color, alpha=brs_alpha,
                  edgecolor=zero_contour_color,
                  linewidth=zero_contour_width, label=brs_name),
        ]
        if static_sdfs:
            for _, color, name, width in static_sdfs:
                legend_elements.append(
                    Line2D([0], [0], color=color, linewidth=width,
                           linestyle="--", label=name)
                )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.show(block=True)

    # --------------------
    # Generic node collector (utility, kept for external use)
    # --------------------

    @staticmethod
    def _collect_nodes(node: STLNode, target_type: type, out: list) -> None:
        if isinstance(node, target_type):
            out.append(node)
        if isinstance(node, Negation):
            HJtTLT._collect_nodes(node.child, target_type, out)
        elif isinstance(node, (Conjunction, Disjunction)):
            HJtTLT._collect_nodes(node.left,  target_type, out)
            HJtTLT._collect_nodes(node.right, target_type, out)
        elif isinstance(node, (Globally, Eventually)):
            HJtTLT._collect_nodes(node.child, target_type, out)
        elif isinstance(node, Until):
            HJtTLT._collect_nodes(node.phi, target_type, out)
            HJtTLT._collect_nodes(node.psi, target_type, out)

    def __repr__(self) -> str:
        return (
            f"HJtTLT(formula={self.formula!r}, "
            f"dynamics={self.dynamics.__class__.__name__}, "
            f"grid_dims={self.grid.dims})"
        )


# --------------------
# MAIN
# --------------------

if __name__ == "__main__":

    # # ── Example 1: Plane2D  F[5,10](G[0,10] mu_1) AND (mu_2 U[0,8] mu_3) ──
    # result = tTLT(
    #     stl_yaml_path  = "configs/stl_spec_hj.yaml",
    #     hj_config_path = "configs/hj_config.yaml",
    # )
    # solver = HJtTLT(result, "configs/hj_config.yaml")
    # solver.solve_all(dt=1, accuracy="medium", plot=True, verbose=False)

    # ── Example 2: DubinsCar  (mu_1 AND NOT mu_2) U[0,12] G[0,8](mu_3) ────
    result = tTLT(
        stl_yaml_path  = "configs/stl_spec_dubins.yaml",
        hj_config_path = "configs/hj_config_dubins.yaml",
    )
    solver = HJtTLT(result, "configs/hj_config_dubins.yaml")
    solver.solve_all(dt=1, accuracy="medium", plot=True, verbose=False)