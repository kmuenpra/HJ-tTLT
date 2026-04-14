'''
tTLT_synthesis
@author: Kasidit Muenprasitivej
'''

from __future__ import annotations

import re
import sys
import os
from PIL import Image
import io

import math
import numpy as np
import yaml
import plotly.graph_objects as go
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")          # must be before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

import copy
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent / "optimized_dp"))

from odp.Grid                     import Grid
from odp.Plots.plotting_utilities import slider_define
from odp.solver                   import HJSolver, computeSpatDerivArray
import odp.dynamics as _dyn_module
from odp.dynamics import DubinsCar2, Plane2D


from stl_robustness import (
    STLHorizonEvaluator,
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
    Post,Pre,Children,Parent,
)

from stl_hj import (
    HJtTLT,
    tTLT,
    _make_2d_meshgrid,
    _make_3d_meshgrid,
    _contour_boundary,
    _iso_trace,
)


# --------------------
# get_MTS
# --------------------

def get_MTS(root: TubeNode) -> list[list[TubeNode]]:
    """
    Given a solved tTLT tree rooted at `root`, return a list of MTS segments.
    Works on a deep copy of the tree so the original is never mutated.

    Segmentation rules
    ------------------
    A segment boundary is created at every Boolean operator node (AND / OR).
    Each MTS is a contiguous chain of TubeNodes with no Boolean operator
    between them.  Concretely:

      1. Root → the TubeNode that is the PARENT of the first Boolean operator
         (this may be just the root itself if the root's operator is Boolean).
      2. Each CHILD TubeNode of a Boolean operator → the TubeNode that is the
         parent of the *next* Boolean operator on that branch (or the leaf if
         there is none).
      3. Every such chain starts and ends with a TubeNode and contains only
         temporal operators (G, F, U) internally.

    An MTS that consists of a single TubeNode (no operator child, or whose
    only operator is Boolean) is valid and is returned as a one-element list.

    Parameters
    ----------
    root : TubeNode
        Root of a fully solved tTLT tree (as returned by HJtTLT.solve_all).

    Returns
    -------
    segments : list[list[TubeNode]]
        Each inner list is one MTS segment — an ordered chain of TubeNodes
        from the segment's root down to (but not including) the children of
        the next Boolean operator or the tree leaf.
    """

    _BOOLEAN_OPS  = (AndOperatorNode, OrOperatorNode)
    _TEMPORAL_OPS = (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)


    segments: list[list[TubeNode]] = []

    def _walk_segment(tube: TubeNode, current_segment: list[TubeNode]) -> None:
        current_segment.append(tube)
        op = tube.operator

        if op is None:
            segments.append(current_segment)
            return

        if isinstance(op, _BOOLEAN_OPS):
            segments.append(current_segment)
            for child in [op.left, op.right]:
                _walk_segment(child, [])

        elif isinstance(op, _TEMPORAL_OPS):
            _walk_segment(op.child, current_segment)

        else:
            segments.append(current_segment)

    _walk_segment(root, [])
    return segments


def get_all_tubes(root: TubeNode) -> list[TubeNode]:
    """
    Walk the tTLT tree from root downward and return every TubeNode
    in DFS order, without copying or segmenting.
    """
    tubes:   list[TubeNode] = []
    visited: set[int]       = set()

    def _walk(tube: TubeNode) -> None:
        tid = id(tube)
        if tid in visited:
            return
        visited.add(tid)
        tubes.append(tube)

        op = tube.operator
        if op is None:
            return

        if isinstance(op, (AndOperatorNode, OrOperatorNode)):
            _walk(op.left)
            _walk(op.right)
        elif isinstance(op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
            _walk(op.child)

    _walk(root)
    return tubes


# --------------------
# Compression
# --------------------

def Compression(segments: list[list[TubeNode]], grid: Grid) -> TubeNode:
    """
    Algorithm 3

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572


    Given a list of MTS segments from get_MTS(), compress each segment into a
    single TubeNode by taking the pointwise union (minimum) of all SDFs in that
    segment, then rewire the compressed nodes into a new tree connected only by
    Boolean operator nodes (AND / OR).

    Works on deep copies of the TubeNodes so the original tTLT tree is
    never mutated.  The returned compressed root is a fully independent tree.

    Union (pointwise minimum)
    -------------------------
    For each MTS segment, iterate over every TubeNode in the segment:
      - If brs_full is not None, use brs_full[..., 0] as the static slice.
      - If sdf.ndim == grid.dims + 1, use sdf[..., 0] as the static slice.
      - Otherwise use sdf directly.
      - Take elementwise np.minimum across all static SDFs in the segment.

    Rewiring
    --------
    Each compressed TubeNode's .parent is already set from first_tube.parent
    in Step 1.  Step 2 walks all segments, finds those whose first tube owns
    a Boolean operator, attaches that operator downward from the compressed
    tube, and replaces op.left / op.right by collecting all compressed tubes
    whose .parent is that operator.

    Parameters
    ----------
    segments : list[list[TubeNode]]
        Output of get_MTS().
    grid : Grid
        HJ grid, used to detect the time dimension (sdf.ndim == grid.dims + 1).

    Returns
    -------
    compressed_root : TubeNode
        Root of the new compressed tTLT tree.
    """

    # Deep-copy all segments so rewiring never touches the originals
    segments = copy.deepcopy(segments)

    # ── Step 1: compress each MTS segment to a single static SDF ─────────
    compressed_tubes: list[TubeNode] = []

    for seg_idx, seg in enumerate(segments):

        union_sdf = None
        union_controlSet = None
        optCtrl_labels   = []

        for tube in seg:

            #Union of reachable set
            if tube.brs_full is not None:
                static = tube.brs_full[..., 0]
            elif tube.sdf.ndim == grid.dims + 1:
                static = tube.sdf[..., 0]
            else:
                static = tube.sdf

            union_sdf = static if union_sdf is None else np.minimum(union_sdf, static)

            #Union of Control set
            if tube.controlSet is None:
                continue
            elif tube.controlSet == 'feasible':
                union_controlSet = 'feasible'
            elif tube.controlSet == 'optCtrl':
                optCtrl_labels.append(tube.label)

        # Resolve final control set label (if there are optimal control)
        if optCtrl_labels:
            union_controlSet = "optCtrl(" + "|".join(optCtrl_labels) + ")"


        label   = "Union(" + ", ".join(tube.label for tube in seg) + ")"
        formula = f"compressed_MTS[{seg_idx}]({seg[0].formula})"

        new_tube        = TubeNode(sdf=union_sdf, formula=formula, brs_full=None)
        new_tube.label  = label
        new_tube.controlSet = union_controlSet
        new_tube.parent = seg[0].parent

        compressed_tubes.append(new_tube)

    # ── Step 2: rewire Boolean operator connections ───────────────────────
    for seg_idx, seg in enumerate(segments):
        op = seg[-1].operator
        if op is None or not isinstance(op, (AndOperatorNode, OrOperatorNode)):
            continue

        comp_tube          = compressed_tubes[seg_idx]
        comp_tube.operator = op
        op.tube            = comp_tube

        children = [ct for ct in compressed_tubes if ct.parent is op]
        if len(children) != 2:
            raise ValueError(
                f"Compression: expected 2 compressed children for Boolean op "
                f"'{op}' at MTS[{seg_idx}], found {len(children)}."
            )
        op.left  = children[0]
        op.right = children[1]

    compressed_root = compressed_tubes[0]
    return compressed_root


def tTLTSatisfaction(compressed_root: TubeNode):
    '''
    Algorithm 2

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    raise NotImplementedError

def Backtracking(compressed_root: TubeNode):
    '''
    Algorithm 4

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    raise NotImplementedError


# --------------------
# Print and Plot utilities
# --------------------

def plot_compressed_reachable_sets(
    compressed_root: TubeNode,
    grid:            Grid,
    brs_color:       str = "lightblue",
) -> None:
    """
    Visualize the union SDF stored in every compressed TubeNode of the
    compressed tTLT tree.  One figure is produced per compressed TubeNode,
    using the same style as HJtTLT._plot_result in stl_hj.py:

        2-D grid : go.Heatmap fill  (values <= 0)  +  go.Contour boundary
        3-D grid : go.Isosurface at the zero level-set

    Because compressed SDFs are static (no time axis), each figure shows a
    single frame — no animation slider is added.

    Parameters
    ----------
    compressed_root : TubeNode
        Root of the compressed tTLT tree returned by Compression().
    grid : Grid
        The HJ grid used during solving (needed for axis coordinates).
    brs_color : str
        Fill / surface colour for the reachable set  (default: "lightblue").
    """

    dims = grid.dims

    # ── collect all compressed TubeNodes (DFS) ────────────────────────────
    tubes:   list[TubeNode] = []
    visited: set[int]       = set()

    def _collect(tube: TubeNode) -> None:
        tid = id(tube)
        if tid in visited:
            return
        visited.add(tid)
        tubes.append(tube)
        op = tube.operator
        if op is None:
            return
        if isinstance(op, (AndOperatorNode, OrOperatorNode)):
            _collect(op.left)
            _collect(op.right)

    _collect(compressed_root)

    # ── plot each tube ────────────────────────────────────────────────────
    for tube in tubes:
        sdf   = tube.sdf
        label = tube.label if tube.label is not None else tube.formula

        if sdf is None:
            print(f"  [plot_compressed_reachable_sets] skipping '{label}' — sdf is None")
            continue

        print(f"  [plot_compressed_reachable_sets] plotting '{label}'  "
              f"sdf min={sdf.min():.4f}  max={sdf.max():.4f}  "
              f"fraction safe: {(sdf <= 0).mean():.4f}")

        if dims == 2:
            x_axis, y_axis = _make_2d_meshgrid(grid)

            sdf_masked = np.where(sdf <= 0, sdf, np.nan)
            fill_trace = go.Heatmap(
                x=x_axis, y=y_axis, z=sdf_masked.T,
                colorscale=[[0, brs_color], [1, brs_color]],
                zmin=-0.1, zmax=0.0,
                opacity=0.5, showscale=False,
                name=f"{label} (fill)",
            )

            boundary_trace = go.Contour(
                x=x_axis, y=y_axis, z=sdf.T,
                contours=dict(start=0, end=0, size=1, coloring="none"),
                contours_coloring="none",
                line=dict(color="blue", width=2),
                showscale=False,
                name=label,
            )

            fig = go.Figure(data=[fill_trace, boundary_trace])
            fig.update_layout(
                title=f"Compressed reachable set: {label}",
                xaxis_title="x",
                yaxis_title="y",
            )

        elif dims == 3:
            mg_X, mg_Y, mg_Z = _make_3d_meshgrid(grid)

            iso_trace = _iso_trace(mg_X, mg_Y, mg_Z, sdf, brs_color, label, opacity=0.6)

            fig = go.Figure(data=[iso_trace])
            fig.update_layout(
                title=f"Compressed reachable set: {label}",
                scene=dict(
                    xaxis={"nticks": 20},
                    zaxis={"nticks": 20},
                    camera_eye={"x": 0, "y": -1, "z": 0.5},
                    aspectratio={"x": 1, "y": 1, "z": 0.6},
                ),
            )

        else:
            print(f"  [plot_compressed_reachable_sets] grid dims={dims} not supported, skipping.")
            continue

        fig.show()


def print_MTS(segments: list[list[TubeNode]]) -> None:
    """
    Pretty-print the MTS segments returned by get_MTS().
    """
    print("=" * 60)
    print(f"  MTS decomposition — {len(segments)} segment(s)")
    print("=" * 60)
    for idx, seg in enumerate(segments):
        print(f"\n  MTS[{idx}]  ({len(seg)} tube node(s))")
        for depth, tube in enumerate(seg):
            indent  = "    " + "  " * depth
            op_name = tube.operator.__class__.__name__ if tube.operator else "leaf"
            print(f"{indent}[TubeNode] '{tube.label}'  op={op_name}  shape={tube.sdf.shape}")
    print("\n" + "=" * 60)


def print_compressed_tree(root: TubeNode, indent: int = 0) -> None:
    """
    Pretty-print the compressed tTLT tree returned by Compression().
    Only Boolean operator nodes (AND / OR) appear as structural connectors;
    temporal operators are implied inside each compressed TubeNode's SDF.
    """
    pad    = "  " * indent
    pad_op = "  " * (indent + 1)
    pad_ch = "  " * (indent + 2)

    shape  = root.sdf.shape if root.sdf is not None else "None"
    par    = root.parent.name if root.parent else "None"
    print(f"{pad}[CompressedTubeNode]  '{root.label}'  shape={shape}  parent={par}")

    op = root.operator
    if op is None:
        print(f"{pad_op}(leaf — no operator child)")
        return

    if isinstance(op, (AndOperatorNode, OrOperatorNode)):
        op_name = "AND" if isinstance(op, AndOperatorNode) else "OR"
        print(f"{pad_op}|")
        print(f"{pad_op}[{op_name}OperatorNode]")
        print(f"{pad_ch}|-- left:")
        print_compressed_tree(op.left,  indent + 3)
        print(f"{pad_ch}|-- right:")
        print_compressed_tree(op.right, indent + 3)
    else:
        print(f"{pad_op}(unexpected operator: {op.__class__.__name__})")



# --------------------
# Online Control Synthesis Algorithm for tube-based Temporal Logic Tree
# --------------------

def Initialization(root: TubeNode, tk_arr: np.ndarray, t0: float = 0) -> list[TubeNode]:
    '''
    Algorithm 6

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    _BOOLEAN_OPS  = (AndOperatorNode, OrOperatorNode)
    _TEMPORAL_OPS = (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)

    #Definition 5.1
    def _assign_time_horizon(tube: TubeNode, t0: float = 0) -> None:
        op = tube.operator

        #leaf node
        if op is None:
            tube.th = np.inf
            return

        if isinstance(op, _BOOLEAN_OPS):
            tube.th = t0
            for child in [op.left, op.right]:
                _assign_time_horizon(child, t0 = tube.th)

        elif isinstance(op, _TEMPORAL_OPS):
            b_hat = tk_arr[(tk_arr >= op.a) & (tk_arr <= op.b)].max()
            tube.th = t0 + b_hat
            _assign_time_horizon(op.child, t0 = tube.th)


    #Definition 5.2
    def _check_reachability_of_boolean_segment(tube: TubeNode, reachable_from_tube: list[TubeNode]) -> list[TubeNode]:

        op = tube.operator
        reachable_from_tube.append(tube)

        if op is None:                              # leaf — stop
            return reachable_from_tube

        if isinstance(op, _BOOLEAN_OPS):            # Boolean — recurse into children
            for child in [op.left, op.right]:
                _check_reachability_of_boolean_segment(child, reachable_from_tube)

        # Temporal op or anything else — stop here (boundary of Boolean cluster)

        return reachable_from_tube

    #set root time horizon to initial time
    root.th = t0
    _assign_time_horizon(root, t0=t0)

    PostSet = _check_reachability_of_boolean_segment(root, [])

    #assign activation time
    for X_j in PostSet:
        X_j.ta = t0

    return PostSet



def trackingSetNode(PostSet: list[TubeNode], state:np.ndarray, grid:Grid, t: float = 0, dt: float = 1) -> list[TubeNode]:
    '''
    Algorithm 7

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    _BOOLEAN_OPS  = (AndOperatorNode, OrOperatorNode)
    _TEMPORAL_OPS = (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)


    # Valid Labelling function
    B_xt: list[TubeNode] = []
    for tube in PostSet:

        # B(x,t) == L(x,t) ∩ PostSet(x-, t-)
        if t <= tube.th:

            #Check if (x,t) lies inside the reachable set of the TubeNode
            op = tube.operator
            if isinstance(op, _TEMPORAL_OPS):
                horizon = op.b - op.a #NOTE choose reachable set horizon to be [0,b]
                tau     = np.arange(0, horizon + 1e-8, dt)

                # #OLD METHOD #TODO verify this time encoding again
                # t_index = np.argmin(np.abs(tau - t))

                # Use relative time (elapsed since activation) to match updatetTLT convention.
                # When ta is None the tube has not been activated yet; use 0 so the full-horizon
                # BRS slice (index 0) is checked on first entry.
                relative_t = (t - tube.ta) if (tube.ta is not None) else 0.0
                t_index = np.argmin(np.abs(tau - relative_t))

                #Find time index w.r.t the reachable set
                if tube.brs_full is not None:
                    sdf = tube.brs_full[..., t_index]
                    tube.t_index = t_index
                else:
                    sdf = tube.sdf

            #For boolean opeartor
            else:
                sdf = tube.sdf


            axes = tuple(
                np.linspace(grid.min[d], grid.max[d], grid.pts_each_dim[d])
                for d in range(grid.dims)
            )

            interp = RegularGridInterpolator(
                axes,
                sdf,
                method       = "linear",
                bounds_error = False,
                fill_value   = None,
            )
            value = float(interp(np.array(state).reshape(1, -1)))

            print("tube", tube.label, "value", value)
            
            #negative value funciton means that the state lies inside reachable set
            if value <= 0.0:
                B_xt.append(tube)
    
    print("    ---trackingSetNode: B_xt:", B_xt)


    #only contains at most one set node for each complete path of tTLT graph
    for tube_i in B_xt.copy():

        Post_tube_i = {t.label for t in Post(tube_i)}

        for tube_j in B_xt.copy():

            if tube_i.label != tube_j.label and tube_j.label in Post_tube_i:
                if tube_i.label in {t.label for t in B_xt}:
                    
                    B_xt.remove(tube_i)

    return B_xt


def updatetTLT(B_xt: list[TubeNode], root:TubeNode, tk_arr:np.ndarray, t: float, dt: float):
    '''
    Algorithm 8

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    _BOOLEAN_OPS  = (AndOperatorNode, OrOperatorNode)
    _TEMPORAL_OPS = (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)

    #Get all set Nodes
    all_tubes = get_all_tubes(root=root)

    for tube in all_tubes:

        child_op = Children(tube) #if a tube node is not a leaf node, its unique child is an operator node
        child_op  = child_op[0] if child_op else None #This is because child_op is a list

        if (child_op is not None) and isinstance(child_op, _TEMPORAL_OPS):
            child_horizon = tk_arr[(tk_arr >= child_op.a) & (tk_arr <= child_op.b)].max() #Definition 5.1
        else:
            child_horizon = 0

        if (tube in B_xt) and ((tube.ta + child_horizon) >= (t+dt)):

            if (child_op is not None) and isinstance(child_op, _TEMPORAL_OPS):
                horizon = child_op.b #- child_op.a #NOTE choose reachable set horizon to be [0,b]
                tau     = np.arange(0, horizon + 1e-8, dt)
                t_index = np.argmin(np.abs(tau  - (t-tube.ta)  ))   #every tube in B_xt should already have activation time "ta" assigned

                #Find time index w.r.t the reachable set
                if tube.brs_full is not None:
                    tube.sdf = tube.brs_full[..., t_index]  #tube.sdf keeps track of the most current time slice from tube.brs_full
                    tube.t_index = t_index

            #Skip if child_op is _BOOLEAN_OPS, since the sdf doesn't changes

        #TODO Add else statement from the paper (update time)

def get_spat_deriv_at_state(g, deriv_arrays, state):
        """
        Interpolate precomputed spatial derivatives at a given state.
        """
        pt = state.reshape(1, -1)
        derivs = []
        for deriv_array in deriv_arrays:
            interp = RegularGridInterpolator(
                tuple(g.grid_points),
                deriv_array,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            derivs.append(float(interp(pt)))

        return tuple(derivs)

            
def buildControlTree(B_xt: list[TubeNode], root:TubeNode, state:np.ndarray, g:Grid, dyn:DubinsCar2|Plane2D, dt:float):
    '''
    Algorithm 9

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    #Get all set Nodes
    all_tubes = get_all_tubes(root=root)
    B_xt_labels = {tube.label for tube in B_xt}   # set of label strings for O(1) lookup

    for tube in all_tubes:

        if tube.label in B_xt_labels:

            child_op = Children(tube) #if a tube node is not an end-leaf node, its unique child is an operator node
            child_op  = child_op[0] if child_op else None #child_op is a list

            if child_op is None: #the tube is a leaf node
                tube.controlSet = 'feasible'   #choose any feasible set of controls
            else:

                tube.controlSet = 'optCtrl'
                
                #TODO in the paper, this should be a robust control invariant set (subset of the reachable set at the current time)                   -> solve robust CBF for "Grad_x(V) * f(x,u) <= d/dt(V)""
                #     in current implementation, we compute a single optimal control in the most negative gradient direction of the HJ value function -> solve u* = argmax Grad_x(V) * f(x,u)

                # #-----------------------------------------
                # #NOTE this method is not sound/incorrect 
                #
                # #NOTE  We apply zero control against worst-case disturbance,
                # #  If the state is still inside the reachable set at next time step -> control set can be any inside feasible limit
                # #  If the state lies outside the reachable set at next time step -> must use optimal control from the reachable set
                # #-----------------------------------------
                
                # #==== Update statat with zero control against worst-case disturbance ====
                # brt = tube.sdf # This should already be set to current time index in updatetTLT()

                # # Spatial derivatives and optimal control at a sample state
                # deriv_arrays = [
                #     computeSpatDerivArray(g, brt, deriv_dim=dim, accuracy="medium")
                #     for dim in range(1, g.dims + 1)
                # ]

                # spat_deriv = get_spat_deriv_at_state(g, deriv_arrays, state)
                # opt_dstb = dyn.optDstb_inPython(state, spat_deriv) #numpy array

                # state_dot = solver.dynamics.dynamics_inPython(state, np.zeros(g.dims), disturbance=opt_dstb)  # returns tuple of any length
                # state_next = state + np.array(state_dot) * dt


                # #==== Get reachable set at next time step =====
                # if tube.brs_full is not None and tube.t_index + 2 < tube.brs_full.shape[-1]:
                #     sdf = tube.brs_full[..., tube.t_index + 2]
                # else:
                #     sdf = tube.sdf


                # axes = tuple(
                #     np.linspace(g.min[d], g.max[d], g.pts_each_dim[d])
                #     for d in range(g.dims)
                # )

                # interp = RegularGridInterpolator(
                #     axes,
                #     sdf,
                #     method       = "linear",
                #     bounds_error = False,
                #     fill_value   = None,
                # )
                # value = float(interp(np.array(state_next).reshape(1, -1)))

                # # If the state is still inside the reachable set at next time step -> control set can be any inside feasible limit
                # if value <= 0.0:
                #     tube.controlSet = 'feasible'
                # else:
                #     tube.controlSet = 'optCtrl'    #gradient-based (of the value function) optimal control


        #If tube is not in active set B_xt, then controlset is empty for now
        else:

            tube.controlSet = None


def Backtracking(root: TubeNode, compressed_root: TubeNode, state:np.ndarray, g:Grid, dyn:DubinsCar2|Plane2D):
    '''
    Algorithm 10

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    #First get the set of all control given the compressed tree
    dims = g.dims

    def _parse_controlSet(root:TubeNode, tube: TubeNode, state:np.ndarray, g:Grid, dyn:DubinsCar2|Plane2D):

        if tube.controlSet is None:
            return np.empty((0, g.dims)), np.inf


        elif tube.controlSet == "feasible":

                #TODO 'feasible' control set means that we are at the leaf node
                # For now, apply zero control (assume no disturbance) should guarantee we stay inside the leaf nodes
                return np.zeros((1, g.dims)), np.inf

                # n_u = 2

                # brt = tube.sdf # This should already be set to correct time index in updatetTLT()

                # # Spatial derivatives and optimal control at a sample state
                # deriv_arrays = [
                #     computeSpatDerivArray(g, brt, deriv_dim=dim, accuracy="medium")
                #     for dim in range(1, g.dims + 1)
                # ]

                # spat_deriv = get_spat_deriv_at_state(g, deriv_arrays, state)

                # # Get worst-case disturbance at current state
                # opt_dstb = dyn.optDstb_inPython(state, spat_deriv)

                # # Build candidate control set
                # if dims == 2 and isinstance(dyn, Plane2D):
                #     vx_samples = np.linspace(dyn.vxMin, dyn.vxMax, n_u)
                #     vy_samples = np.linspace(dyn.vyMin, dyn.vyMax, n_u)
                #     vx_grid, vy_grid = np.meshgrid(vx_samples, vy_samples)
                #     candidate_ctrls = np.column_stack([vx_grid.ravel(), vy_grid.ravel()])

                # elif dims == 3 and isinstance(dyn, DubinsCar2):
                #     speed_samples = np.linspace(dyn.speedMin, dyn.speedMax, n_u)
                #     w_samples     = np.linspace(dyn.wMin,     dyn.wMax,     n_u)
                #     sp_grid, w_grid = np.meshgrid(speed_samples, w_samples)
                #     candidate_ctrls = np.column_stack([sp_grid.ravel(), w_grid.ravel()])

                # #  Simulate one step forward for each candidate control
                # #  and keep only the ones that stay inside the reachable set
                # feasible_ctrls = []


                # axes = tuple(
                #     np.linspace(g.min[d], g.max[d], g.pts_each_dim[d])
                #     for d in range(g.dims)
                # )

                # interp = RegularGridInterpolator(
                #     axes,
                #     brt,
                #     method       = "linear",
                #     bounds_error = False,
                #     fill_value   = None,
                # )

                # for u in candidate_ctrls:
                #     state_dot  = dyn.dynamics_inPython(state, u, disturbance=opt_dstb)
                #     next_state = state + np.array(state_dot) * dt

                #     # V <= 0 means inside the reachable set
                #     if float(interp(np.array(next_state).reshape(1, -1))) <= 0:
                #         feasible_ctrls.append(u)

                # return np.array(feasible_ctrls)  # shape (n_feasible, n_controls)

        elif tube.controlSet.startswith("optCtrl"):

            inner = tube.controlSet[len("optCtrl("):-1]   # "X1|X3|X5"
            labels = inner.split("|") # e.g. get the tube node label X1, X3. X5

            #Get all tube nodes
            all_tubes = get_all_tubes(root=root)

            # Build lookup dict by label
            tube_by_label = {tube.label: tube for tube in all_tubes}

            # Access by label name
            referenced_tubes = [tube_by_label[lbl] for lbl in labels]


            # ==========================
            #TODO for now, instead of collect all optimal control from the compressed tube, we just choose one optimal control that has the earliest deadline (t_h) according to STL formula
            # ==========================

            # gradient_controls = []

            # for tube_i in referenced_tubes:

            #     #Single Integrator or Dubins Car
            #     if (dims == 2 and isinstance(dyn, Plane2D)) or (dims == 3 and isinstance(dyn, DubinsCar2)):

            #         brt = tube_i.sdf # This should already be set to correct time index in updatetTLT()

            #         # Spatial derivatives and optimal control at a sample state
            #         deriv_arrays = [
            #             computeSpatDerivArray(g, brt, deriv_dim=dim, accuracy="medium")
            #             for dim in range(1, g.dims + 1)
            #         ]

            #         spat_deriv = get_spat_deriv_at_state(g, deriv_arrays, state)
            #         opt_ctrl = dyn.optCtrl_inPython(state, spat_deriv)

            #         gradient_controls.append(opt_ctrl)

            # admissCtrlSet = np.array(gradient_controls)
            # return admissCtrlSet




            # find the tube whose horizon ends soonest
            nearest_tube = min(referenced_tubes, key=lambda t: t.th)

            #Single Integrator or Dubins Car
            if (dims == 2 and isinstance(dyn, Plane2D)) or (dims == 3 and isinstance(dyn, DubinsCar2)):

                brt          = nearest_tube.sdf
                deriv_arrays = [
                    computeSpatDerivArray(g, brt, deriv_dim=dim, accuracy="medium")
                    for dim in range(1, g.dims + 1)
                ]
                spat_deriv = get_spat_deriv_at_state(g, deriv_arrays, state)
                opt_ctrl   = dyn.optCtrl_inPython(state, spat_deriv)

                admissCtrlSet = np.array([opt_ctrl])   # shape (1, dims)

            else:

                raise("Invalid dynamics")
            
            return admissCtrlSet, nearest_tube.th



            

    def _backtrack_boolean(tube: TubeNode, root:TubeNode, state:np.ndarray, g:Grid, dyn:DubinsCar2|Plane2D):
        
        op = tube.operator
        parent_admissControlSet, th = _parse_controlSet(root=root, tube=tube, state=state, g=g, dyn=dyn)
        parent_set = set(map(tuple, parent_admissControlSet)) 
        
        #at end leaf node
        if op is None:
            return np.array(list(parent_admissControlSet)), th


        if isinstance(op, AndOperatorNode):
            left_admissControlSet, left_th = _backtrack_boolean(op.left, root=root, state=state, g=g, dyn=dyn)
            right_admissControlSet, right_th = _backtrack_boolean(op.right, root=root, state=state, g=g, dyn=dyn)

            #Intersection of two control sets
            left_set = set(map(tuple, left_admissControlSet))
            right_set = set(map(tuple, right_admissControlSet))

            print("AND: (left, th)", (left_set, left_th))
            print("AND: (right, th)", (right_set, right_th))


            # METHOD 1 -----------------------------------

            # if op.left.controlSet is None or op.right.controlSet is None:
            #     return np.empty((0, g.dims))

            # elif op.left.controlSet.startswith("optCtrl") and op.right.controlSet.startswith("optCtrl"):
            #     intersected_set = (left_set & right_set)
            
            # else:
            #     intersected_set = (left_set | right_set) #NOTE Since feasible control set can be anything, optCtrl should also be in feasible set

            # #Union with parents
            # result = parent_set | intersected_set


            # METHOD 2 -----------------------------------
            #TODO: In the paper, this should be Intersection, not Union for left and right control set (Check how can we return "feasible control set")

            # #Union with parents
            # result = parent_set | (left_set | right_set) 

            # return np.array(list(result)) if result else np.empty((0, g.dims))


            # METHOD 3 -----------------------------------
            # Return only one optimal control based on the earliest time horizon

            if left_set and left_th <= right_th:

                result = parent_set | left_set
                result_th = min(left_th, th)

            else:

                result = parent_set | right_set
                result_th = min(right_th, th)

            return (np.array(list(result)), result_th) if result else (np.empty((0, g.dims)), result_th)


        #NOTE for OR operator, the union of control is safe; no risk of returning empty set
        elif isinstance(op, OrOperatorNode):
            left_admissControlSet, left_th = _backtrack_boolean(op.left, root=root, state=state, g=g, dyn=dyn)
            right_admissControlSet, right_th = _backtrack_boolean(op.right, root=root, state=state, g=g, dyn=dyn)

            #Union of the two set
            left_set  = set(map(tuple, left_admissControlSet))
            right_set = set(map(tuple, right_admissControlSet))

             #Untion with parents
            result = parent_set | left_set | right_set
            # return np.array(list(result)) if result else np.empty((0, g.dims))
            return (np.array(list(result)), min(th, left_th, right_th)) if result else (np.empty((0, g.dims)), min(th, left_th, right_th))


    admissCtrlSet, _ = _backtrack_boolean(tube=compressed_root, root=root, state=state, g=g, dyn=dyn)
    return admissCtrlSet


def postSet(B_xt:list[TubeNode], t:float, root: TubeNode):
    '''
    Algorithm 11

    Online control synthesis for uncertain systems under signal temporal logic speciﬁcations (Yu, et. al., 2023)
    https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572
    '''

    #Definition 5.2
    def _check_reachability_of_boolean_segment(tube: TubeNode, reachable_from_tube: list[TubeNode]) -> list[TubeNode]:

        op = tube.operator
        reachable_from_tube.append(tube)

        if op is None:                              # leaf — stop
            return reachable_from_tube

        if isinstance(op, _BOOLEAN_OPS):            # Boolean — recurse into children
            for child in [op.left, op.right]:
                _check_reachability_of_boolean_segment(child, reachable_from_tube)

        # Temporal op or anything else — stop here (boundary of Boolean cluster)

        return reachable_from_tube

    _BOOLEAN_OPS  = (AndOperatorNode, OrOperatorNode)
    _TEMPORAL_OPS = (EventuallyOperatorNode, UntilOperatorNode)

    PostSet: list[TubeNode] = []

    # print("B_xt (Input to postSet)", B_xt)
    all_tubes = get_all_tubes(root=root)

    for S_i in B_xt:

        op  = S_i.operator
        # op_par = S_i.parent

        if isinstance(op, _BOOLEAN_OPS):
            
            PostSet.append(S_i)

            Post_Si = _check_reachability_of_boolean_segment(S_i, [])
            for S_j in Post_Si:
                PostSet.append(S_j)


        elif isinstance(op, _TEMPORAL_OPS):

            PostSet.append(S_i) #TODO This is inside if-statement in the paper

            PreNode = Pre(S_i)
            PostNode = Post(S_i)
            if PreNode is not None and (t >= (S_i.ta + op.a)):

                if PostNode:
                    for node in PostNode:
                        PostSet.append(node)


        elif isinstance(op, GloballyOperatorNode):

            PostSet.append(S_i) #TODO This is inside if-statement in the paper

            PreNode = Pre(S_i)
            PostNode = Post(S_i)
            if PreNode is not None and (t >= (S_i.ta + op.b)):

                if PostNode:
                    for node in PostNode:
                        PostSet.append(node)
    
        elif op is None:

            PostSet.append(S_i)

    # Deduplicate by label (preserves order, per label-identity convention)
    seen_labels: set[str] = set()
    unique_PostSet: list[TubeNode] = []
    for tube in PostSet:
        lbl = tube.label
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            unique_PostSet.append(tube)

    return unique_PostSet


         



# --------------------
# Debuggin
# --------------------

if __name__ == "__main__":

    #Paremeter 
    t = 0.0 #intial time
    x = np.array([0.5, 0.8]) #initial state
    dt = 0.5 #time step for reachable set computation

    # ─────────────────────────────────────────────────────────────

    # ── Plane2D  F[5,10](G[0,10] mu_1) AND (mu_2 U[0,8] mu_3) ──
    result = tTLT(
        stl_yaml_path  = "configs/stl_spec_hj.yaml",
        hj_config_path = "configs/hj_config.yaml",
    )
    solver = HJtTLT(result, "configs/hj_config.yaml")
    solver.solve_all(dt=dt, accuracy="medium", plot=False, verbose=False)
    stl_horizon = STLHorizonEvaluator("configs/stl_spec_hj.yaml").evaluate()


    # # ── DubinsCar  (mu_1 AND NOT mu_2) U[0,12] G[0,8](mu_3) ────
    # result = tTLT(
    #     stl_yaml_path  = "configs/stl_spec_dubins.yaml",
    #     hj_config_path = "configs/hj_config_dubins.yaml",
    # )
    # solver = HJtTLT(result, "configs/hj_config_dubins.yaml")
    # solver.solve_all(dt=1, accuracy="medium", plot=True, verbose=False)
    # stl_horizon = STLHorizonEvaluator("configs/stl_spec_dubins.yaml").evaluate()

    # ─────────────────────────────────────────────────────────────

    # A maximal temporal segment
    maxTempSeg = get_MTS(solver.root)
    print_MTS(maxTempSeg)

    # compressed_root = Compression(maxTempSeg, result["grid"])

    # print("\n" + "=" * 60)
    # print("  COMPRESSED TREE STRUCTURE")
    # print("=" * 60)
    # print_compressed_tree(compressed_root)
    # print("=" * 60)

    # # plot compressed tTLT tree structure
    # plot_ttlt_tree(compressed_root)

    # # plot each compressed tube's union reachable set
    # plot_compressed_reachable_sets(compressed_root, result["grid"])

    # # ─────────────────────────────────────────────────────────────

    # --- GIF stuff ---
    gif_frames = [] 
    gif_path   = "tube_traversal.gif" 
    save_gif = True

    # ─────────────────────────────────────────────────────────────

    #Plot Settings

    fig, ax       = plt.subplots(figsize=(7, 6))
    state_history = [x]   # list of (x0, x1) tuples
    plt.ion()            # interactive mode on
    g = solver.grid
    x_coords = g.grid_points[0]
    y_coords = g.grid_points[1]
    extent   = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
    X, Y     = np.meshgrid(x_coords, y_coords)

    brs_colors  = ["lightblue", "gold", "lightgreen", "lightsalmon", "plum"]
    line_colors = ["blue", "goldenrod", "darkgreen", "darkorange", "purple"]


    #Initial Plot
    brs    = solver.root.sdf
    z      = brs[:, :, 0] if brs.ndim == 3 else brs
    z_plot = z.T
    z_masked = np.where(z_plot <= 0, 0.0, np.nan)

    fill_color = brs_colors[-1]
    line_color = line_colors[-1]

    ax.imshow(
        z_masked,
        extent=extent, origin="lower", aspect="auto",
        cmap=ListedColormap([fill_color]),
        vmin=-0.5, vmax=0.5,
        alpha=0.4, interpolation="nearest",
    )
    ax.contour(X, Y, z_plot, levels=[0.0],
                colors=line_color, linewidths=2,
                label=f"BRS {0}")

    # predicate boundaries
    predicate_tubes = [e["sdf"] for e in solver._predicate_cache.values()]
    for sdf in predicate_tubes:
        sdf_2d = sdf[:, :, 0] if sdf.ndim == 3 else sdf
        ax.contour(X, Y, sdf_2d.T, levels=[0.0],
                    colors="green", linewidths=1.5, linestyles="--")

    # state trajectory history
    if len(state_history) > 1:
        xs = [s[0] for s in state_history]
        ys = [s[1] for s in state_history]
        ax.plot(xs, ys, "r--o", markersize=4, linewidth=1,
                label="trajectory", zorder=5)

    # current state
    ax.plot(x[0], x[1], "ro", markersize=8,
            label=f"state t={t:.3g}", zorder=6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Reachable Set at t={t:.3g}\n"
                 f"state = {np.round(x, 3)}\n"
                 f"Root Node = X0")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    # ---- capture frame ----
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    gif_frames.append(Image.open(buf).copy())
    buf.close()
    # -----------------------

    plt.pause(1)


    # ─────────────────────────────────────────────────────────────

    #Online Control Synthesis
    tk_array = np.arange(0, stl_horizon + 1e-8, dt)

    # Initialization
    PostSet = Initialization(solver.root, tk_arr=tk_array, t0=t)
    print("Initialization")
    print("Post(B_xt-)", PostSet)

    all_tubes = get_all_tubes(root=solver.root)
    print(f"  {'Label':<6}  {'ta':<10}  {'th':<10}   ")#  Formula")
    print("  " + "-" * 56)
    for tube in all_tubes:
        label   = tube.label   if tube.label   is not None else "—"
        ta_str  = f"{tube.ta:.4f}" if tube.ta  is not None else "None"
        th_str  = f"{tube.th:.4f}" if tube.th  != np.inf  else "inf"
        formula = tube.formula if tube.formula             else "—"
        print(f"  {label:<6}  {ta_str:<10}  {th_str:<10}  ") #{formula}")




    # Main algorithm

    print("--------------")

    for t in tk_array:

        print("t", t)

        B_xt = trackingSetNode(PostSet=PostSet, state=x, grid=solver.grid, t=t, dt=dt)
        print("B_xt (trackingSetNode)", B_xt)

        for S_t in B_xt:
            if S_t.ta is None:
                S_t.ta = t #set activation time

        updatetTLT(B_xt=B_xt, root=solver.root, tk_arr=tk_array, t=t, dt=dt)

        buildControlTree(B_xt=B_xt, root=solver.root, state=x, g=solver.grid, dyn=solver.dynamics, dt=dt)
        print("B_xt (updatetTLT & buildControlTree)", B_xt)

        compressed_root = Compression(maxTempSeg, solver.grid)
        print("Compressed Tree", get_all_tubes(root=compressed_root))

        admissCtrlSet = Backtracking(root=solver.root, compressed_root=compressed_root, state=x, g=solver.grid, dyn=solver.dynamics)
        print("Admissable Control", admissCtrlSet)

        #TODO small issue with Backtracking
        if admissCtrlSet.shape[1] == 0:
            plt.ioff()
            plt.show(block=True)
            raise ValueError("No admissible control found.")

        # First control vector,
        u_opt = admissCtrlSet[0]
        print("Control", u_opt)

        # Update State
        state_dot = solver.dynamics.dynamics_inPython(x, u_opt)  # returns tuple of any length
        x = x + np.array(state_dot) * dt
        state_history.append(tuple(x))
        print("Updated state", x)


        PostSet = postSet(B_xt=B_xt, t=t, root=solver.root)
        print("Post Set", PostSet)

        all_tubes = get_all_tubes(root=solver.root)
        print(f"  {'Label':<6}  {'ta':<10}  {'th':<10}   ")#  Formula")
        print("  " + "-" * 56)
        for tube in all_tubes:
            label   = tube.label   if tube.label   is not None else "—"
            ta_str  = f"{tube.ta:.4f}" if tube.ta  is not None else "None"
            th_str  = f"{tube.th:.4f}" if tube.th  != np.inf  else "inf"
            formula = tube.formula if tube.formula             else "—"
            print(f"  {label:<6}  {ta_str:<10}  {th_str:<10}  ") #{formula}")

        print("--------------")



        # ---------- plot reachable set + current state ----------

        ax.clear()

        for idx, S_t in enumerate(B_xt):
            if not (hasattr(S_t, 'sdf') and S_t.sdf is not None):
                continue
            brs    = S_t.sdf
            z      = brs[:, :, 0] if brs.ndim == 3 else brs
            z_plot = z.T
            z_masked = np.where(z_plot <= 0, 0.0, np.nan)

            fill_color = brs_colors[idx % len(brs_colors)]
            line_color = line_colors[idx % len(line_colors)]

            ax.imshow(
                z_masked,
                extent=extent, origin="lower", aspect="auto",
                cmap=ListedColormap([fill_color]),
                vmin=-0.5, vmax=0.5,
                alpha=0.4, interpolation="nearest",
            )
            ax.contour(X, Y, z_plot, levels=[0.0],
                       colors=line_color, linewidths=2,
                       label=f"BRS {idx}")

        # predicate boundaries
        predicate_tubes = [e["sdf"] for e in solver._predicate_cache.values()]
        for sdf in predicate_tubes:
            sdf_2d = sdf[:, :, 0] if sdf.ndim == 3 else sdf
            ax.contour(X, Y, sdf_2d.T, levels=[0.0],
                       colors="green", linewidths=1.5, linestyles="--")

        # state trajectory history
        if len(state_history) > 1:
            xs = [s[0] for s in state_history]
            ys = [s[1] for s in state_history]
            ax.plot(xs, ys, "r--o", markersize=4, linewidth=1,
                    label="trajectory", zorder=5)

        # current state
        ax.plot(x[0], x[1], "ro", markersize=8,
                label=f"state t={t:.3g}", zorder=6)

        ax.set_xlabel("x")
        ax.set_ylabel("y")


        bxt_labels  = ", ".join(S_t.label for S_t in B_xt  if hasattr(S_t, "label"))
        post_labels = ", ".join(S_t.label for S_t in PostSet if hasattr(S_t, "label"))
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

        # ---- capture frame ----
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())
        buf.close()
        # -----------------------

        plt.pause(1)
        # --------------------------------------------------------

    plt.ioff()

    # ---- write GIF after the loop ----
    if gif_frames and save_gif:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=500,   # ms per frame — adjust as needed
            loop=0,         # 0 = loop forever
        )
        print(f"GIF saved to {os.path.abspath(gif_path)}")
    # ----------------------------------

    plt.show(block=True)


