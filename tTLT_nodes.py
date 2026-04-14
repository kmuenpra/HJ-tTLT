"""
ttlt_nodes.py
Defines the two node types for the Tube-based Temporal Logic Tree (tTLT):

    TubeNode     — wraps a BRS/SDF array; holds bidirectional pointers to
                   its parent OperatorNode (above) and child OperatorNode (below)

    OperatorNode — base class for AND, OR, G, F, U operator nodes; holds
                   bidirectional pointers to its owner TubeNode (above) and
                   its child TubeNode(s) (below)

Pointer directions
------------------
    TubeNode.parent      upward   --> OperatorNode that this tube is a child of
    TubeNode.operator    downward --> OperatorNode that this tube owns as child
    OperatorNode.tube    upward   --> TubeNode that owns this operator
    OperatorNode.left / .right    downward (AND, OR)  --> child TubeNodes
    OperatorNode.child            downward (G, F, U)  --> child TubeNode

Use wire() to set all six pointers in one call after constructing a new layer.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from stl_robustness import STLNode



# --------------------
# TubeNode
# --------------------

class TubeNode:
    """
    Wraps a single BRS / SDF array.

    Convention: sdf <= 0 means safe / feasible.

    Attributes
    ----------
    sdf      : np.ndarray
    label    : str                    Human-readable description.
    parent   : OperatorNode | None    Upward: the operator this tube is a child of.
    operator : OperatorNode | None    Downward: the operator this tube owns.
    """

    def __init__(
        self,
        sdf:      np.ndarray,
        formula:    str                   = "",
        brs_full: np.ndarray | None     = None,
    ) -> None:
        self.sdf:      np.ndarray             = sdf
        self.brs_full: np.ndarray | None      = brs_full
        self.formula:  str                    = formula
        self.label:    str                    = None
        self.parent:   Optional[OperatorNode] = None
        self.operator: Optional[OperatorNode] = None
        self.ta = None  #activation time of the tube
        self.th = np.inf  #time horizon of the tube (when deactivated)
        self.controlSet: str = None #optimal control is set as string: 'feasible' || 'optCtrl'
        self.t_index: int = 0

    def __repr__(self) -> str:
        shape = self.sdf.shape if self.sdf is not None else "None"
        par   = self.parent.name   if self.parent   else "None"
        op    = self.operator.name if self.operator else "None"
        return (
            f"TubeNode(label={self.label!r}, ctrlSet={self.controlSet}, t_index={self.t_index}, "# shape={shape}, "
            f"parent={par}, operator={op})"
        )


# --------------------
# OperatorNode  (abstract base)
# --------------------

class OperatorNode:
    """
    Base class for all operator nodes.

    Attributes
    ----------
    tube : TubeNode | None    Upward: the TubeNode that owns this operator.
    """

    def __init__(self, stl_node:STLNode) -> None:
        self.stl_node = stl_node
        self.tube: Optional[TubeNode] = None


    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.name


# --------------------
# Boolean operators  (two TubeNode children)
# --------------------

class AndOperatorNode(OperatorNode):
    """
    AND (intersection).

    Upward   : .tube          --> combined root TubeNode  (max of left, right)
    Downward : .left, .right  --> left and right root TubeNodes from sub-tTLTs
    """

    def __init__(self, left: TubeNode, right: TubeNode, stl_node: STLNode = None) -> None:
        super().__init__(stl_node=stl_node)
        self.left:  TubeNode = left
        self.right: TubeNode = right

    def __repr__(self) -> str:
        return f"AND(left={self.left.label!r}, right={self.right.label!r})"


class OrOperatorNode(OperatorNode):
    """
    OR (union).

    Upward   : .tube          --> combined root TubeNode  (min of left, right)
    Downward : .left, .right  --> left and right root TubeNodes from sub-tTLTs
    """

    def __init__(self, left: TubeNode, right: TubeNode, stl_node: STLNode = None) -> None:
        super().__init__(stl_node=stl_node)
        self.left:  TubeNode = left
        self.right: TubeNode = right

    def __repr__(self) -> str:
        return f"OR(left={self.left.label!r}, right={self.right.label!r})"


# --------------------
# Temporal operators  (single TubeNode child)
# --------------------

class GloballyOperatorNode(OperatorNode):
    """
    G[a,b] operator.

    Upward   : .tube   --> TubeNode holding the invariance BRS
    Downward : .child  --> child TubeNode (SDF of mu or sub-tTLT root)
    """

    def __init__(self, a: float, b: float, child: TubeNode, stl_node: STLNode = None) -> None:
        super().__init__(stl_node=stl_node)
        self.a:     float    = a
        self.b:     float    = b
        self.child: TubeNode = child

    def __repr__(self) -> str:
        return f"G[{self.a},{self.b}](child={self.child.label!r})"


class EventuallyOperatorNode(OperatorNode):
    """
    F[a,b] operator.

    Upward   : .tube   --> TubeNode holding the BRS
    Downward : .child  --> child TubeNode (target SDF or sub-tTLT root)
    """

    def __init__(self, a: float, b: float, child: TubeNode, stl_node: STLNode = None) -> None:
        super().__init__(stl_node=stl_node)
        self.a:     float    = a
        self.b:     float    = b
        self.child: TubeNode = child

    def __repr__(self) -> str:
        return f"F[{self.a},{self.b}](child={self.child.label!r})"


class UntilOperatorNode(OperatorNode):
    """
    U[a,b] operator.

    Upward   : .tube   --> TubeNode holding the reach-avoid BRS
    Downward : .child  --> child TubeNode for psi (reach target SDF).
                          phi (stay constraint) is consumed internally in
                          solve_until() and does not appear as a child tube.
    """

    def __init__(self, a: float, b: float, child: TubeNode, stl_node: STLNode = None) -> None:
        super().__init__(stl_node=stl_node)
        self.a:     float    = a
        self.b:     float    = b
        self.child: TubeNode = child

    def __repr__(self) -> str:
        return f"U[{self.a},{self.b}](child={self.child.label!r})"


# --------------------
# Wiring helper
# --------------------

def _ttlt_collect_layout(
    root: TubeNode,
) -> tuple[dict, dict, dict, list[str]]:
    """
    Walk the tTLT tree and collect:
        positions   : id(obj) -> (x, y)   — computed bottom-up
        labels      : id(obj) -> str       — display label
        kinds       : id(obj) -> 'tube'|'op'
        edges       : list of (parent_id, child_id)
        tube_legend : ["X0 = ...", ...]
    Uses post-order DFS; x positions are assigned leaf-first so parents
    centre over their children.  No external libraries required.
    """
    labels:      dict[int, str]            = {}
    kinds:       dict[int, str]            = {}
    edges:       list[tuple[int, int]]     = []
    tube_legend: list[str]                 = []
    x_pos:       dict[int, float]          = {}
    y_pos:       dict[int, float]          = {}
    tube_counter = [0]
    leaf_counter = [0]          # global leaf slot counter for x spacing
    X_STEP       = 2.0          # horizontal gap between leaves
    Y_STEP       = -2.0         # vertical gap between levels

    visited_tubes: set[int] = set()
    visited_ops:   set[int] = set()

    def _walk_tube(tube: TubeNode, depth: int) -> float:
        tid = id(tube)
        if tid in visited_tubes:
            return x_pos[tid]
        visited_tubes.add(tid)

        if tube.label is None:
            xi = f"X{tube_counter[0]}"
            tube.label = xi

            tube_counter[0] += 1
        else:
            xi = tube.label

        
        tube_legend.append(f"{xi} = {tube.formula}")
        labels[tid] = xi
        kinds[tid]  = "tube"
        y_pos[tid]  = depth * Y_STEP

        op = tube.operator
        if op is None:
            # leaf — assign next x slot
            x = leaf_counter[0] * X_STEP
            leaf_counter[0] += 1
            x_pos[tid] = x
            return x

        oid = id(op)
        edges.append((tid, oid))
        cx = _walk_op(op, depth + 1)
        x_pos[tid] = cx
        return cx

    def _walk_op(op: OperatorNode, depth: int) -> float:
        oid = id(op)
        if oid in visited_ops:
            return x_pos[oid]
        visited_ops.add(oid)

        if isinstance(op, AndOperatorNode):
            short = "AND"
        elif isinstance(op, OrOperatorNode):
            short = "OR"
        elif isinstance(op, GloballyOperatorNode):
            short = f"G[{op.a},{op.b}]"
        elif isinstance(op, EventuallyOperatorNode):
            short = f"F[{op.a},{op.b}]"
        elif isinstance(op, UntilOperatorNode):
            short = f"U[{op.a},{op.b}]"
        else:
            short = op.name

        labels[oid] = short
        kinds[oid]  = "op"
        y_pos[oid]  = depth * Y_STEP

        child_xs = []
        if isinstance(op, (AndOperatorNode, OrOperatorNode)):
            for child in (op.left, op.right):
                edges.append((oid, id(child)))
                child_xs.append(_walk_tube(child, depth + 1))
        elif isinstance(op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
            edges.append((oid, id(op.child)))
            child_xs.append(_walk_tube(op.child, depth + 1))

        cx = sum(child_xs) / len(child_xs) if child_xs else 0.0
        x_pos[oid] = cx
        return cx

    _walk_tube(root, 0)

    positions = {nid: (x_pos[nid], y_pos[nid]) for nid in labels}
    return positions, labels, kinds, edges, tube_legend


def plot_ttlt_tree(root: TubeNode, save_path: str = "ttlt_tree.html") -> None:
    """
    Plot the tTLT tree as a layered directed graph and save to an HTML file
    via plotly — no matplotlib, no display server required.

    Visual conventions
    ------------------
    TubeNode     : light-grey circle,  labeled X0, X1, X2, ...
    OperatorNode : light-blue diamond, labeled G[a,b] / F[a,b] / U[a,b] / AND / OR
    Hover tooltip shows the full label for every node.
    Legend table (annotation) lists  Xi = full tube label.

    Parameters
    ----------
    root      : TubeNode   root of the tTLT tree
    save_path : str        output HTML file path  (default: "ttlt_tree.html")
    """
    import plotly.graph_objects as go

    positions, labels, kinds, edges, tube_legend = _ttlt_collect_layout(root)

    # ── edge traces (lines with arrows) ──────────────────────────────────
    edge_x, edge_y = [], []
    for src_id, dst_id in edges:
        x0, y0 = positions[src_id]
        x1, y1 = positions[dst_id]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#888888", width=1.5),
        hoverinfo="none",
        showlegend=False,
    )

    # Arrow heads as a separate scatter (tiny markers at dst)
    arrow_x = [positions[dst][0] for _, dst in edges]
    arrow_y = [positions[dst][1] for _, dst in edges]
    arrow_trace = go.Scatter(
        x=arrow_x, y=arrow_y,
        mode="markers",
        marker=dict(symbol="arrow-bar-up", size=10,
                    color="#888888", angleref="previous"),
        hoverinfo="none",
        showlegend=False,
    )

    # ── tube node trace (circles) ─────────────────────────────────────────
    tube_ids = [nid for nid, k in kinds.items() if k == "tube"]
    tube_trace = go.Scatter(
        x=[positions[nid][0] for nid in tube_ids],
        y=[positions[nid][1] for nid in tube_ids],
        mode="markers+text",
        marker=dict(symbol="circle", size=40,
                    color="#E8E8E8", line=dict(color="#555555", width=1.5)),
        text=[labels[nid] for nid in tube_ids],
        textposition="middle center",
        textfont=dict(size=12, color="#222222"),
        customdata=[labels[nid] for nid in tube_ids],
        hovertemplate="%{customdata}<extra></extra>",
        name="Tube node",
        showlegend=True,
    )

    # ── operator node trace (diamonds) ───────────────────────────────────
    op_ids = [nid for nid, k in kinds.items() if k == "op"]
    op_trace = go.Scatter(
        x=[positions[nid][0] for nid in op_ids],
        y=[positions[nid][1] for nid in op_ids],
        mode="markers+text",
        marker=dict(symbol="diamond", size=50,
                    color="#C8D8F0", line=dict(color="#335599", width=1.5)),
        text=[labels[nid] for nid in op_ids],
        textposition="middle center",
        textfont=dict(size=10, color="#223366"),
        customdata=[labels[nid] for nid in op_ids],
        hovertemplate="%{customdata}<extra></extra>",
        name="Operator node",
        showlegend=True,
    )

    # ── legend annotation (tube node index table) ─────────────────────────
    legend_lines = []
    for entry in tube_legend:
        xi, lbl = entry.split(" = ", 1)
        # wrap long labels at 80 chars for readability
        wrapped = "<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(
            lbl[i:i+80] for i in range(0, len(lbl), 80)
        )
        legend_lines.append(f"<b>{xi}</b>: {wrapped}")
    legend_str = "<br>".join(legend_lines)

    fig = go.Figure(data=[edge_trace, arrow_trace, tube_trace, op_trace])

    # Compute axis ranges so we can place the annotation inside the figure
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max((x_max - x_min) * 0.05, 0.5)
    y_pad = max((y_max - y_min) * 0.05, 0.5)

    fig.update_layout(
        title=dict(text="tTLT graph structure", font=dict(size=16)),
        xaxis=dict(visible=False, range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(visible=False, range=[y_min - y_pad * 6, y_max + y_pad],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=200),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        annotations=[dict(
            x=0.5,
            y=-0.02,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            text=legend_str,
            showarrow=False,
            align="left",
            font=dict(size=10, family="monospace"),
            bgcolor="#F9F9F9",
            bordercolor="#AAAAAA",
            borderwidth=1,
            borderpad=8,
        )],
    )

    fig.show()
    # fig.write_html(save_path)
    # print(f"  [plot_ttlt_tree] saved to {save_path}")


def print_ttlt_tree(root: TubeNode, indent: int = 0) -> None:
    """
    Recursively print the tTLT tree structure from a root TubeNode downward.

    Format
    ------
    [TubeNode] label  shape=(...)  parent=<op name or None>
      |
      [OperatorNode] AND/OR/G/F/U  repr
        |-- left/right/child:
          [TubeNode] ...
    """
    pad    = "  " * indent
    pad_op = "  " * (indent + 1)
    pad_ch = "  " * (indent + 2)

    shape = root.sdf.shape if root.sdf is not None else "None"
    par   = root.parent.name if root.parent else "None"
    print(f"{pad}[TubeNode]  '{root.label}'  shape={shape}  parent={par}")

    op = root.operator
    if op is None:
        print(f"{pad_op}(leaf — no operator child)")
        return

    print(f"{pad_op}|")
    print(f"{pad_op}[{op.name}]  {op!r}")

    if isinstance(op, (AndOperatorNode, OrOperatorNode)):
        print(f"{pad_ch}|-- left:")
        print_ttlt_tree(op.left,  indent + 3)
        print(f"{pad_ch}|-- right:")
        print_ttlt_tree(op.right, indent + 3)
    elif isinstance(op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
        print(f"{pad_ch}|-- child:")
        print_ttlt_tree(op.child, indent + 3)



# --------------------
# Relationship between Nodes
# --------------------

def wire(
    combined_root: TubeNode,
    op:            OperatorNode,
    children:      list[TubeNode],
) -> None:
    """
    Set all bidirectional pointers for one tree layer in a single call.

        combined_root  <-->  op  <-->  children[...]

    Pointers set
    ------------
        combined_root.operator = op             downward: root tube → its operator
        op.tube                = combined_root   upward:   operator → its tube
        child.parent           = op             upward:   each child tube → operator
    """
    combined_root.operator = op            # child of the tube node
    op.tube                = combined_root #parent of the operator node
    for child in children:
        child.parent = op #children of the operator nodes



def Parent(node: TubeNode | OperatorNode) -> OperatorNode | TubeNode | None:
    """
    Return the immediate parent of a node.

      TubeNode     -> its .parent    (an OperatorNode or None if root)
      OperatorNode -> its .tube      (the TubeNode that owns this operator)

    Parameters
    ----------
    node : TubeNode | OperatorNode

    Returns
    -------
    OperatorNode | TubeNode | None
    """
    if isinstance(node, TubeNode):
        return node.parent          # OperatorNode above, or None if root
    elif isinstance(node, OperatorNode):
        return node.tube            # TubeNode that owns this operator
    else:
        raise TypeError(f"Parent: expected TubeNode or OperatorNode, got {type(node).__name__}.")


def Children(node: TubeNode | OperatorNode) -> list[TubeNode | OperatorNode]:
    """
    Return all immediate children of a node.

      TubeNode     -> [node.operator]           if operator exists, else []
      AndOperatorNode / OrOperatorNode  -> [left, right]
      G / F / U OperatorNode            -> [child]

    Parameters
    ----------
    node : TubeNode | OperatorNode

    Returns
    -------
    list[TubeNode | OperatorNode]
        Direct children; empty list if node is a leaf.
    """
    if isinstance(node, TubeNode):
        return [node.operator] if node.operator is not None else []

    elif isinstance(node, (AndOperatorNode, OrOperatorNode)):
        return [node.left, node.right]

    elif isinstance(node, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
        return [node.child]

    elif isinstance(node, OperatorNode):
        # Unknown OperatorNode subclass — no children known
        return []

    else:
        raise TypeError(f"Children: expected TubeNode or OperatorNode, got {type(node).__name__}.")


def Pre(node: TubeNode) -> TubeNode | None:
    """
    Return the grandparent TubeNode of a given TubeNode by skipping the
    intermediate OperatorNode.

      TubeNode  --(parent)-->  OperatorNode  --(tube)-->  TubeNode  <-- returned

    Returns None if the node is the root (no parent operator) or if the
    parent operator has no owning TubeNode above it.

    Parameters
    ----------
    node : TubeNode

    Returns
    -------
    TubeNode | None
    """
    if not isinstance(node, TubeNode):
        raise TypeError(f"Pre: expected TubeNode, got {type(node).__name__}.")

    parent_op = node.parent           # step 1: TubeNode -> OperatorNode
    if parent_op is None:
        return None

    return parent_op.tube             # step 2: OperatorNode -> TubeNode (grandparent)


def Post(node: TubeNode) -> list[TubeNode]:
    """
    Return all grandchild TubeNodes of a given TubeNode by collecting the
    children of every immediate child OperatorNode.

      TubeNode  --(operator)-->  OperatorNode  --(left/right/child)-->  [TubeNodes]

    Concretely:
      1. Get the operator child of `node`  (via Children).
      2. Collect Children of that operator — these are TubeNodes one level down.

    Returns an empty list if `node` is a leaf (no operator child).

    Parameters
    ----------
    node : TubeNode

    Returns
    -------
    list[TubeNode]
        All TubeNode grandchildren; empty if node is a leaf.
    """
    if not isinstance(node, TubeNode):
        raise TypeError(f"Post: expected TubeNode, got {type(node).__name__}.")

    result: list[TubeNode] = []
    for child_op in Children(node):           # child_op is an OperatorNode
        for grandchild in Children(child_op): # grandchild is a TubeNode
            if isinstance(grandchild, TubeNode):
                result.append(grandchild)

    return result