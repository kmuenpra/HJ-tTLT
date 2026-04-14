# tTLT Session Context

## 1. Primary Request and Intent

1. **LIDAR detection system**: Add `lidar_detect()`, `collect_predicates()`, `find_tube_in_tree()`, and `plot_predicate_offsets()` to `main.py` so that unknown circular obstacles are gradually detected as the robot moves, and the tTLT is updated online with the true predicate center.
2. **Precomputed F/G BRS**: Generate and load precomputed backward reachable sets for `Eventually` and `Globally` operators (origin-centered, Plane2D dynamics) from a `.mat` file — analogous to how `solve_until` uses FNO.
3. **BRS translation**: Translate the precomputed origin-centered BRS by `(+c_x, +c_y)` when the predicate has a non-zero center, using `translate_brs()` (bilinear `RegularGridInterpolator`).
4. **Online update methods**: Implement `update_until`, `update_eventually`, `update_globally` in `stl_hj.py` so the tTLT is updated in-place when LIDAR detects the true center of a predicate.
5. **`stl_node` parameter on OperatorNode subclasses**: All 5 subclasses of `OperatorNode` in `tTLT_nodes.py` updated to accept `stl_node=None` and pass it to `super().__init__()`.
6. **Bug fix — `update_until` print**: `op.psi` → `op.stl_node.psi` / `op.child` (correct attribute paths).
7. **Bug fix — `update_eventually` / `update_globally`**: These methods wrote the translated BRS into `op.child` (the raw predicate SDF leaf) instead of `op.tube` (the BRS output TubeNode read by control synthesis). Fixed to update `op.tube`.

---

## 2. Key Technical Concepts

- **tTLT tree pointer convention**:
  - `op.tube` → TubeNode above the operator (BRS output, read by synthesis algorithms)
  - `op.child` → TubeNode below the operator (predicate SDF input)
  - `op.stl_node` → corresponding STL formula node (carries `a`, `b`, `.phi`, `.psi`, `.c_x`, `.c_y`, `.r`)
- **LIDAR detection**: `||robot_pos - true_center||₂ ≤ lidar_range`; once detected, the predicate's BRS is retranslated and the tTLT is updated in-place via `update_*`.
- **`find_tube_in_tree`**: DFS through the wired tTLT tree (not `_predicate_cache`) matching `.formula` with exact match or restricted `endswith` for phi/copy naming conventions.
- **`collect_predicates`**: Returns only Predicate STL nodes whose tTLT TubeNode is a leaf (`tube.operator is None`).
- **FNO translation for G/F**: Precomputed BRS is origin-centered. `translate_brs(brs_full, g, dx, dy)` queries the origin BRS at `(x - dx, y - dy)` for each output point `(x, y)`, effectively shifting the reachable set center by `(+dx, +dy)`.
- **`solve_until` phi wiring**: When `phi` is a direct predicate, `phi_tube` is wired as the owner of `UntilOperatorNode` (not replaced by a new `brs_tube_i`), so it stays reachable in the tree with correct `.parent` pointers.

---

## 3. Files and Code Sections

### `main.py`
All LIDAR + detection infrastructure lives here.

**`find_tube_in_tree(root, formula_repr)`** — DFS through wired tTLT tree:
```python
def find_tube_in_tree(root: TubeNode, formula_repr: str) -> TubeNode | None:
    wrapped = f"({formula_repr})"
    def _matches(formula: str) -> bool:
        if formula == formula_repr: return True
        if formula.endswith(wrapped) and (
            formula.startswith("phi(") or formula.startswith("copy_")
        ): return True
        return False
    stack = [root]; visited = set()
    while stack:
        node = stack.pop()
        if id(node) in visited: continue
        visited.add(id(node))
        if _matches(node.formula): return node
        op = node.operator
        if op is None: continue
        if isinstance(op, (AndOperatorNode, OrOperatorNode)):
            stack.extend([op.left, op.right])
        elif isinstance(op, (GloballyOperatorNode, EventuallyOperatorNode, UntilOperatorNode)):
            stack.append(op.child)
    return None
```

**`collect_predicates(stl_root, ttlt_root)`** — leaf-only Predicate nodes:
```python
def collect_predicates(stl_root, ttlt_root) -> list[Predicate]:
    found = []
    stack = [stl_root]
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
```

**LIDAR detection loop** (inside main loop):
```python
for i, tc in pred_true_centers.items():
    if i not in pred_detected:
        if np.linalg.norm(robot_pos - np.array(tc)) <= lidar_range:
            pred_detected.add(i)
            predicates[i].c_x = tc[0]
            predicates[i].c_y = tc[1]
            pred      = predicates[i]
            tube      = find_tube_in_tree(solver.root, repr(pred))
            parent_op = tube.parent if tube else None
            parent_tube = Pre(tube)  if tube else None
            if isinstance(parent_op, UntilOperatorNode):
                solver.update_until(op=parent_op, psi_center=tc, dt=dt)
            elif isinstance(parent_op, EventuallyOperatorNode):
                solver.update_eventually(op=parent_op, new_center=tc)
            elif isinstance(parent_op, GloballyOperatorNode):
                solver.update_globally(op=parent_op, new_center=tc)
                if isinstance(parent_tube.parent, EventuallyOperatorNode):
                    solver.update_eventually(op=parent_tube.operator, new_center=tc)
```

### `stl_hj.py`

**`translate_brs(brs_full, g, dx, dy, fill=1.0)`** — BRS spatial translation:
```python
def translate_brs(self, brs_full, g, dx, dy, fill=1.0):
    x_coords = np.linspace(g.min[0], g.max[0], g.pts_each_dim[0])
    y_coords = np.linspace(g.min[1], g.max[1], g.pts_each_dim[1])
    T = brs_full.shape[2]
    brs_translated = np.full_like(brs_full, fill_value=fill)
    for k in range(T):
        interp = RegularGridInterpolator(
            (x_coords, y_coords), brs_full[:, :, k],
            method='linear', bounds_error=False, fill_value=fill)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
        query_points = np.stack([xx - dx, yy - dy], axis=-1)
        brs_translated[:, :, k] = interp(query_points)
    return brs_translated
```

**Precomputed BRS loading in `__init__`**:
```python
_fg_mat = sio.loadmat(
    '/home/kmuenpra/git/tTLT/training_HJFNO/HJB_training_mat/Plane2D_FG_50x50.mat')
self.brs_full_eventually = _fg_mat['brs_F'].astype(np.float64)  # (50, 50, T)
self.brs_full_globally   = _fg_mat['brs_G'].astype(np.float64)  # (50, 50, T)
```

**`update_eventually(op, new_center)`** and **`update_globally(op, new_center)`** — update `op.tube` (not `op.child`!):
```python
def update_eventually(self, op, new_center):
    c_x, c_y = new_center[0], new_center[1]
    brs_full = self.translate_brs(self.brs_full_eventually, g=self.grid, dx=c_x, dy=c_y)
    op.tube.sdf      = brs_full[..., op.tube.t_index]
    op.tube.brs_full = brs_full

def update_globally(self, op, new_center):
    c_x, c_y = new_center[0], new_center[1]
    brs_full = self.translate_brs(self.brs_full_globally, g=self.grid, dx=c_x, dy=c_y)
    op.tube.sdf      = brs_full[..., op.tube.t_index]
    op.tube.brs_full = brs_full
```

**`update_until(op, psi_center, dt)`** — rerun FNO + translate + update `op.tube`:
```python
def update_until(self, op, psi_center, dt):
    phi = op.stl_node.phi;  psi = op.stl_node.psi
    psi.c_x, psi.c_y = psi_center[0], psi_center[1]
    tau = np.arange(0, op.b + 1e-8, dt)
    sdf_phi_translated = self.translated_predicate_sdf_until(phi, psi, self.grid)
    _, FNO_pred = HJR_FNO2d_SuperResQuery(
        g=self.grid, sdf_input=-sdf_phi_translated,
        time_hyparam=(10-op.b)+tau, model=self.model_until)
    brs_full = FNO_pred.detach().cpu().numpy()
    brs_full = np.minimum.accumulate(brs_full[..., ::-1], axis=-1)[..., ::-1]
    brs_full = self.translate_brs(brs_full, g=self.grid, dx=psi_center[0], dy=psi_center[1])
    op.tube.sdf      = brs_full[..., op.tube.t_index]
    op.tube.brs_full = brs_full
    # Also update psi child TubeNode SDF
    translated_node = Predicate(expression=..., c_x=psi_center[0], c_y=psi_center[1], r=op.stl_node.psi.r)
    translated_node.hj_params = {"region": "target"}
    hj_result = predicate_to_hj(translated_node, self.grid)
    op.child.sdf = hj_result["sdf"];  op.child.brs_full = None
```

### `tTLT_nodes.py`
All 5 `OperatorNode` subclasses updated to accept and forward `stl_node`:
```python
class AndOperatorNode(OperatorNode):
    def __init__(self, left, right, stl_node=None):
        super().__init__(stl_node=stl_node)
        self.left = left;  self.right = right

class GloballyOperatorNode(OperatorNode):
    def __init__(self, a, b, child, stl_node=None):
        super().__init__(stl_node=stl_node)
        self.a=a;  self.b=b;  self.child=child

# OrOperatorNode, EventuallyOperatorNode, UntilOperatorNode follow same pattern
```

### `training_HJFNO/Plane2D_FG_data_gen.py`
New script: generates origin-centered BRS for F (Eventually) and G (Globally) operators, saves to `.mat`:
- Grid: 80×80, bounds `[-10,10]²`, lookback=10, t_step=0.25
- Saves: `brs_F` (shape `(80,80,T)`), `brs_G`, `target_sdf`, `tau`

---

## 4. Problem Solving

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| `tube.parent` always None | `_predicate_cache["tube"]` is different instance than what `solve_globally` wires | Created `find_tube_in_tree` to walk actual wired tree |
| phi predicate tube (phi-of-Until) not found | formula renamed to `phi(U[a,b])(repr(pred))` | Added `endswith` check restricted to `startswith("phi(")` or `startswith("copy_")` |
| G-operator tube matched instead of predicate leaf | `"G[...](repr(pred))"` also matched `endswith(f"({repr(pred)})")` | Restrict endswith check to phi/copy prefixes only |
| phi_tube detached from tree in solve_until | `_replace_leaf(phi_tube, ...)` no-op when phi_tube.parent is None | When `leaf is phi_tube`, wire phi_tube directly as the UntilOperatorNode owner |
| `update_eventually`/`update_globally` don't translate BRS | Wrote translated BRS to `op.child` (predicate SDF leaf) instead of `op.tube` (BRS output) | Changed all writes to `op.tube.sdf` and `op.tube.brs_full` |
| `update_until` print crashed with AttributeError | `op.psi` doesn't exist — should be `op.stl_node.psi` and `op.child` | Fixed attribute paths in print statement |
| `TypeError: __init__() got unexpected keyword 'stl_node'` | OperatorNode subclasses didn't accept `stl_node` parameter | Added `stl_node=None` to all 5 subclass `__init__` signatures |

---

## 5. Pending Tasks

None explicitly requested. Last bug fixed: `update_eventually`/`update_globally` writing to `op.child` instead of `op.tube`.

---

## 6. Current Work

Fixed the root bug causing `update_eventually` and `update_globally` to not translate the BRS:

```python
# WRONG (before fix) — writes to predicate SDF leaf, not BRS output:
op.child.sdf      = brs_full[..., op.child.t_index]
op.child.brs_full = brs_full

# CORRECT (after fix) — writes to BRS output tube read by synthesis algorithms:
op.tube.sdf      = brs_full[..., op.tube.t_index]
op.tube.brs_full = brs_full
```

Both `update_eventually` and `update_globally` now correctly update `op.tube` (the BRS output), consistent with how `update_until` already used `op.tube`.

---

## 7. Optional Next Step

No pending tasks. System is ready to test LIDAR-triggered BRS updates end-to-end with the corrected `update_eventually`/`update_globally` writing to `op.tube`.
