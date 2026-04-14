# tTLT Project — Full Context Summary

**Author:** Kasidit Muenprasitivej  
**Reference:** Yu et al., 2023 — *Online control synthesis for uncertain systems under signal temporal logic specifications*  
[Paper PDF](https://journals.sagepub.com/doi/epdf/10.1177/02783649231212572)

---

## 1. Project Goal

Implement an **online control synthesis pipeline** for robots/agents satisfying **STL (Signal Temporal Logic)** specifications. The approach:

1. Parse an STL formula from YAML.
2. Convert it into a **Tube-based Temporal Logic Tree (tTLT)** — a binary tree where each node stores a **HJ reachable set (BRS/BRT)** as a Signed Distance Function (SDF).
3. At runtime, use the tTLT online: track which reachable set the state is inside, and extract a safe admissible control set.

**Sign convention throughout:** `l(x) ≤ 0` = safe / inside reachable set. All SDFs follow this convention.

---

## 2. File Map

```
tTLT/
├── main.py                          # Entry point — full online synthesis loop + LIDAR detection
├── stl_robustness.py                # STL node classes + robustness evaluator + YAML parser
├── tTLT_nodes.py                    # TubeNode / OperatorNode tree data structures
├── stl_hj.py                        # HJtTLT — solves HJ reachable sets from STL formula
├── tTLT_synthesis.py                # Online synthesis algorithms (Alg 3, 6–11)
├── HJR_FNO/
│   ├── HJR_FNO2d.py                 # FNO2d neural operator architecture
│   ├── HJR_FNO.py                   # 1D FNO (older)
│   └── neural_utils.py
├── training_HJFNO/
│   ├── HJRNO_training_Plane2D.ipynb # Training notebook for FNO model (Until)
│   ├── Plane2D_data_gen.py          # Dataset generation: Until BRS (with constraints)
│   ├── Plane2D_FG_data_gen.py       # Dataset generation: F/G BRS (Eventually + Globally)
│   ├── finite_time_HJB.py
│   ├── HJB_training_mat/
│   │   └── Plane2D_FG_50x50.mat    # Precomputed origin-centered F/G BRS (80×80, T steps)
│   └── model/
│       ├── 02_finetune_t8_best.pt   # CURRENT active model (fine-tuned checkpoint)
│       └── 02_hjrno_Plane2D_Until_u[neg1_1]_d[neg01_01].pt
├── optimized_dp/                    # HJ PDE solver library (odp)
│   └── odp/
│       ├── Grid/
│       ├── solver.py                # HJSolver, computeSpatDerivArray
│       └── dynamics/
│           ├── Plane2D.py           # 2D single integrator
│           └── DubinsCar2.py        # 3D Dubins car
└── configs/
    ├── stl_spec_hj.yaml             # Plane2D STL spec
    ├── hj_config.yaml               # Plane2D grid + dynamics config
    ├── stl_spec_dubins.yaml         # DubinsCar STL spec
    └── hj_config_dubins.yaml
```

---

## 3. Module Descriptions

### `stl_robustness.py`
- **STL node classes**: `Predicate`, `Negation`, `Conjunction`, `Disjunction`, `Globally`, `Eventually`, `Until`, `Release`
- **Robustness semantics**: standard min/max over time windows
- **`parse_stl_yaml(path, pnf=False)`**: loads YAML → STLNode tree; with `pnf=True` rewrites to Positive Normal Form (pushes negations to leaves using De Morgan / temporal duality)
- **`STLRobustnessEvaluator`**: evaluates `rho(Φ, x, t)` over a trajectory
- **`STLHorizonEvaluator`**: returns `||Φ||` (minimum signal duration needed)

### `tTLT_nodes.py`
- **`TubeNode`**: wraps a BRS/SDF array. Key fields:
  - `sdf`: current time-slice SDF (shape `(Nx, Ny)` or `(Nx, Ny, Nz)`)
  - `brs_full`: full time-indexed BRS (shape `(Nx, Ny, T)`)
  - `formula`: string formula this tube represents
  - `label`: short label e.g. `"X0"`, assigned lazily during plotting
  - `ta`: activation time (set by Initialization / online loop)
  - `th`: time horizon / deadline (set by Initialization)
  - `t_index`: current time index into `brs_full`
  - `controlSet`: `None` | `"feasible"` | `"optCtrl"` (set by buildControlTree)
  - `parent`, `operator`: bidirectional tree pointers
- **`OperatorNode` subclasses**: `AndOperatorNode`, `OrOperatorNode`, `GloballyOperatorNode`, `EventuallyOperatorNode`, `UntilOperatorNode`
  - All subclasses accept `stl_node=None` (pointer to the corresponding STL formula node)
  - `op.stl_node` gives access to `.a`, `.b`, `.phi`, `.psi`, `.c_x`, `.c_y`, `.r` for online updates
- **`wire(combined_root, op, children)`**: sets all 6 bidirectional pointers at once
- **`Parent`, `Children`, `Pre`, `Post`**: tree navigation helpers
- **Critical pointer distinction**:
  - `op.tube` → TubeNode **above** the operator (BRS output — what synthesis algorithms read)
  - `op.child` → TubeNode **below** the operator (predicate SDF input)
  - Always update `op.tube` when refreshing a BRS after online detection

### `stl_hj.py`

#### `tTLT(stl_yaml_path, hj_config_path)` (free function)
Loads STL + HJ config, returns a dict with keys: `formula`, `grid`, `dynamics`, `target_sets`, `obstacle_sets`, `labels`.

#### `HJtTLT` class
Wraps a `tTLT` result and provides solver methods:

| Method | What it does |
|--------|-------------|
| `solve_all(dt, accuracy, plot, matplot)` | Recursively solves the full STL tree |
| `solve_negation(node, ...)` | Negates SDF: `l(x) → -l(x)` |
| `solve_and(node, ...)` | AND = pointwise `max(l1, l2)` |
| `solve_or(node, ...)` | OR = pointwise `min(l1, l2)` |
| `solve_globally(node, ...)` | G[a,b] → precomputed BRS translated by `(+c_x,+c_y)` if FNO_enable, else HJSolver |
| `solve_eventually(node, ...)` | F[a,b] → precomputed BRS translated by `(+c_x,+c_y)` if FNO_enable, else HJSolver |
| `solve_until(node, ...)` | U[a,b] → reach-avoid BRS via **FNO** if `FNO_enable=True` |
| `translate_brs(brs_full, g, dx, dy)` | Shift origin-centered BRS by `(+dx,+dy)` via bilinear interp |
| `update_until(op, psi_center, dt)` | Rerun FNO + translate + update `op.tube` when psi detected |
| `update_eventually(op, new_center)` | Retranslate `brs_full_eventually` + update `op.tube` |
| `update_globally(op, new_center)` | Retranslate `brs_full_globally` + update `op.tube` |

**Model + precomputed BRS loading in `__init__`:**
```python
self.FNO_enable = True
# Until operator — fine-tuned checkpoint:
ft_save_path = '/home/kmuenpra/git/tTLT/training_HJFNO/model/02_finetune_t8_best.pt'
self.model_until = FNO2d(modes1=16, modes2=16, width=64).to('cuda')
ft_ckpt = torch.load(ft_save_path, map_location='cuda', weights_only=False)
self.model_until.load_state_dict(ft_ckpt['model_state_dict'])
self.model_until.eval()

# Eventually + Globally — precomputed HJ BRS:
_fg_mat = sio.loadmat('.../Plane2D_FG_50x50.mat')
self.brs_full_eventually = _fg_mat['brs_F'].astype(np.float64)  # (Nx, Ny, T)
self.brs_full_globally   = _fg_mat['brs_G'].astype(np.float64)  # (Nx, Ny, T)
```

**`solve_until` FNO path:**
1. Compute origin-translated phi SDF via `translated_predicate_sdf_until(phi, psi, grid)`
2. Call `HJR_FNO2d_SuperResQuery(g, -sdf_phi_translated, (10-horizon)+tau, model)`
3. Apply suffix-min over time: `np.minimum.accumulate(brs[..., ::-1], axis=-1)[..., ::-1]`
4. Translate BRS to psi's center: `translate_brs(brs_full, dx=psi.c_x, dy=psi.c_y)`
5. Wire phi_tube directly as UntilOperatorNode owner (when phi is a direct predicate)

**`solve_globally` / `solve_eventually` FNO path:**
- If `FNO_enable` and predicate has `c_x`/`c_y`: `translate_brs(brs_full_globally/eventually, dx=c_x, dy=c_y)`
- For `F(G(pred))` structure: `_pred = node.child.child`

**CRITICAL — `op.tube` vs `op.child` for online updates:**
Always update `op.tube` (the BRS output above the operator), NOT `op.child` (the predicate SDF input below):
```python
# CORRECT — op.tube is what synthesis algorithms (trackingSetNode etc.) read:
op.tube.sdf      = brs_full[..., op.tube.t_index]
op.tube.brs_full = brs_full

# WRONG — op.child is the raw predicate SDF, not used by synthesis:
# op.child.sdf = brs_full[..., op.child.t_index]  # <-- bug pattern
```

**`_predicate_cache`**: dict `{repr(node): {"node", "sdf", "region", "tube"}}` — populated during `_resolve_node_to_sdf`. WARNING: `_predicate_cache["tube"]` may be stale; use `find_tube_in_tree()` for correct `.parent` pointers.

**Plotter:** supports both Plotly (`plot=True`) and Matplotlib/TkAgg (`matplot=True`)

### `tTLT_synthesis.py` — Online Synthesis Algorithms

| Function | Algorithm # | Status |
|----------|------------|--------|
| `get_MTS(root)` | — | Done |
| `get_all_tubes(root)` | — | Done |
| `Compression(segments, grid)` | Alg 3 | Done |
| `tTLTSatisfaction(compressed_root)` | Alg 2 | **Not implemented** |
| `Initialization(root, tk_arr, t0)` | Alg 6 | Done |
| `trackingSetNode(PostSet, state, grid, t, dt)` | Alg 7 | Done |
| `updatetTLT(B_xt, root, tk_arr, t, dt)` | Alg 8 | Done (partial — missing else branch from paper) |
| `buildControlTree(B_xt, root, state, g, dyn, dt)` | Alg 9 | Done |
| `Backtracking(root, compressed_root, state, g, dyn)` | Alg 10 | Done |
| `postSet(B_xt, t, root)` | Alg 11 | Done |

**Key details:**

**`get_MTS(root)`**: Splits tTLT tree into Maximal Temporal Segments — chains of TubeNodes connected only by temporal ops (G/F/U), broken at Boolean ops (AND/OR).

**`Compression(segments, grid)`**:
- For each MTS segment, takes pointwise `min` (union) of all SDFs → one compressed `TubeNode`
- Rewires Boolean operators using `seg[-1].operator` (not `seg[0]` — past bug)
- Compresses `controlSet` labels: `"feasible"` | `"optCtrl(X1|X3|...)"` 

**`Initialization(root, tk_arr, t0)`**:
- `_assign_time_horizon`: `th = t0 + b_hat` where `b_hat = max(tk_arr within [a,b])`
- `_check_reachability_of_boolean_segment`: collects `PostSet` through Boolean ops only
- Returns `PostSet` (all tubes reachable from root through Boolean ops)

**`trackingSetNode(PostSet, state, grid, t, dt)`**:
- For each tube in PostSet: interpolate SDF at state, check if `value ≤ 0` and `t ≤ tube.th`
- Uses `tube.brs_full[..., t_index]` when available (temporal tubes), else static `tube.sdf`
- Filters so each complete path through tree has at most one active tube (removes ancestors when descendant is active)

**`buildControlTree(B_xt, root, state, g, dyn, dt)`**:
- Leaf node in B_xt → `controlSet = "feasible"`
- Non-leaf node in B_xt → `controlSet = "optCtrl"` (TODO: should include tube labels for Backtracking)
- Not in B_xt → `controlSet = None`

**`Backtracking(root, compressed_root, state, g, dyn)`** (Alg 10):
- `_parse_controlSet`: returns `(admissCtrlSet, th)`
  - `None` → `(empty (0,dims), inf)`
  - `"feasible"` → `(zeros (1,dims), inf)` — TODO: should return actual feasible set
  - `"optCtrl(X1|X3)"` → find tube with earliest `th`, compute `optCtrl` via spatial gradient
- `_backtrack_boolean`: **Method 3** — for AND, picks child with earlier deadline
- Returns `admissCtrlSet` shape `(n_controls, dims)`

**`postSet(B_xt, t, root)`** (Alg 11):
- For temporal ops (F, U): adds child tubes if `t ≥ S_i.ta + op.a`
- For G: adds child tubes if `t ≥ S_i.ta + op.b`

### `HJR_FNO/HJR_FNO2d.py`

**Architecture:** `FNO2d` (4-layer 2D Fourier Neural Operator)
- Input: `(batch, X, Y, 4)` — [SDF(x,y), x, y, t]
- Output: `(batch, X, Y, 1)` — predicted value function
- `SpectralConv2d`: rfft2 → two weight tensors (top-left + bottom-left frequency corners) → irfft2
- Lifting: `Conv2d(4 → width, kernel=1)`, Projection: `Conv2d(width → 1, kernel=1)`
- 4 Fourier blocks: spectral path K(u) + bypass W(u) + ReLU

**`HJR_FNO2d_SuperResQuery(g, sdf_input, time_hyparam, model)`**:
- Builds `(T, Nx, Ny, 4)` query tensor from SDF + coordinate channels + time
- Returns `(xx_query, pred_reshaped)` where `pred_reshaped` shape is `(Nx, Ny, T)`

**Training details:**
- Dataset: Plane2D dynamics, Until operator, u ∈ [-1,1], d ∈ [-0.1,0.1]
- Grid: 50×50, bounds [-10,10] but coordinate channels use **hardcoded [-1,1]** (known mismatch)
- Loss: L2 warmup → smooth L∞ refinement (log-sum-exp)
- Model saves: `torch.save(model, ...)` — full model, no class re-import needed at load time

### `main.py` — Online Loop

**LIDAR detection helpers:**
- `find_tube_in_tree(root, formula_repr)` — DFS wired tTLT tree; matches exact formula or `phi(U[...])(...)` / `copy_N(...)` prefixes; never matches operator-level formulas like `G[...](...)`.
- `collect_predicates(stl_root, ttlt_root)` — returns only Predicate STL nodes whose tTLT TubeNode is a leaf (`tube.operator is None`); excludes phi-of-Until which owns a UntilOperatorNode.
- `plot_predicate_offsets(ax, predicates, r_2, N, true_centers, detected)` — undetected: N dashed low-opacity offset circles at radius `r_2`; detected: single solid magenta circle at true center.
- `lidar_detect(state, obstacles, lidar_range)` — returns obstacles whose center is within `lidar_range` of robot.

```
Step 1: Build tTLT + solve all reachable sets (HJtTLT.solve_all)
Step 2: MTS decomposition (get_MTS)
Step 3: collect_predicates → assign random true center per predicate
Step 4: Initialize plot + GIF buffer
Step 5: Initialization (Alg 6) → initial PostSet

Loop over tk_array (t = 0 to stl_horizon, step dt):
    Alg 7: trackingSetNode   → B_xt
    Alg 8: updatetTLT        → updates tube.sdf to current t_index
    Alg 9: buildControlTree  → assigns controlSet to each tube
    Alg 3: Compression       → compressed_root
    Alg 10: Backtracking     → admissCtrlSet
    Apply control u_opt = admissCtrlSet[0]
    Integrate: x = x + f(x,u)*dt
    Alg 11: postSet          → next PostSet

    LIDAR loop: for each undetected predicate i:
        if ||robot_pos - true_center[i]||₂ ≤ lidar_range:
            mark detected; update predicates[i].c_x/c_y
            find tube = find_tube_in_tree(solver.root, repr(pred))
            parent_op = tube.parent
            if UntilOperatorNode: solver.update_until(op, psi_center, dt)
            if EventuallyOperatorNode: solver.update_eventually(op, new_center)
            if GloballyOperatorNode:   solver.update_globally(op, new_center)
                if parent_tube.parent is EventuallyOperatorNode:
                    solver.update_eventually(parent_tube.operator, new_center)

    Update matplotlib plot + GIF frame (LIDAR circle, predicate offsets, trajectory)
```

**Key parameters**: `lidar_range=2.0`, `r_2=0.7`, `N_offset=6`, `x=[5,-1]`, `dt=0.5`

---

## 4. Current Config (Active Test Case)

**STL Formula:** `F[5,10](G[0,10] μ₁) AND (μ₂ U[0,8] μ₃)` — Plane2D  
**Config files:** `configs/stl_spec_hj.yaml`, `configs/hj_config.yaml`  
**Initial state:** `x = [0.5, 0.8]`, `t = 0.0`, `dt = 0.5`  
**Dynamics:** `Plane2D` (2D single integrator), `u ∈ [-1,1]²`, `d ∈ [-0.1,0.1]²`  
**Grid:** 2D, bounds `[-10,10]²`, 50×50 pts

---

## 5. Known Issues / Open TODOs

| Issue | Location | Priority |
|-------|----------|----------|
| `buildControlTree` assigns `"optCtrl"` without tube labels → `_parse_controlSet` can't parse `"optCtrl(X1\|X3)"` format | `tTLT_synthesis.py:700` | High |
| AND in `_backtrack_boolean` uses earliest-deadline heuristic (Method 3), not true intersection | `tTLT_synthesis.py:963` | Medium |
| `"feasible"` controlSet returns zero control — not a true feasible set | `tTLT_synthesis.py:784` | Medium |
| `tTLTSatisfaction` (Alg 2) not implemented | `tTLT_synthesis.py:300` | Medium |
| FNO trained with `[-1,1]` coordinate channels but grid is `[-10,10]` | `HJR_FNO2d.py`, training notebook | Low (known mismatch, works empirically) |
| `updatetTLT` missing else-branch from paper (update time) | `tTLT_synthesis.py:656` | Low |
| `Backtracking` does not distinguish AND intersection vs union | `tTLT_synthesis.py:952` | Medium |

---

## 6. Key Design Decisions

- **Identity vs label comparison**: Python object identity (`tube in B_xt`) breaks across deepcopy boundaries. Always use `tube.label in {t.label for t in B_xt}` (string set lookup).
- **`deepcopy` removed from `get_MTS`**: it only reads the tree — copying caused identity failures in buildControlTree/Backtracking. `Compression` makes its own deepcopy internally.
- **Horizon convention**: reachable set computed over `[0, b]` (not `[a, b]`). This is intentional; `NOTE` comments flag this throughout the code.
- **Coordinate normalization**: FNO query uses `g.min/g.max`; training used hardcoded `[-1,1]` linspace. Future retraining should use `g.grid_points[i]`.
- **`brs_full` vs `sdf`**: `brs_full` shape `(Nx, Ny, T)` stores all time slices. `tube.sdf` is updated in `updatetTLT` to the current time slice `brs_full[..., t_index]`.
- **`odp.Grid` attributes**: use `g.pts_each_dim` (not `g.N`), `g.min`, `g.max`, `g.grid_points[i]` (not `g.vs[i]` for meshgrid).

---

## 7. HJ Reachability Conventions

| Operator | `uMode` | `dMode` | `compMethod` | Meaning |
|----------|---------|---------|--------------|---------|
| G[a,b] | `"max"` | `"min"` | `minVOverTime` | Invariance BRT |
| F[a,b] | from config | from config | `minVWithV0` | BRS |
| U[a,b] (phi) | from config | from config | `minVWithV0` + `maxVWithObstacle` | Reach-avoid BRS |

SDF sign: **predicates stored as `l(x) = -f(x)`** so `l(x) ≤ 0` = safe.

---

## 8. FNO / Precomputed BRS Info

### Until FNO model

| Field | Value |
|-------|-------|
| Active model path | `/home/kmuenpra/git/tTLT/training_HJFNO/model/02_finetune_t8_best.pt` |
| Operator | Until (Plane2D) |
| Input channels | 4: SDF(x,y), x-coord, y-coord, time |
| Modes | 16, Width 64 |
| Grid resolution (training) | 50×50 |
| Dynamics | Plane2D, u∈[-1,1], d∈[-0.1,0.1] |
| Coordinate channels | Hardcoded `[-1,1]` linspace (mismatch with physical `[-10,10]` grid) |
| Load pattern | Checkpoint dict — instantiate `FNO2d(modes1=16, modes2=16, width=64)` then `load_state_dict(ckpt['model_state_dict'])` |

### Precomputed F/G BRS

| Field | Value |
|-------|-------|
| File | `/home/kmuenpra/git/tTLT/training_HJFNO/HJB_training_mat/Plane2D_FG_50x50.mat` |
| Keys | `brs_F` (Eventually BRS), `brs_G` (Globally BRS), `target_sdf`, `tau`, `x_axis`, `y_axis` |
| Shape | `(80, 80, T)` — origin-centered, `[-10,10]²`, lookback=10, dt=0.25 |
| Script | `training_HJFNO/Plane2D_FG_data_gen.py` |
| Usage | Loaded as `brs_full_eventually` / `brs_full_globally` in `HJtTLT.__init__`, then translated by `(+c_x,+c_y)` at solve / update time |