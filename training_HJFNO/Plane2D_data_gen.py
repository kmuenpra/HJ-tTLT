import sys
import os
repo_root = os.path.abspath(os.getcwd())
sys.path.insert(0, os.path.join(repo_root, "optimized_dp"))

import numpy as np
import scipy.io as sio
import argparse

from odp.Grid     import Grid
from odp.Shapes   import CylinderShape, ShapeRectangle, ShapeEllipsoid
from odp.solver   import HJSolver
from odp.dynamics import Plane2D

# ── CLI argument ──────────────────────────────────────────────────────────────

'''
Sample command:

python3 training_HJFNO/Plane2D_data_gen.py --M 10 --seed 42 --plot

'''

parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=10, help="Number of samples to generate")
parser.add_argument("--seed", type=int, default=0,  help="Random seed")
parser.add_argument("--out", type=str,
                    default="/home/kmuenpra/git/tTLT/training_HJFNO/HJB_training_mat/Plane2D_50x50_100pts_u[neg1_1]_d[neg01_01].mat",
                    help="Output .mat file")
parser.add_argument("--plot", action="store_true", default=False,             help="Plot each BRS after solving")
args = parser.parse_args()

M    = args.M
rng  = np.random.default_rng(args.seed)
out  = args.out
plot = args.plot

# ── 1. Grid ───────────────────────────────────────────────────────────────────
g = Grid(
    minBounds    = np.array([-10.0, -10.0]),
    maxBounds    = np.array([ 10.0,  10.0]),
    dims         = 2,
    pts_each_dim = np.array([50, 50]),
    periodicDims = [],
)

# ── 2. Dynamics ───────────────────────────────────────────────────────────────
plane = Plane2D(
    x     = [0.5, 0.8],
    vxMin = -1.0,
    vxMax =  1.0,
    vyMin = -1.0,
    vyMax =  1.0,
    dMin  = -0.1,
    dMax  =  0.1,
    uMode = "min",
    dMode = "max",
)

# ── 3. Target set (fixed) ─────────────────────────────────────────────────────
target_set = CylinderShape(g, ignore_dims=[], center=[0.0, 0.0], radius=1.0)

# ── 4. Time horizon ───────────────────────────────────────────────────────────
lookback_length = 12
t_step          = 0.25
tau = np.arange(0, lookback_length + 1e-5, t_step)

# ── 5. Random constraint set ──────────────────────────────────────────────────
def random_constraint_set(grid: Grid, rng: np.random.Generator) -> np.ndarray:
    shape_type = rng.integers(0, 3)
    offset     = rng.uniform(-2.0, 2.0, size=grid.dims)

    if shape_type == 0:
        radius = rng.uniform(3.0, 8.0)
        print(f"  [constraint] CylinderShape  | center={np.round(offset,2).tolist()} | radius={radius:.2f}")
        return CylinderShape(grid, ignore_dims=[], center=offset.tolist(), radius=radius)

    elif shape_type == 1:
        half_w = rng.uniform(3.0, 8.0, size=grid.dims)
        print(f"  [constraint] ShapeRectangle | center={np.round(offset,2).tolist()} | half_widths={np.round(half_w,2).tolist()}")
        return ShapeRectangle(grid, target_min=offset - half_w, target_max=offset + half_w)

    else:
        semi_axes = rng.uniform(3.0, 8.0, size=grid.dims)
        print(f"  [constraint] ShapeEllipsoid | center={np.round(offset,2).tolist()} | semi_axes={np.round(semi_axes,2).tolist()}")
        return ShapeEllipsoid(grid, center=offset.tolist(), semiAxLen=semi_axes.tolist())



# ── 6. Data generation loop ───────────────────────────────────────────────────
# Preallocate:
#   constraints : (M, 50, 50)       — one SDF per sample
#   results     : (M, 50, 50, T)    — full time history per sample
nx, ny = g.pts_each_dim
T      = len(tau)

constraints = np.zeros((M, nx, ny), dtype=np.float32)
results     = np.zeros((M, nx, ny, T), dtype=np.float32)


# ------ Solve Until --------
compMethods = {
    "TargetSetMode":   "minVWithV0",
    "ObstacleSetMode": "maxVWithObstacle",
}

for i in range(M):
    print(f"\n[{i+1}/{M}] Generating sample...")

    constraint_set = random_constraint_set(g, rng)

    result = HJSolver(
        dynamics_obj     = plane,
        grid             = g,
        multiple_value   = [target_set, -constraint_set],
        tau              = tau,
        compMethod       = compMethods,
        saveAllTimeSteps = True,
        accuracy         = "medium",
        verbose          = False,
    )

    constraints[i] = constraint_set.astype(np.float32)
    results[i]     = result.astype(np.float32)

    print(f"  result shape: {result.shape}")

    if plot:
        import plotly.graph_objects as go

        brs       = result
        brs_name  = "BRS"
        brs_color = "blue"
        x_axis    = np.linspace(-10.0, 10.0, nx)
        y_axis    = np.linspace(-10.0, 10.0, ny)

        if brs.ndim == 2:
            brs = brs[:, :, np.newaxis]
        N_t = brs.shape[2]

        def _fill(z):
            z_m = np.where(z <= 0, z, np.nan)
            return go.Heatmap(
                x=x_axis, y=y_axis, z=z_m.T,
                colorscale=[[0, brs_color], [1, brs_color]],
                zmin=-0.1, zmax=0.0,
                opacity=0.3, showscale=False,
                name=f"{brs_name} (fill)", hoverinfo="skip",
            )

        def _boundary(z, color=brs_color, name=brs_name, dash="solid"):
            return go.Contour(
                x=x_axis, y=y_axis, z=z.T,
                contours=dict(start=0, end=0, size=1),
                contours_coloring="none",
                line=dict(color=color, width=2, dash=dash),
                showscale=False, name=name,
            )

        static_traces = [
            _boundary(target_set,      color="green", name="Target set",     dash="dash"),
            _boundary(-constraint_set, color="red",   name="Constraint set", dash="dash"),
        ]

        frames = [
            go.Frame(
                data=[_fill(brs[:, :, k]), _boundary(brs[:, :, k])],
                traces=[0, 1], name=str(k),
            )
            for k in range(N_t)
        ]

        slider_steps = [
            dict(
                args=[[str(k)], dict(frame=dict(duration=0, redraw=True),
                                     mode="immediate", transition=dict(duration=0))],
                label=f"t={tau[k]:.1f}", method="animate",
            )
            for k in range(N_t)
        ]

        fig = go.Figure(
            data=[_fill(brs[:, :, 0]), _boundary(brs[:, :, 0]), *static_traces],
            frames=frames,
            layout=go.Layout(
                title=f"BRS Sample {i+1}/{M}",
                xaxis=dict(title="x", range=[-10.0, 10.0], constrain="domain"),
                yaxis=dict(title="y", range=[-10.0, 10.0], scaleanchor="x", scaleratio=1),
                legend=dict(x=1.02, y=1.0),
                updatemenus=[dict(
                    type="buttons", showactive=False, y=1.08, x=0.5, xanchor="center",
                    buttons=[
                        dict(label="▶ Play",  method="animate",
                             args=[None, dict(frame=dict(duration=100, redraw=True),
                                              fromcurrent=True, transition=dict(duration=0))]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate", transition=dict(duration=0))]),
                    ],
                )],
                sliders=[dict(
                    steps=slider_steps,
                    currentvalue=dict(prefix="Time: ", visible=True, xanchor="center"),
                    pad=dict(t=50),
                )],
            ),
        )
        fig.show()

# ── 7. Save to .mat ───────────────────────────────────────────────────────────
sio.savemat(out, {
    "constraints": constraints,   # (M, 50, 50)
    "results":     results,        # (M, 50, 50, T)
    "target_set":  target_set.astype(np.float32),  # (50, 50)  — shared across all
    "tau":         tau,            # (T,)
    "M":           M,
    "nx":          nx,
    "ny":          ny,
    "T":           T,
    "x_axis":      np.linspace(-10.0, 10.0, nx),
    "y_axis":      np.linspace(-10.0, 10.0, ny),
})

print(f"\nSaved {M} samples → {out}")
print(f"  constraints : {constraints.shape}")
print(f"  results     : {results.shape}")