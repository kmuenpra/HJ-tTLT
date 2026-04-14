import sys
import os
repo_root = os.path.abspath(os.getcwd())
sys.path.insert(0, os.path.join(repo_root, "optimized_dp"))

import numpy as np
import scipy.io as sio
import argparse

from odp.Grid     import Grid
from odp.Shapes   import CylinderShape
from odp.solver   import HJSolver
from odp.dynamics import Plane2D

# ── CLI arguments ─────────────────────────────────────────────────────────────

'''
Sample command:

python3 training_HJFNO/Plane2D_FG_data_gen.py --plot

'''

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str,
                    default="/home/kmuenpra/git/tTLT/training_HJFNO/HJB_training_mat/test_Plane2D_FG_50x50.mat",
                    help="Output .mat file")
parser.add_argument("--plot", action="store_true", default=False,
                    help="Plot BRS after solving")
args = parser.parse_args()

out  = args.out
plot = args.plot

# ── 1. Grid ───────────────────────────────────────────────────────────────────
g = Grid(
    minBounds    = np.array([-10.0, -10.0]),
    maxBounds    = np.array([ 10.0,  10.0]),
    dims         = 2,
    pts_each_dim = np.array([80, 80]),
    periodicDims = [],
)

# ── 2. Time horizon ───────────────────────────────────────────────────────────
lookback_length = 10
t_step          = 0.25
tau = np.arange(0, lookback_length + 1e-5, t_step)

# ── 3. Target set (shared) ────────────────────────────────────────────────────
target_sdf = CylinderShape(g, ignore_dims=[], center=[0.0, 0.0], radius=1.0)

nx, ny = g.pts_each_dim
T      = len(tau)

# ── 4. Solve Eventually (F): standard BRS ────────────────────────────────────
#   uMode = "min"  (minimizer reaches target)
#   dMode = "max"
#   TargetSetMode = "minVWithV0"
#   multiple_value = target_sdf  (positive = inside target)

print("\n" + "=" * 60)
print("  Solving Eventually (F): BRS")
print(f"  Horizon: [0, {lookback_length}]  steps: {T}  dt: {t_step}")
print("=" * 60)

plane_F = Plane2D(
    x     = [0.0, 0.0],
    vxMin = -1.0,
    vxMax =  1.0,
    vyMin = -1.0,
    vyMax =  1.0,
    dMin  = -0.1,
    dMax  =  0.1,
    uMode = "min",
    dMode = "max",
)

brs_F = HJSolver(
    dynamics_obj     = plane_F,
    grid             = g,
    multiple_value   = target_sdf,
    tau              = tau,
    compMethod       = {"TargetSetMode": "minVWithV0"},
    saveAllTimeSteps = True,
    accuracy         = "medium",
    verbose          = False,
)

print(f"  target_sdf : min={target_sdf.min():.4f}  max={target_sdf.max():.4f}")
print(f"  brs_F(t=0) : min={brs_F[...,0].min():.4f}  max={brs_F[...,0].max():.4f}")
print(f"  Fraction in BRS (t=0): {(brs_F[...,0] <= 0).mean():.4f}")

# ── 5. Solve Globally (G): invariance set ────────────────────────────────────
#   uMode = "max"  (maximizer maintains invariance)
#   dMode = "min"
#   TargetSetMode = "minVOverTime"
#   multiple_value = -target_sdf  (sign-flipped for invariance formulation)
#   result is sign-flipped back after solving

print("\n" + "=" * 60)
print("  Solving Globally (G): invariance set")
print(f"  Horizon: [0, {lookback_length}]  steps: {T}  dt: {t_step}")
print("=" * 60)

plane_G = Plane2D(
    x     = [0.0, 0.0],
    vxMin = -1.0,
    vxMax =  1.0,
    vyMin = -1.0,
    vyMax =  1.0,
    dMin  = -0.1,
    dMax  =  0.1,
    uMode = "max",
    dMode = "min",
)

brs_G = HJSolver(
    dynamics_obj     = plane_G,
    grid             = g,
    multiple_value   = -target_sdf,
    tau              = tau,
    compMethod       = {"TargetSetMode": "minVOverTime"},
    saveAllTimeSteps = True,
    accuracy         = "medium",
    verbose          = False,
)
brs_G = -1.0 * brs_G   # flip back: l(x) <= 0 ↔ state can be kept in target

print(f"  target_sdf : min={target_sdf.min():.4f}  max={target_sdf.max():.4f}")
print(f"  brs_G(t=0) : min={brs_G[...,0].min():.4f}  max={brs_G[...,0].max():.4f}")
print(f"  Fraction in invariance set (t=0): {(brs_G[...,0] <= 0).mean():.4f}")

# ── 6. Optional plot ──────────────────────────────────────────────────────────
if plot:
    import plotly.graph_objects as go

    x_axis = np.linspace(-10.0, 10.0, nx)
    y_axis = np.linspace(-10.0, 10.0, ny)

    def _make_fig(brs: np.ndarray, label: str, color: str) -> None:
        N_t = brs.shape[2]

        def _fill(z):
            return go.Heatmap(
                x=x_axis, y=y_axis, z=np.where(z <= 0, z, np.nan).T,
                colorscale=[[0, color], [1, color]],
                zmin=-0.1, zmax=0.0,
                opacity=0.3, showscale=False,
                name=f"{label} (fill)", hoverinfo="skip",
            )

        def _boundary(z, c=color, name=label, dash="solid"):
            return go.Contour(
                x=x_axis, y=y_axis, z=z.T,
                contours=dict(start=0, end=0, size=1),
                contours_coloring="none",
                line=dict(color=c, width=2, dash=dash),
                showscale=False, name=name,
            )

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
                label=f"t={tau[k]:.2f}", method="animate",
            )
            for k in range(N_t)
        ]

        fig = go.Figure(
            data=[_fill(brs[:, :, 0]), _boundary(brs[:, :, 0]),
                  _boundary(target_sdf, c="green", name="Target set", dash="dash")],
            frames=frames,
            layout=go.Layout(
                title=label,
                xaxis=dict(title="x", range=[-10.0, 10.0], constrain="domain"),
                yaxis=dict(title="y", range=[-10.0, 10.0], scaleanchor="x", scaleratio=1),
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

    _make_fig(brs_F, label="Eventually (F): BRS",           color="blue")
    _make_fig(brs_G, label="Globally (G): Invariance Set",  color="red")

# ── 7. Save to .mat ───────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(out), exist_ok=True)

sio.savemat(out, {
    "brs_F":      brs_F.astype(np.float32),       # (50, 50, T)  Eventually BRS
    "brs_G":      brs_G.astype(np.float32),       # (50, 50, T)  Globally invariance set
    "target_sdf": target_sdf.astype(np.float32),  # (50, 50)     shared target set
    "tau":        tau,                             # (T,)
    "nx":         nx,
    "ny":         ny,
    "T":          T,
    "x_axis":     np.linspace(-10.0, 10.0, nx),
    "y_axis":     np.linspace(-10.0, 10.0, ny),
})

print(f"\nSaved → {out}")
print(f"  brs_F  : {brs_F.shape}   (Eventually BRS)")
print(f"  brs_G  : {brs_G.shape}   (Globally invariance set)")
print(f"  target : {target_sdf.shape}")
print(f"  tau    : {tau.shape}  [{tau[0]:.2f} … {tau[-1]:.2f}]")
