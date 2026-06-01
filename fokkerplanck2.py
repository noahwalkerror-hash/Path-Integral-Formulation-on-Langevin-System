import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Fokker--Planck visualization via Langevin ensemble sampling
#
# System:
#   L(x,v,S) = 1/2 m v^2 - U(x,S)
#
# For visualization, ignore explicit S-dynamics and use
#   U(x,S) = 1/2 k x^2 + U_S(S)
#
# Then:
#   dx = p/m dt
#   dp = (-k x - gamma p/m) dt + sqrt(2D) dW
#
# Corresponding Fokker--Planck equation:
#   ∂_t P =
#      - ∂_x [(p/m) P]
#      - ∂_p [(-kx - gamma p/m) P]
#      + D ∂_p^2 P
# ============================================================


# -----------------------------
# Physical parameters
# -----------------------------
m = 1.0
k = 1.0

# Smaller gamma and D make relaxation slower,
# so the early probability flow is easier to see.
gamma = 0.25
D = 0.15


# -----------------------------
# Numerical parameters
# -----------------------------
dt = 0.002
n_steps = 8000
n_particles = 60000

rng = np.random.default_rng(seed=4)


# -----------------------------
# Initial ensemble
# -----------------------------
# A localized probability blob displaced from equilibrium.
x0_mean = -2.5
p0_mean = 0.0

x0_std = 0.18
p0_std = 0.18

x = rng.normal(loc=x0_mean, scale=x0_std, size=n_particles)
p = rng.normal(loc=p0_mean, scale=p0_std, size=n_particles)


# -----------------------------
# Histogram / plotting domain
# -----------------------------
x_min, x_max = -4.0, 4.0
p_min, p_max = -4.0, 4.0
bins = 120

extent = [x_min, x_max, p_min, p_max]


# -----------------------------
# Snapshot schedule
# -----------------------------
# Dense snapshots at early times, sparse later.
snapshot_steps = []

for step in range(n_steps + 1):
    if step <= 1000:
        if step % 10 == 0:
            snapshot_steps.append(step)
    elif step <= 3000:
        if step % 40 == 0:
            snapshot_steps.append(step)
    else:
        if step % 100 == 0:
            snapshot_steps.append(step)

snapshot_steps = set(snapshot_steps)

snapshots = []
snapshot_times = []


# -----------------------------
# Helper: store density snapshot
# -----------------------------
def store_snapshot(x_values, p_values, step):
    H, x_edges, p_edges = np.histogram2d(
        x_values,
        p_values,
        bins=bins,
        range=[[x_min, x_max], [p_min, p_max]],
        density=True,
    )

    # histogram2d returns shape [x_bin, p_bin].
    # imshow expects [vertical, horizontal], so transpose.
    snapshots.append(H.T)
    snapshot_times.append(step * dt)


# -----------------------------
# Time evolution: Euler--Maruyama
# -----------------------------
for step in range(n_steps + 1):
    if step in snapshot_steps:
        store_snapshot(x, p, step)

    # Deterministic drift
    dx = (p / m) * dt
    dp_drift = (-k * x - gamma * p / m) * dt

    # Gaussian white-noise increment
    dW = np.sqrt(dt) * rng.normal(size=n_particles)
    dp_noise = np.sqrt(2 * D) * dW

    # Update ensemble
    x = x + dx
    p = p + dp_drift + dp_noise


# -----------------------------
# Plot setup
# -----------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.2))

im = ax.imshow(
    snapshots[0],
    origin="lower",
    extent=extent,
    aspect="auto",
    interpolation="nearest",
)

ax.set_xlabel("position x")
ax.set_ylabel("momentum p")

title = ax.set_title(
    f"Fokker--Planck density P(x,p,t), t = {snapshot_times[0]:.3f}"
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("probability density P(x,p,t)")

# Mark equilibrium center
ax.plot([0], [0], marker="x", markersize=8)
ax.text(0.1, 0.1, "equilibrium center", fontsize=9)


# -----------------------------
# Animation update
# -----------------------------
def update(frame):
    density = snapshots[frame]

    im.set_data(density)

    # Keep contrast adaptive but avoid excessive flicker
    vmax = max(np.max(density), 1e-12)
    im.set_clim(0.0, vmax)

    title.set_text(
        f"Fokker--Planck density P(x,p,t), t = {snapshot_times[frame]:.3f}"
    )

    return im, title


ani = FuncAnimation(
    fig,
    update,
    frames=len(snapshots),
    interval=70,
    blit=False,
)


plt.tight_layout()
plt.show()
