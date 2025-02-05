import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 150  # Grid resolution
dt = 1  # Time step
diff = 0.0001  # Diffusion coefficient
visc = 0.0001  # Viscosity coefficient
iterations = 10  # Pressure solve iterations

# Initialize fields
ux = np.zeros((N, N))  # X-direction velocity
uy = np.zeros((N, N))  # Y-direction velocity
dens = np.zeros((N, N))  # Density field

# Mouse interaction parameters
mouse_down = False
last_mx, last_my = -1, -1

def add_density(x, y, amount=100):
    """Add density to a region."""
    dens[x-2:x+2, y-2:y+2] += amount

def add_velocity(x, y, dx, dy):
    """Add velocity to a region."""
    global ux, uy
    ux[x-2:x+2, y-2:y+2] += dx
    uy[x-2:x+2, y-2:y+2] += dy

def diffuse(b, x, x0, diff):
    """Diffusion process."""
    a = dt * diff * (N-2)**2
    for _ in range(iterations):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a*(x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2]))/(1+4*a)

def project(ux, uy, p, div):
    """Pressure projection (ensures incompressibility)."""
    div[1:-1, 1:-1] = -0.5*(ux[2:, 1:-1] - ux[:-2, 1:-1] + uy[1:-1, 2:] - uy[1:-1, :-2])/N
    p[:] = 0
    for _ in range(iterations):
        p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2])/4
    ux[1:-1, 1:-1] -= 0.5*(p[2:, 1:-1] - p[:-2, 1:-1])*N
    uy[1:-1, 1:-1] -= 0.5*(p[1:-1, 2:] - p[1:-1, :-2])*N

def advect(b, d, d0, ux, uy):
    """Advection process."""
    dt0 = dt * (N-2)
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = i - dt0 * ux[i,j]
            y = j - dt0 * uy[i,j]
            x = max(0.5, min(N-2.5, x))
            y = max(0.5, min(N-2.5, y))
            i0, j0 = int(x), int(y)
            s1, s0 = x - i0, 1 - (x - i0)
            t1, t0 = y - j0, 1 - (y - j0)
            d[i,j] = s0 * (t0 * d0[i0,j0] + t1 * d0[i0,j0+1]) + s1 * (t0 * d0[i0+1,j0] + t1 * d0[i0+1,j0+1])

def step():
    """Execute a complete simulation step."""
    global ux, uy, dens
    ux_prev, uy_prev = ux.copy(), uy.copy()
    diffuse(2, ux, ux_prev, visc)
    diffuse(3, uy, uy_prev, visc)
    project(ux, uy, np.zeros_like(ux), np.zeros_like(ux))
    ux_prev, uy_prev = ux.copy(), uy.copy()
    advect(2, ux, ux_prev, ux_prev, uy_prev)
    advect(3, uy, uy_prev, ux_prev, uy_prev)
    project(ux, uy, np.zeros_like(ux), np.zeros_like(ux))
    dens_prev = dens.copy()
    diffuse(0, dens, dens_prev, diff)
    advect(0, dens, dens_prev, ux, uy)

# Mouse interaction
def on_mouse_move(event):
    """Add density and velocity based on mouse movement."""
    global mouse_down, last_mx, last_my
    if mouse_down and event.xdata is not None and event.ydata is not None:
        mx, my = int(event.xdata), int(event.ydata)
        if 0 < mx < N and 0 < my < N:
            dx, dy = mx - last_mx, my - last_my
            add_density(mx, my, 100)
            add_velocity(mx, my, dx * 5, dy * 5)
            last_mx, last_my = mx, my

def on_mouse_down(event):
    """Record initial mouse position."""
    global mouse_down, last_mx, last_my
    mouse_down = True
    if event.xdata is not None and event.ydata is not None:
        last_mx, last_my = int(event.xdata), int(event.ydata)

def on_mouse_up(event):
    """Stop recording mouse movements."""
    global mouse_down
    mouse_down = False

# Visualization
fig, ax = plt.subplots()
img = ax.imshow(dens, cmap='plasma', vmin=0, vmax=100, origin='lower')
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_mouse_down)
fig.canvas.mpl_connect('button_release_event', on_mouse_up)
ax.set_xlim(0, N)
ax.set_ylim(0, N)
ax.set_xticks(np.linspace(0, N, 10))
ax.set_yticks(np.linspace(0, N, 10))

def update(frame):
    """Update the density field and redraw the visualization."""
    step()
    dens[1:-1, 1:-1] *= 0.995
    img.set_data(dens)
    return [img]

ani = FuncAnimation(fig, update, frames=200, interval=10, blit=True)
plt.show()