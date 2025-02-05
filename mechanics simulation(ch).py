import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 仿真参数
N = 150  # 网格分辨率
dt = 1  # 时间步长
diff = 0.0001  # 扩散系数
visc = 0.0001  # 粘性系数
iterations = 10  # 压力求解迭代次数

# 初始化场
ux = np.zeros((N, N))  # x方向速度
uy = np.zeros((N, N))  # y方向速度
dens = np.zeros((N, N))  # 密度场

# 鼠标交互相关参数
mouse_down = False
last_mx, last_my = -1, -1

def add_density(x, y, amount=100):
    """添加密度源"""
    dens[x-2:x+2, y-2:y+2] += amount

def add_velocity(x, y, dx, dy):
    """添加速度源"""
    global ux, uy
    ux[x-2:x+2, y-2:y+2] += dx
    uy[x-2:x+2, y-2:y+2] += dy

def diffuse(b, x, x0, diff):
    """扩散过程"""
    a = dt * diff * (N-2) * (N-2)
    for _ in range(iterations):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a*(x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2]))/(1+4*a)

def project(ux, uy, p, div):
    """压力投影（确保质量守恒）"""
    div[1:-1, 1:-1] = -0.5*(ux[2:, 1:-1] - ux[:-2, 1:-1] + 
                           uy[1:-1, 2:] - uy[1:-1, :-2])/N
    p[:] = 0

    for _ in range(iterations):
        p[1:-1, 1:-1] = (div[1:-1, 1:-1] + 
                         p[2:, 1:-1] + p[:-2, 1:-1] + 
                         p[1:-1, 2:] + p[1:-1, :-2])/4

    ux[1:-1, 1:-1] -= 0.5*(p[2:, 1:-1] - p[:-2, 1:-1])*N
    uy[1:-1, 1:-1] -= 0.5*(p[1:-1, 2:] - p[1:-1, :-2])*N

def advect(b, d, d0, ux, uy):
    """平流过程"""
    dt0 = dt * (N-2)
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = i - dt0 * ux[i,j]
            y = j - dt0 * uy[i,j]
            
            x = max(0.5, min(N-2.5, x))
            y = max(0.5, min(N-2.5, y))
            
            i0 = int(x)
            j0 = int(y)
            
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            
            d[i,j] = s0 * (t0 * d0[i0,j0] + t1 * d0[i0,j0+1]) + \
                     s1 * (t0 * d0[i0+1,j0] + t1 * d0[i0+1,j0+1])

def step():
    """执行一个完整的仿真步"""
    global ux, uy, dens
    
    # 速度扩散
    ux_prev = ux.copy()
    uy_prev = uy.copy()
    diffuse(2, -ux, ux_prev, visc)
    diffuse(3, -uy, uy_prev, visc)
    
    # 投影速度场
    project(ux, uy, np.zeros_like(ux), np.zeros_like(ux))
    
    # 速度平流
    ux_prev = ux.copy()
    uy_prev = uy.copy()
    advect(2, ux, ux_prev, ux_prev, uy_prev)
    advect(3, uy, uy_prev, ux_prev, uy_prev)
    
    # 再次投影
    project(ux, uy, np.zeros_like(ux), np.zeros_like(ux))
    
    # 密度扩散和平流
    dens_prev = dens.copy()
    diffuse(0, dens, dens_prev, diff)
    dens_prev = dens.copy()
    advect(0, dens, dens_prev, ux, uy)

# 设置交互事件
def on_mouse_move(event):
    global mouse_down, last_mx, last_my
    if mouse_down and event.xdata is not None and event.ydata is not None:
        mx = int(event.xdata)
        my = int(event.ydata)
        if 0 < mx < N and 0 < my < N:
            # 计算鼠标移动的方向
            dx = mx - last_mx
            dy = my - last_my   # 修正 dy 的方向，因为鼠标 y 坐标是逆序的
            
            # 添加密度和速度
            add_density(mx, my, 100)
            add_velocity(mx, my, dx * 5, dy * 5)
            
            last_mx, last_my = mx, my

def on_mouse_down(event):
    global mouse_down, last_mx, last_my
    mouse_down = True
    if event.xdata is not None and event.ydata is not None:
        last_mx = int(event.xdata)
        last_my = int(event.ydata)

def on_mouse_up(event):
    global mouse_down
    mouse_down = False

# 初始化图形界面
fig, ax = plt.subplots()
img = ax.imshow(dens, cmap='plasma', vmin=0, vmax=100, origin='lower')  # origin='lower' 使 (0,0) 在左下角
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_mouse_down)
fig.canvas.mpl_connect('button_release_event', on_mouse_up)

# 设置坐标轴范围和刻度
ax.set_xlim(0, N)
ax.set_ylim(0, N)
ax.set_xticks(np.linspace(0, N, 10))  # 添加 x 轴刻度
ax.set_yticks(np.linspace(0, N, 10))  # 添加 y 轴刻度

def update(frame):
    step()
    dens[1:-1, 1:-1] *= 0.995  # 逐渐减少密度
    img.set_data(dens)
    return [img]

ani = FuncAnimation(fig, update, frames=200, interval=10, blit=True)
plt.show()