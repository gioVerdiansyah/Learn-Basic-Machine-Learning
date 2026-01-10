import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# =====================
# KONSTANTA FISIKA
# =====================
v = 343
f = 40000
omega = 2 * np.pi * f

# =====================
# AREA WAKTU
# =====================
t = np.linspace(0, 0.0005, 3000)
dt = 0.00002

# =====================
# JARAK AWAL
# =====================
jarak_awal = 50  # cm

# =====================
# PLOT
# =====================
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

(line_tx,) = ax.plot(
    t * 1000, np.zeros_like(t), color="red", linewidth=2, label="Gelombang Dikirim"
)
(line_rx,) = ax.plot(
    t * 1000, np.zeros_like(t), color="blue", linewidth=2, label="Gelombang Pantul"
)

ax.set_xlim(0, 0.5)
ax.set_ylim(-1.5, 1.5)
ax.grid()

# =====================
# SLIDER JARAK
# =====================
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "Jarak (cm)", 5, 200, valinit=jarak_awal)

# =====================
# ANIMASI
# =====================
time_shift = 0


def animate(frame):
    global time_shift

    # Waktu pantul
    d = slider.val / 100
    delay = 2 * d / v

    attenuation = 1 / (1 + d)

    # ke kanan
    y_tx = np.sin(omega * (t - time_shift))

    # ke kiri (arah dibalik)
    y_rx = attenuation * np.sin(omega * (t + time_shift - delay))

    line_tx.set_ydata(y_tx)
    line_rx.set_ydata(y_rx)

    time_shift += dt

    ax.set_title(f"Jarak = {slider.val:.1f} cm | Waktu Pantul = {delay*1000:.2f} ms")

    return line_tx, line_rx


ani = FuncAnimation(fig, animate, interval=20)
plt.show()
