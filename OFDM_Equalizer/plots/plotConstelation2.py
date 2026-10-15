import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx

# ------------------------------------------------------------------
# 1) Rutas y carga de clases ---------------------------------------
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:  # __file__ no existe en intérprete interactivo
    ROOT = os.getcwd()

sys.path.insert(0, os.path.join(ROOT, "..", "controllers"))

from Recieved import RX      # Asegúrate de que el nombre sea correcto

# ------------------------------------------------------------------
# 2) Parámetros de experimento -------------------------------------
QAM         = 4
realization = 6063
step        = 1 / 4            # tamaño de bin para ejes

# ------------------------------------------------------------------
# 3) Generación / carga de datos -----------------------------------
rx    = RX(QAM, "Unit_Pow", "Complete", "cpu")
bits  = rx.Qsym.bits[:, realization].squeeze()

# Y = Hx + w, normalizado
Y = rx.Qsym.r[:, realization]
Y = Y / np.max(np.abs(Y))

# u = Hᴴ Y, normalizado
u = np.array(np.matrix(rx.H[:, :, realization]).H @ rx.Qsym.r[:, realization])
u = u / np.max(np.abs(u))

# ------------------------------------------------------------------
# 4) Configuración de colores --------------------------------------
cmap         = cmx.jet
norm         = mcolors.Normalize(vmin=0, vmax=48)
scalar_map   = cmx.ScalarMappable(norm=norm, cmap=cmap)
point_colors = [scalar_map.to_rgba(i) for i in range(48)]

# ------------------------------------------------------------------
# 5) Función auxiliar de pintado -----------------------------------
def plot_constellation(ax, vect, colors, title):
    """Dibuja un vector complejo en ax."""
    ax.scatter(vect.real, vect.imag, s=10, c=colors, marker='*')
    ax.set_xlabel('Parte real')
    ax.set_ylabel('Parte imaginaria')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    # límites y ticks
    bins = np.arange(-1, 1 + step, step)
    ax.set_xticks(bins)
    ax.set_yticks(bins)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

# ------------------------------------------------------------------
# 6) Pintado --------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

plot_constellation(ax1, Y, point_colors, r'$u = Gs + z$')
plot_constellation(ax2, u, point_colors, r'$u_d = G^{\mathrm{H}}u$')

fig.tight_layout()
fig.savefig('result2.png', dpi=300)
# fig.show()   # Descomenta si corres en un IDE

print("Gráfica guardada como result2.png")
