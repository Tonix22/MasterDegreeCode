import os, sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Cargar clase RX (ajusta la ruta si es necesario)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "controllers"))
from Recieved import RX

# ------------------------------------------------------------------
# 2) Parámetros y obtención de datos
QAM, realization = 4, 6063
rx = RX(QAM, "Unit_Pow", "Complete", "cpu")

# Vector recibido Y normalizado
Y = rx.Qsym.r[:, realization]
Y = Y / np.max(np.abs(Y))

# Vector corregido u = HᴴY normalizado
u = np.asmatrix(rx.H[:, :, realization]).H @ rx.Qsym.r[:, realization]
u = np.asarray(u).flatten()            # pasar de matriz 1-col a 1-D
u = u / np.max(np.abs(u))

# ------------------------------------------------------------------
# 3) Gráfica conjunta
plt.figure(figsize=(6, 6))

plt.scatter(Y.real, Y.imag,
            c='tab:blue', marker='o', s=12, alpha=0.7, label=r'$u = Gs + z$')

plt.scatter(u.real, u.imag,
            c='tab:orange', marker='x', s=20, alpha=0.8, label=r'$u_d = G^{\mathrm{H}}u$')

# Ejes y estética
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.gca().set_aspect('equal', 'box')
plt.grid(True, linestyle='--', alpha=0.3)

step = 0.25
ticks = np.arange(-1, 1 + step, step)
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.legend()
plt.tight_layout()

plt.savefig('constelaciones_Y_vs_u.png', dpi=900)
# plt.show()  # Descomenta si prefieres visualizar en pantalla

print("Gráfica guardada como constelaciones_Y_vs_u.png")
