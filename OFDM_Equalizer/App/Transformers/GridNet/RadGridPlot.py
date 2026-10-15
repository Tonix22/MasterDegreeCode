import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import sys
import struct  # if you still need it elsewhere

main_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../"
sys.path.insert(0, main_path + "controllers")
from Recieved import RX  # if you still need it


class PolarGridCode():
    def __init__(self, step_radius, step_angle):
        self.step_radius = step_radius
        self.step_angle  = step_angle  # e.g. np.pi/6

        self.binsr = torch.arange(0, 1 + self.step_radius, self.step_radius, dtype=torch.float64)
        self.binsa = torch.arange(-np.pi, np.pi + self.step_angle, self.step_angle, dtype=torch.float64)

        self.binra = torch.arange(start=0, end=(len(self.binsr) - 1) * len(self.binsa)) \
                       .view(len(self.binsr) - 1, len(self.binsa))

    def Encode(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        r, a = data.abs(), data.angle()

        r_indices = torch.zeros(r.shape, dtype=torch.long)
        a_indices = torch.zeros(a.shape, dtype=torch.long)

        for i in range(len(self.binsr) - 1):
            r_mask = (self.binsr[i] <= r) & (r < self.binsr[i + 1])
            r_indices[r_mask] = i

        for j in range(len(self.binsa) - 1):
            a_mask = (self.binsa[j] <= a) & (a < self.binsa[j + 1])
            a_indices[a_mask] = j

        encoded = self.binra[r_indices, a_indices]
        return encoded

    def Decode(self, encoded):
        if isinstance(encoded, np.ndarray):
            encoded = torch.from_numpy(encoded)

        r_decoded = torch.zeros(encoded.shape, dtype=torch.float64)
        a_decoded = torch.zeros(encoded.shape, dtype=torch.float64)

        for i in range(len(self.binsr) - 1):
            for j in range(len(self.binsa) - 1):
                indices = (encoded == self.binra[i, j])
                r_decoded[indices] = self.binsr[i] + self.step_radius / 2
                a_decoded[indices] = self.binsa[j] + self.step_angle / 2

        data_decoded = torch.polar(r_decoded, a_decoded)
        return data_decoded

    def _set_polar_ticks(self, ax):
        """Set r/theta ticks avoiding 0° and 360° overlapping."""
        r_ticks = self.binsr
        a_ticks = (self.binsa + torch.pi)         # shift [-π,π] -> [0,2π]
        a_deg = (a_ticks * 180 / np.pi).numpy()
        a_deg = a_deg[:-1]                        # drop 360°

        ax.set_rticks(r_ticks)
        ax.set_thetagrids(a_deg)

        ax.set_rlim(0, 1)
        ax.set_ylim(0, 1)

    def plot_grid_values(self, ax):
        grid_num = 0
        for i in range(len(self.binsr) - 1):
            for j in range(len(self.binsa) - 1):
                r = (self.binsr[i] + self.binsr[i + 1]) / 2
                a = (self.binsa[j] + self.binsa[j + 1]) / 2
                ax.text(a, r, str(grid_num), fontsize=8, ha='center', va='center')
                grid_num += 1

        self._set_polar_ticks(ax)
        ax.set_title("Grid Encoding")
        ax.set_xlabel("Angle")
        ax.set_ylabel("Radius")

    def _get_colors(self, n):
        cmap = cmx.jet
        c_norm = colors.Normalize(vmin=0, vmax=n - 1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
        return [scalar_map.to_rgba(i) for i in range(n)]


    def plot_original_and_decoded(self, ax, original, decoded, colorsMap, title):
        if isinstance(original, np.ndarray):
            original = torch.from_numpy(original)
        if isinstance(decoded, np.ndarray):
            decoded = torch.from_numpy(decoded)

        n = len(original)

        r_o, a_o = torch.abs(original), torch.angle(original)
        r_d, a_d = torch.abs(decoded), torch.angle(decoded)

        # Original: stars
        ax.scatter(a_o, r_o, s=60, c=colorsMap, marker='*', label="Original")

        # Decoded: squares
        ax.scatter(a_d, r_d, s=40, c=colorsMap, marker='s', label="Decoded")

        self._set_polar_ticks(ax)

        ax.set_title(title)
        ax.set_xlabel("Angle")
        ax.set_ylabel("Radius")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))


    def plot_table_original_decoded(self, ax, original, decoded, encoded, colorsMap):
        if isinstance(original, np.ndarray):
            original = torch.from_numpy(original)
        if isinstance(decoded, np.ndarray):
            decoded = torch.from_numpy(decoded)
        if isinstance(encoded, np.ndarray):
            encoded = torch.from_numpy(encoded)

        n = len(original)

        table_data = []
        for i in range(n):
            z_o = complex(original[i].item())
            z_d = complex(decoded[i].item())
            grid_id = int(encoded[i].item())

            orig_str = f"{z_o.real:+.3f} + {z_o.imag:+.3f}j"
            dec_str  = f"{z_d.real:+.3f} + {z_d.imag:+.3f}j"

            table_data.append([
                i,
                orig_str,
                dec_str,
                grid_id,
            ])

        col_labels = ["idx", "original z", "decoded z", "grid id"]

        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        ax.set_title("Samples (same colors as plot)")

        # 🎨 Color each row with the same color as its point
        # row 0 in the table is the header, so data row i -> table row (i+1)
        for i in range(n):
            row_color = colorsMap[i]
            for j in range(len(col_labels)):
                cell = table[(i + 1, j)]
                cell.set_facecolor(row_color)
                # optional: darker edge for contrast
                cell.set_edgecolor("black")


step_radius = 0.25
step_angle = np.pi / 6
coding = PolarGridCode(step_radius, step_angle)

# Random complex data
data = np.random.rand(10) * np.exp(1j * np.random.rand(10) * 2 * np.pi)

encoded = coding.Encode(data)
decoded = coding.Decode(encoded)
data_torch = torch.from_numpy(data)

n = len(data_torch)
colorsMap = coding._get_colors(n)   # shared colors

fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 2])

ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
ax_table = fig.add_subplot(gs[1, 0])

coding.plot_grid_values(ax_polar)  # optional grid indices
coding.plot_original_and_decoded(ax_polar, data_torch, decoded, colorsMap,
                                 "Original (*) vs Decoded (□)")

coding.plot_table_original_decoded(ax_table, data_torch, decoded, encoded, colorsMap)

plt.tight_layout()
plt.savefig("OriginalVsDecoded_withColoredTable.png", dpi=200)
#plt.show()

