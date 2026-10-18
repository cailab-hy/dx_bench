"""Side (x-z) and top (x-y) views of the towel fold test node clouds: flat / held / end (after release).
    python3 tools/cloth_spike/plot_towel.py reports/towel_fold/runs/<tag>_nodes.npz ... -> reports/towel_fold/<tags>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

files = sys.argv[1:]
fig, ax = plt.subplots(len(files), 2, figsize=(11, 2.6 * len(files)), squeeze=False)
for r, f in enumerate(files):
    d = np.load(f)
    tag = Path(f).stem.replace("_nodes", "")
    for k, c in (("flat", "0.8"), ("held", "tab:blue"), ("end", "tab:red")):
        n = d[k]
        ax[r, 0].scatter(n[:, 0], n[:, 2] - 0.725, s=2, c=c, label=k)
        ax[r, 1].scatter(n[:, 0], n[:, 1], s=2, c=c, label=k)
    ax[r, 0].set_title(f"{tag}: side (x, z above desk)")
    ax[r, 1].set_title(f"{tag}: top (x, y)")
    ax[r, 0].set_ylim(-0.01, 0.2); ax[r, 0].set_aspect("equal")
    ax[r, 1].set_aspect("equal")
    ax[r, 0].legend(fontsize=7, loc="upper left")
plt.tight_layout()
out = Path("reports/towel_fold") / ("_".join(Path(f).stem.replace("_nodes", "") for f in files) + ".png")
plt.savefig(out, dpi=90)
print(out)
