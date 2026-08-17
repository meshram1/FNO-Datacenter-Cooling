"""Render a temperature-evolution GIF from a cooling / aisle dataset.

    python animate.py --data data/aisle_64.npz --out figures/cooling.gif
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aisle_64.npz")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--out", default="figures/cooling.gif")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    d = np.load(args.data)
    T = d["T"][args.traj]                        # (frames, H, W)
    tv = float(np.abs(T).max())
    has_S = "S" in d.files
    S = d["S"][args.traj] if has_S else None

    if has_S:
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.7))
        sv = float(np.abs(S).max())
        ax[0].imshow(S.T, cmap="coolwarm", origin="lower", vmin=-sv, vmax=sv)
        ax[0].set_title("racks (red) + cold vents (blue)"); ax[0].axis("off")
        im = ax[1].imshow(T[0].T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv)
        ax[1].set_title("temperature"); ax[1].axis("off")
    else:
        fig, a = plt.subplots(figsize=(4, 4))
        im = a.imshow(T[0].T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv)
        a.set_title("temperature"); a.axis("off")

    def upd(i):
        im.set_data(T[i].T)
        return [im]

    ani = animation.FuncAnimation(fig, upd, frames=T.shape[0],
                                  interval=1000 // args.fps, blit=True)
    fig.tight_layout()
    ani.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    print("saved", args.out)


if __name__ == "__main__":
    main()
