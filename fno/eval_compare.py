"""Apples-to-apples one-step accuracy: from-scratch FNO vs PhysicsNeMo FNO.

Both models see the SAME input windows from the SAME trajectories, and errors are
measured in PHYSICAL units after decoding -- so neither the differing normalization
schemes nor the differing train splits can flatter either model.

Split note: the from-scratch model held out the last 40 trajectories
(dataset_cool.load_cooling, n_test=40); the PhysicsNeMo model held out the last 20
(modulus_dataset, train_frac=0.9). The PhysicsNeMo model therefore TRAINED on
trajectories 160-179, which the from-scratch model treats as test. Only the last 20
are unseen by both, so those are the evaluation set.

    python eval_compare.py --data ../data/aisle_64.npz
"""
import argparse

import numpy as np
import torch

from fno2d_cool import FNO2d
from modulus_model import build_model, T_IN

N_EVAL = 20            # last 20 trajectories -- held out by BOTH models
N_TEST_SCRATCH = 40    # the from-scratch model's own held-out count (for its S stats)


def rel_l2_per_channel(pred, true):
    """pred/true: (B, C, H, W) in physical units -> (B, C) relative L2."""
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = true.reshape(true.shape[0], true.shape[1], -1)
    return torch.linalg.norm(p - t, dim=2) / (torch.linalg.norm(t, dim=2) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/aisle_64.npz")
    ap.add_argument("--scratch", default="aisle_fno.pt")
    ap.add_argument("--modulus", default="aisle_modulus.pt")
    ap.add_argument("--bs", type=int, default=25)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data)
    S = torch.from_numpy(d["S"]).float()
    field = torch.stack([torch.from_numpy(d["w"]).float(),
                         torch.from_numpy(d["T"]).float()], dim=2)   # (N, Nt, C, H, W) physical
    N, Nt, C, H, W = field.shape

    # --- from-scratch model: per-channel stats from its checkpoint --------------
    ck_s = torch.load(args.scratch, map_location=dev, weights_only=False)
    cs = ck_s["config"]
    m_s = FNO2d(cs["t_in"], cs["n_fields"], cs["n_static"],
                cs["modes"], cs["modes"], cs["width"], cs["layers"]).to(dev)
    m_s.load_state_dict(ck_s["model_state"]); m_s.eval()
    f_mean = ck_s["normalizer"]["mean"].to(dev).view(1, C, 1, 1)
    f_std = ck_s["normalizer"]["std"].to(dev).view(1, C, 1, 1)
    s_mean = S[:-N_TEST_SCRATCH].mean().to(dev)      # recomputed exactly as load_cooling does
    s_std = S[:-N_TEST_SCRATCH].std().to(dev)

    # --- PhysicsNeMo model: scalar per-field stats from its checkpoint ----------
    ck_m = torch.load(args.modulus, map_location=dev, weights_only=False)
    m_m = build_model(ck_m["config"]["modes"], ck_m["config"]["width"]).to(dev)
    m_m.load_state_dict(ck_m["model_state"]); m_m.eval()
    wm, ws, Tm, Ts, Sm, Ss = [torch.tensor(float(v), device=dev) for v in ck_m["stats"]]
    g_mean = torch.stack([wm, Tm]).view(1, C, 1, 1)
    g_std = torch.stack([ws, Ts]).view(1, C, 1, 1)

    idx = [(n, t) for n in range(N - N_EVAL, N) for t in range(Nt - T_IN)]
    tot_s = torch.zeros(C, device=dev); tot_m = torch.zeros(C, device=dev); cnt = 0

    with torch.no_grad():
        for b0 in range(0, len(idx), args.bs):
            chunk = idx[b0:b0 + args.bs]
            hist = torch.stack([field[n, t:t + T_IN] for n, t in chunk]).to(dev)  # (B,T_IN,C,H,W)
            true = torch.stack([field[n, t + T_IN] for n, t in chunk]).to(dev)    # (B,C,H,W)
            Sb = torch.stack([S[n] for n, _ in chunk]).to(dev)                    # (B,H,W)
            B = hist.shape[0]

            # from-scratch: channels-LAST, (B, H, W, t_in*C + 1); grid added inside the model
            hn = (hist - f_mean.unsqueeze(1)) / (f_std.unsqueeze(1) + 1e-8)
            x = hn.permute(0, 3, 4, 1, 2).reshape(B, H, W, T_IN * C)   # t-major, field-minor
            x = torch.cat([x, ((Sb - s_mean) / (s_std + 1e-8)).unsqueeze(-1)], dim=-1)
            pred_s = m_s(x).permute(0, 3, 1, 2) * f_std + f_mean       # -> (B,C,H,W) physical

            # PhysicsNeMo: channels-FIRST, (B, t_in*C + 1, H, W); same channel ordering
            hn2 = (hist - g_mean.unsqueeze(1)) / (g_std.unsqueeze(1) + 1e-8)
            x2 = hn2.reshape(B, T_IN * C, H, W)
            x2 = torch.cat([x2, ((Sb - Sm) / (Ss + 1e-8)).unsqueeze(1)], dim=1)
            pred_m = m_m(x2) * g_std + g_mean

            tot_s += rel_l2_per_channel(pred_s, true).sum(0)
            tot_m += rel_l2_per_channel(pred_m, true).sum(0)
            cnt += B

    rs, rm = (tot_s / cnt).tolist(), (tot_m / cnt).tolist()
    print(f"\n  one-step relative L2 (physical units) -- {cnt} windows from the "
          f"{N_EVAL} trajectories held out by BOTH models\n")
    print(f"  {'model':<22}{'vorticity w':>14}{'temperature T':>16}")
    print(f"  {'from-scratch FNO':<22}{rs[0]:>13.2%}{rs[1]:>16.2%}")
    print(f"  {'PhysicsNeMo FNO':<22}{rm[0]:>13.2%}{rm[1]:>16.2%}")


if __name__ == "__main__":
    main()


