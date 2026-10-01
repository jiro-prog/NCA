"""
(し1) の A の重複が、なぜ「別のパターン」と判定されたか（しまうタイミングの診断の補い。読み方は事前に決めていない。判定には使わない）。

    .venv/Scripts/python.exe step3_memory/precheck/storage_why.py [--device cuda]

12/20 と 12/24 の組で、(し1) がしまう瞬間ごとに、しまう係数 w と、いちばん合う保存先 u について、
  当てはまりの差 (u − w)ᵀR_L(u − w) ÷ σ²（6 以下なら寄せる）、w と u の、正確な A との予測の差（きれいな波の上の1歩先、二乗平均の平方根）
を、A に当たる保存先があるのに新しくしまった瞬間（重複のもと）と、寄せた瞬間に分けて並べる。出力：step3_memory/runs/precheck_storage/why.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import storage_check as SC  # noqa: E402
from storage_check import C, E, W, Physio2, pred_gap, quad  # noqa: E402


@torch.no_grad()
def run(wcfg, seed, start, dev, plan=E.Plan()):
    B = plan.n_per * len(plan.groups)
    S_ = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in plan.groups for _ in range(plan.n_per)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in plan.seq]
    T = len(plan.seq) * plan.dwell
    g = torch.Generator(device=dev).manual_seed(seed)
    if start == "adult":
        core, ws = E.adult_core(wcfg, sched[0], g, dev)
        st = C.wrap(core)
    else:
        ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
        st = C.init_state(B, S_, S_, dev)
    cell = C.Cell3(sleep=True, device=dev)
    book = SC.Book(B, S_, S_, dev)
    wave = ~W.zone_mask(wcfg, dev)
    ev = {gi: dict(dup=[], merge=[]) for gi in range(len(plan.groups))}
    for t in range(T):
        j = t // plan.dwell
        if t % plan.dwell == plan.dwell - 1 and j + 1 < len(plan.seq):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        st, rec = cell.step(st, sense)
        book.tick()
        M = rec["sleep"][:, 0] & (st.k[:, 0] >= 2)
        if bool(M.any()):
            sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
            w = st.core.w
            occ = book.n > 0
            d = torch.stack([quad(st.RL, book.slot[:, i] - w) for i in range(SC.K)], 1)
            d = torch.where(occ, d, torch.full_like(d, float("inf")))
            d_best, i_best = d.min(1)
            u = torch.gather(book.slot, 1, i_best.view(B, 1, 1, S_, S_).expand(B, 1, 3, S_, S_)).squeeze(1)
            for gi, grp in enumerate(plan.groups):
                sl_ = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
                PA = grp[0]
                gw = pred_gap(w[sl_].permute(0, 2, 3, 1), PA)
                gu = pred_gap(u[sl_].permute(0, 2, 3, 1), PA)
                m = M[sl_] & wave & (gw <= SC.MATCH) & occ[sl_].any(1)               # A に当たる係数をしまう瞬間
                r = (d_best[sl_] / sig2[sl_])
                isdup = m & (r > SC.TAU) & (gu <= SC.MATCH)                            # A に当たる保存先があるのに「別」と判定
                ismerge = m & (r <= SC.TAU)
                for key, mm in (("dup", isdup), ("merge", ismerge)):
                    if bool(mm.any()):
                        ev[gi][key].append(torch.stack([r[mm], gw[mm], gu[mm]], 1).cpu())
        book.store(M, st.core.w, st.RL, st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0])
        ws = W.advance(ws)
    out = []
    for gi, grp in enumerate(plan.groups):
        row = dict(group="/".join(f"{p:g}" for p in grp))
        for key in ("dup", "merge"):
            a = torch.cat(ev[gi][key]).numpy() if ev[gi][key] else np.zeros((0, 3))
            row[key] = dict(n=int(a.shape[0]), r_med=float(np.median(a[:, 0])) if len(a) else None, r_q90=float(np.quantile(a[:, 0], 0.9)) if len(a) else None,
                            gw_med=float(np.median(a[:, 1])) if len(a) else None, gu_med=float(np.median(a[:, 2])) if len(a) else None)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    L = ["# (し1) の A の重複は、なぜ「別のパターン」と判定されたか（読み方は事前に決めていない）", "",
         "A に当たる係数をしまう瞬間を、「A に当たる保存先があるのに新しくしまった（重複のもと）」と「寄せた」に分けた。",
         "差÷σ² は当てはまりの差 (u − w)ᵀR_L(u − w) をそのセルの σ² で割ったもの（6 以下なら寄せる）。w・u の A との差は、きれいな波の上の1歩先の予測の差（0.02 以下で A に当たる）", "",
         "| 始め方 | 評価の乱数 | 組 | 重複のもと：回数 | 差÷σ²（中央値／90%点） | w・u の A との差（中央値） | 寄せた：回数 | 差÷σ²（中央値／90%点） | w・u の A との差（中央値） |",
         "|---|---:|---|---:|---|---|---:|---|---|"]
    f = lambda x: "—" if x is None else f"{x:.3g}"
    for start in ("adult", "grow"):
        for seed in (300, 301):
            for r in run(wcfg, seed, start, args.device):
                dd, mm = r["dup"], r["merge"]
                L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {r['group']} | {dd['n']} | {f(dd['r_med'])}／{f(dd['r_q90'])} | {f(dd['gw_med'])}・{f(dd['gu_med'])} | "
                         f"{mm['n']} | {f(mm['r_med'])}／{f(mm['r_q90'])} | {f(mm['gw_med'])}・{f(mm['gu_med'])} |")
            print(start, seed, "done", flush=True)
    out = SC.ROOT / "step3_memory" / "runs" / "precheck_storage" / "why.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
