"""
余地の模擬（headroom.py）で見えた「床は2回目以降の切り替えで発火しなくなる」の原因の確かめ（診断。判定には使わない）。

    .venv/Scripts/python.exe step3_memory/precheck/diagnose_floor.py [--device cuda]

見立て：最初の学び直しが不正確（学び残し）なので、4歩先の外れの大きさ σ₄ が、正確な係数のときより大きいところで落ち着き、
次の切り替えの z₄² がしきい値に届かなくなる（慣れ）。床と天井で、切り替えの直前の σ₄（正確な係数の定常の大きさとの比）と、
新しい波が届いてから30ステップの z₄² の平均、窓の中で1回でも発火したセルの割合を比べる。
ノイズ0.02、評価の乱数300、headroom.py と同じ世界・同じ並び。出力：step3_memory/runs/precheck_headroom/diagnose_floor.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import headroom as H  # noqa: E402
from headroom import W, AttentionCell, Physio2, exact_w_t, noise_gain, rewrite  # noqa: E402


@torch.no_grad()
def trace(variant, wcfg, seed, dev):
    B = H.N_PER * len(H.GROUPS)
    S_ = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in H.GROUPS for _ in range(H.N_PER)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in H.SEQ]
    T = len(H.SEQ) * H.DWELL
    g = torch.Generator(device=dev).manual_seed(seed)
    st, ws = H.adult_init(wcfg, sched[0], g, dev)
    cell = AttentionCell(dna=False).to(dev)
    wave = ~W.zone_mask(wcfg, dev)
    SK = torch.zeros(T, B, S_, device=dev)        # 列ごとの σ₄ の中央値（波の行）
    ZK = torch.zeros(T, B, S_, device=dev)        # 列ごとの z₄² の平均
    FI = torch.zeros(T, B, S_, S_, dtype=torch.bool, device=dev)
    for t in range(T):
        j = t // H.DWELL
        if t % H.DWELL == H.DWELL - 1 and j + 1 < len(H.SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        st, rec = cell.step(st, sense, policy={"kind": "trigger"})
        if variant == "天井":
            w_star = exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, S_, S_)
            st = rewrite(st, w_star, rec["hard"] & wave)
        sk = st.vk.clamp(min=Physio2.sigma_min ** 2).sqrt()[:, 0]
        for x in range(S_):
            rows = wave[:, x]
            SK[t, :, x] = sk[:, rows, x].median(1).values
            ZK[t, :, x] = rec["zk2"][:, 0][:, rows, x].mean(1)
        FI[t] = rec["hard"][:, 0]
        ws = W.advance(ws)
    return SK, ZK, FI, sched, wave


def sig4_exact(noise, P):
    a, b = noise_gain(P, 4)
    return max(noise * (1 + a * a + b * b) ** 0.5, Physio2.sigma_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--groups", default="", help="追加の診断：組を差し替える（例 8,12,16）。読み方は事前に決めていない")
    args = ap.parse_args()
    tag = ""
    if args.groups:
        H.GROUPS = (tuple(float(v) for v in args.groups.split(",")),)
        tag = "_" + args.groups.replace(",", "-")
    wcfg = W.WorldConfig(noise=0.02)
    lines = ["# 床が2回目以降の切り替えで発火しなくなる原因の確かめ（ノイズ0.02、評価の乱数300）", "",
             *(["**追加の診断（組を差し替えた。読み方は事前に決めていない）**", ""] if args.groups else []),
             "σ₄比＝切り替えの直前50ステップの σ₄（波の区域の列ごとの中央値を列で平均）÷ いまの周期の正確な係数での σ₄ の定常の大きさ。",
             "z₄²（届いてから30）＝新しい波が届いてから30ステップの z₄² の平均（列で平均）。反応＝窓（届いてから300）の中で1回でも発火した波の区域のセルの割合", ""]
    out = {}
    for var in ("床", "天井"):
        SK, ZK, FI, sched, wave = trace(var, wcfg, 300, args.device)
        out[var] = (SK, ZK, FI, sched)
    for gi, grp in enumerate(H.GROUPS):
        sl = slice(gi * H.N_PER, (gi + 1) * H.N_PER)
        lines += [f"## A={grp[0]:g}, B={grp[1]:g}, C={grp[2]:g}", "",
                  "| 切り替え | 床：σ₄比（直前） | 天井：σ₄比（直前） | 床：z₄²（届いてから30） | 天井：z₄²（届いてから30） | 床：反応 | 天井：反応 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for j in range(1, len(H.SEQ)):
            P_old = float(out["床"][3][j - 1][sl][0])
            row = [f"{H.SEQ[j - 1]}→{H.SEQ[j]}（{P_old:g}→{float(out['床'][3][j][sl][0]):g}）"]
            vals = {}
            for var in ("床", "天井"):
                SK, ZK, FI, sched = out[var]
                t_sw = j * H.DWELL
                ratio = float(SK[t_sw - 50:t_sw, sl].mean()) / sig4_exact(wcfg.noise, P_old)
                zs, react = [], []
                for x in range(wcfg.size):
                    a0 = t_sw + x * wcfg.lag
                    zs.append(float(ZK[a0:a0 + 30, sl, x].mean()))
                    rows = wave[:, x]
                    react.append(float(FI[a0:a0 + H.WINDOW, sl][:, :, rows, x].any(0).float().mean()))
                vals[var] = (ratio, float(np.mean(zs)), float(np.mean(react)))
            row += [f"{vals['床'][0]:.2f}", f"{vals['天井'][0]:.2f}", f"{vals['床'][1]:.1f}", f"{vals['天井'][1]:.1f}",
                    f"{vals['床'][2]:.2f}", f"{vals['天井'][2]:.2f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    path = H.ROOT / "step3_memory" / "runs" / "precheck_headroom" / f"diagnose_floor{tag}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
