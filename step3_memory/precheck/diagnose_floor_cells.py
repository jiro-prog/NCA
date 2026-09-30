"""
床の慣れの原因の確かめ、その2（追加の診断。読み方は事前に決めていない。判定には使わない）。

    .venv/Scripts/python.exe step3_memory/precheck/diagnose_floor_cells.py [--start adult|grow] [--device cuda]

diagnose_floor.py では、届いてから30ステップの z₄² の平均が床と天井でそれほど違わないのに、反応したセルは大きく違った。そこで：
(a) セルごとに見る：切り替えの直前50ステップの σ₄（そのセル）と、そのセルが反応したか（窓の中で1回でも発火）の関係。
    届いてから5ステップの z₄² の平均と、30ステップの中で濃度（リセット前）がしきい値を超えたセルの割合も、床と天井で並べる
(b) --start grow：人工の「正確に合わせた状態」ではなく、予測器0から A で600ステップ育った状態から同じ並びを回す（床だけ）
ノイズ0.02、評価の乱数300、組 12/24/8・12/20/8、並びは headroom.py と同じ。
出力：step3_memory/runs/precheck_headroom/diagnose_floor_cells_<start>.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import headroom as H  # noqa: E402
from headroom import W, AttentionCell, Physio2, exact_w_t, init_state, noise_gain, rewrite  # noqa: E402

BINS = (0.0, 1.2, 1.5, 2.0, 3.0, float("inf"))


def sig4_exact(noise, P):
    a, b = noise_gain(P, 4)
    return max(noise * (1 + a * a + b * b) ** 0.5, Physio2.sigma_min)


@torch.no_grad()
def trace(variant, wcfg, seed, dev, start):
    B = H.N_PER * len(H.GROUPS)
    S_ = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in H.GROUPS for _ in range(H.N_PER)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in H.SEQ]
    T = len(H.SEQ) * H.DWELL
    n_sw = len(H.SEQ) - 1
    g = torch.Generator(device=dev).manual_seed(seed)
    if start == "adult":
        st, ws = H.adult_init(wcfg, sched[0], g, dev)
    else:
        ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
        st = init_state(B, S_, S_, dev)
    cell = AttentionCell(dna=False).to(dev)
    wave = ~W.zone_mask(wcfg, dev)
    xs = torch.arange(S_, device=dev)
    lag = wcfg.lag
    sig_pre = torch.zeros(n_sw, B, S_, S_, device=dev)      # 切り替えの直前50ステップの σ₄ の平均（セルごと）
    z5 = torch.zeros_like(sig_pre)                           # 届いてから5ステップの z₄² の平均
    over30 = torch.zeros_like(sig_pre, dtype=torch.bool)     # 届いてから30ステップの中で、濃度（リセット前）がしきい値を超えた
    fired = torch.zeros_like(sig_pre, dtype=torch.bool)      # 届いてから300ステップの中で1回でも発火
    for t in range(T):
        j = t // H.DWELL
        if t % H.DWELL == H.DWELL - 1 and j + 1 < len(H.SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        c_prev = st.c
        st, rec = cell.step(st, sense, policy={"kind": "trigger"})
        if variant == "天井":
            w_star = exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, S_, S_)
            st = rewrite(st, w_star, rec["hard"] & wave)
        c_pre = Physio2.c_decay * c_prev + (1 - Physio2.c_decay) * rec["zk2"]          # その回の引き金の判定に使った濃度
        sk = st.vk.clamp(min=Physio2.sigma_min ** 2).sqrt()[:, 0]
        if j + 1 < len(H.SEQ) and t >= (j + 1) * H.DWELL - 50:                         # 次の切り替えの直前50ステップ
            sig_pre[j] += sk / 50
        if j >= 1:
            sw = j - 1
            rel = (t - (j * H.DWELL + xs * lag)).view(1, 1, S_)
            z5[sw] += rec["zk2"][:, 0] * ((rel >= 0) & (rel < 5)).float() / 5
            over30[sw] |= (c_pre[:, 0] > Physio2.theta) & (rel >= 0) & (rel < 30)
            fired[sw] |= rec["hard"][:, 0] & (rel >= 0) & (rel < H.WINDOW)
        ws = W.advance(ws)
    return dict(sig_pre=sig_pre, z5=z5, over30=over30, fired=fired, sched=sched, wave=wave)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--start", default="adult", choices=("adult", "grow"))
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    variants = ("床", "天井") if args.start == "adult" else ("床",)
    res = {v: trace(v, wcfg, 300, args.device, args.start) for v in variants}
    head = "学び終えた大人の状態から" if args.start == "adult" else "予測器0から A で600ステップ育った状態から"
    L = [f"# 床の慣れの原因の確かめ、その2（{head}。ノイズ0.02、評価の乱数300）", "",
         "**追加の診断（読み方は事前に決めていない。判定には使わない）**", "",
         "σ₄比＝切り替えの直前50ステップのそのセルの σ₄ ÷ 古い周期の正確な係数での σ₄ の定常の大きさ（波の区域の中央値）。z₄²（5）＝届いてから5ステップの z₄² の平均。",
         "しきい値越え（30）＝届いてから30ステップの中で、濃度（リセット前）がしきい値を超えたセルの割合。反応＝届いてから300ステップの中で1回でも発火したセルの割合", ""]
    for gi, grp in enumerate(H.GROUPS):
        sl = slice(gi * H.N_PER, (gi + 1) * H.N_PER)
        cols = " | ".join(f"{v}：{k}" for v in variants for k in ("σ₄比", "z₄²（5）", "しきい値越え（30）", "反応"))
        L += [f"## A={grp[0]:g}, B={grp[1]:g}, C={grp[2]:g}", "", f"| 切り替え | {cols} |", "|---|" + "---:|" * (4 * len(variants))]
        for sw in range(len(H.SEQ) - 1):
            P_old = float(res[variants[0]]["sched"][sw][sl][0])
            P_new = float(res[variants[0]]["sched"][sw + 1][sl][0])
            vals = []
            for v in variants:
                r = res[v]
                wv = r["wave"]
                ratio = (r["sig_pre"][sw][sl][:, wv] / sig4_exact(wcfg.noise, P_old)).median()
                vals += [f"{float(ratio):.2f}", f"{float(r['z5'][sw][sl][:, wv].mean()):.1f}",
                         f"{float(r['over30'][sw][sl][:, wv].float().mean()):.2f}", f"{float(r['fired'][sw][sl][:, wv].float().mean()):.2f}"]
            L.append(f"| {H.SEQ[sw]}→{H.SEQ[sw + 1]}（{P_old:g}→{P_new:g}） | " + " | ".join(vals) + " |")
        # セルごと：床の σ₄比の区間ごとの反応（切り替えごと）
        r = res["床"]
        wv = r["wave"]
        L += ["", "床：切り替えの直前の σ₄比の区間ごとの反応（セルの数）", "",
              "| 切り替え | " + " | ".join(f"{BINS[i]:g}〜{BINS[i + 1]:g}" for i in range(len(BINS) - 1)) + " |",
              "|---|" + "---:|" * (len(BINS) - 1)]
        for sw in range(len(H.SEQ) - 1):
            P_old = float(r["sched"][sw][sl][0])
            ratio = (r["sig_pre"][sw][sl][:, wv] / sig4_exact(wcfg.noise, P_old)).flatten().cpu().numpy()
            f = r["fired"][sw][sl][:, wv].flatten().cpu().numpy()
            idx = np.digitize(ratio, BINS[1:-1])
            cells = []
            for b_ in range(len(BINS) - 1):
                m = idx == b_
                cells.append(f"{f[m].mean():.2f}（{m.sum()}）" if m.sum() else "—")
            L.append(f"| {H.SEQ[sw]}→{H.SEQ[sw + 1]} | " + " | ".join(cells) + " |")
        L.append("")
    path = H.ROOT / "step3_memory" / "runs" / "precheck_headroom" / f"diagnose_floor_cells_{args.start}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
