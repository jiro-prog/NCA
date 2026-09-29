"""
候補B（数歩先も予測する）の実装前の模擬（DESIGN.md 13章。条件は 2026-09-29 に走らせる前に決めた）。

    .venv/Scripts/python.exe step2_attention/precheck/precheck_b.py

- 世界は今のもの（32×32、v=0.25、周期の変え方、右下のノイズの区域）で、ノイズと σ の下限だけ元の0.02 に戻す（S2-7 を丸ごと戻す）
- セルは DNA なし（η=1、g=1、Δw=0）。予測器・痕跡・最小二乗の書き換え・1歩先の σ（窓100）は今のまま
- B：書き換えたあとの係数を4回まわして4歩先を予測し、覚えておく。4歩先の外れ e₄ の移動平均 σ₄（窓100、下限0.02）で z₄² = e₄²/σ₄² を出し、
  濃度 c ← 0.9c ＋ 0.1·z₄² だけで驚く（1歩先の z² は引き金に使わない）。引き金は元の設計（1回書き換えて c を0、1ステップ休む）
- 質（NLL）は1歩先の予測と σ で測る。育つ時期と大人の時期は 7.5 の run_period と同じ世界・同じ乱数の並び
- 対照：DNA なしの時計とサイコロ（dnaless_curves.py の曲線、同じ世界）。参考：同じ世界での元の設計の引き金（1歩先の驚き）
- 合否：評価の乱数 1000・1001 の2本とも、両方の時期で、B が時計とサイコロの下の包みをどちらも「上回る」→ 見込みあり。それ以外は見込みなし
- 一致の確かめ：この回し方で元の設計の引き金を回すと、run_period ＋ 本物のセル（DNA なし）と数字が一致すること。一致しなければ判定の前に止める
- 見える度合い：大人の時期の組ごとに、新しい波が届いてから20ステップの z₄²・z² の平均（波の区域）を、止めた予測器（静）と回したまま（動）で
出力：step2_attention/runs/precheck_b/（summary.md、summary.json、pareto_<時期>_<乱数>.png）
"""
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
sys.path.insert(0, str(S2.parent))
sys.path.insert(0, str(S2))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from dnaless_curves import curves  # noqa: E402
from model import N_HIST, AttentionCell, Physio2, init_state, irreducible, ls_solution, nll  # noqa: E402
from protocol import PAIRS, compare, envelope, exact_w, run_period  # noqa: E402

NOISE = 0.02
SIGMA_MIN = 0.02
K_AHEAD = 4
EVAL_SEEDS = (1000, 1001)
VIS_WINDOW = 20
CONTROLS = {
    "時計": [{"kind": "clock", "k": k} for k in (2, 4, 10, 30, 100, 300)],
    "サイコロ": [{"kind": "dice", "p": p} for p in (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)],
}


class StateB(NamedTuple):
    w: torch.Tensor      # (B,3,H,W) 係数
    R: torch.Tensor      # (B,6,H,W) 痕跡の2次
    r: torch.Tensor      # (B,3,H,W) 痕跡の1次
    v: torch.Tensor      # (B,1,H,W) 1歩先の誤差の2乗の移動平均
    c: torch.Tensor      # (B,1,H,W) 濃度
    ref: torch.Tensor    # (B,1,H,W) 不応期の残り
    hist: torch.Tensor   # (B,N_HIST,H,W) 最近の値
    mu: torch.Tensor     # (B,1,H,W) 1歩先の予測
    buf: torch.Tensor    # (B,K,H,W) 覚えている4歩先の予測。buf[:,0] が次のステップの値の予測（いちばん古い）
    v4: torch.Tensor     # (B,1,H,W) 4歩先の誤差の2乗の移動平均


def ahead(w, x0, x1, k):
    """係数を k 回まわした k 歩先の予測（x0 = s(t)、x1 = s(t−1)）"""
    a, b = x0, x1
    for _ in range(k):
        a, b = w[:, 0:1] * a + w[:, 1:2] * b + w[:, 2:3], a
    return a


def noise_gain(P, k):
    """学び終えた係数を k 回まわしたとき、入力のノイズ n(t)、n(t−1) に掛かる係数 (α_k, β_k)"""
    w1, w2, _ = exact_w(P)
    al = [0.0, 1.0]      # α_{−1}, α_0
    be = [1.0, 0.0]      # β_{−1}, β_0
    for _ in range(k):
        al.append(w1 * al[-1] + w2 * al[-2])
        be.append(w1 * be[-1] + w2 * be[-2])
    return al[-1], be[-1]


def step_b(s, sense, surprise="z4", frozen=False):
    """DNA なしの1ステップ（model.AttentionCell.step の DNA なし・元の引き金と同じ式）。
    surprise="z4" は B（4歩先の驚き）、"z1" は元の設計（1歩先の驚き）。frozen=True は静的な見積もり用（書き換えず、σ・σ₄ も止める）"""
    P = Physio2
    f1, f2 = s.hist[:, 0:1], s.hist[:, 1:2]
    e = sense - s.mu
    sigma = torch.clamp(s.v, min=P.sigma_min ** 2).sqrt()
    z2 = e ** 2 / sigma ** 2
    e4 = sense - s.buf[:, 0:1]
    sigma4 = torch.clamp(s.v4, min=P.sigma_min ** 2).sqrt()
    z4 = e4 ** 2 / sigma4 ** 2

    one = torch.ones_like(f1)
    R = P.gamma_R * s.R + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
    r = P.gamma_R * s.r + sense * torch.cat([f1, f2, one], 1)
    c = P.c_decay * s.c + (1 - P.c_decay) * (z4 if surprise == "z4" else z2)
    a = 1.0 / P.sigma_window
    v = (1 - a) * s.v + a * e ** 2
    v4 = (1 - a) * s.v4 + a * e4 ** 2
    can = s.ref <= 0
    trig = (c > P.theta) & can
    if frozen:
        trig = torch.zeros_like(trig)
        v, v4 = s.v, s.v4
    hard = trig
    c = torch.where(trig, torch.zeros_like(c), c)
    ref = torch.where(trig, torch.full_like(s.ref, P.refractory), (s.ref - 1).clamp(min=0))

    hist = torch.cat([sense, s.hist[:, :-1]], 1)
    w_ls = ls_solution(R, r, P.eps)
    diff = w_ls - s.w
    B_, _, H, Wd = diff.shape
    eta = torch.ones(B_, 1, H, Wd, device=diff.device)          # DNA なし（model.gates と同じ形）
    g = torch.ones(B_, 1, H, Wd, device=diff.device)
    dw = torch.zeros_like(diff)
    wf = hard.float()
    w_new = s.w + wf * (g * s.w + torch.tanh(eta * diff + dw) - s.w)
    mu = (w_new[:, 0:1] * sense + w_new[:, 1:2] * s.hist[:, 0:1] + w_new[:, 2:3])
    m4 = ahead(w_new, sense, s.hist[:, 0:1], K_AHEAD)
    buf = torch.cat([s.buf[:, 1:], m4], 1)
    new = StateB(w_new, R, r, v, c, ref, hist, mu, buf, v4)
    return new, dict(e=e, sigma=sigma, z2=z2, z4=z4, hard=hard)


@torch.no_grad()
def run_period_b(surprise, wcfg, period, eval_seed, device, pairs=PAIRS, n_per=4, T_grow=400, T_adult=1000,
                 change_at=400, relearn=300, frozen=False):
    """protocol.run_period と同じ世界・同じ乱数の並び・同じ指標。B の状態（覚えている4歩先の予測、σ₄）を足したもの"""
    B = n_per * len(pairs)
    P1 = torch.tensor([a for a, b in pairs for _ in range(n_per)], dtype=torch.float64)
    P2 = torch.tensor([b for a, b in pairs for _ in range(n_per)], dtype=torch.float64)
    g = torch.Generator(device=device).manual_seed(eval_seed)
    ws = W.reset(wcfg, P1.to(device), n=B, generator=g, device=device)
    H = Wd = wcfg.size
    zone = W.zone_mask(wcfg, device)
    wave = ~zone
    s0 = init_state(B, H, Wd, device)
    st = StateB(s0.w, s0.R, s0.r, s0.v, s0.c, s0.ref, s0.hist, s0.mu,
                torch.zeros(B, K_AHEAD, H, Wd, device=device), s0.v.clone())
    if period == "adult":                                   # 学び終えた状態（run_period と同じ始め方）＋ B の分
        frames = []
        for _ in range(N_HIST):
            frames.append(W.sense(ws, wcfg, g))
            ws = W.advance(ws)
        hist = torch.cat(frames[::-1], 1)                   # [s(t), s(t−1), s(t−2), s(t−3)]
        wv = torch.tensor([exact_w(float(p)) for p in P1], device=device, dtype=torch.float32).view(B, 3, 1, 1)
        wv = wv.expand(B, 3, H, Wd).clone()
        irr = torch.tensor([irreducible(wcfg.noise, float(p)) for p in P1], device=device, dtype=torch.float32)
        v = (irr ** 2).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
        mu = wv[:, 0:1] * hist[:, 0:1] + wv[:, 1:2] * hist[:, 1:2] + wv[:, 2:3]
        # σ₄ は学び終えた予測器での4歩先の外れの定常の大きさ σ_n²(1 ＋ α₄² ＋ β₄²)
        g4 = torch.tensor([1 + sum(x * x for x in noise_gain(float(p), K_AHEAD)) for p in P1], device=device, dtype=torch.float32)
        v4 = (wcfg.noise ** 2 * g4).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
        # 覚えている4歩先の予測：t−3・t−2・t−1・t に出したもの（いちばん古いものは s(t−4) がないので、t−2 に出した3歩先で代える）
        h0, h1, h2, h3 = (hist[:, i:i + 1] for i in range(4))
        buf = torch.cat([ahead(wv, h2, h3, K_AHEAD - 1), ahead(wv, h2, h3, K_AHEAD),
                         ahead(wv, h1, h2, K_AHEAD), ahead(wv, h0, h1, K_AHEAD)], 1)
        st = st._replace(w=wv, v=v, hist=hist, mu=mu, v4=v4, buf=buf)
    T = T_grow if period == "grow" else T_adult
    NL = torch.zeros(T, B, H, Wd, device=device)
    FI = torch.zeros(T, B, H, Wd, dtype=torch.bool, device=device)
    SG = torch.zeros(T, B, H, Wd, device=device)
    E2 = torch.zeros(T, B, H, Wd, device=device)
    Z1 = torch.zeros(T, B, H, Wd, device=device)
    Z4 = torch.zeros(T, B, H, Wd, device=device)
    for t in range(T):
        if period == "adult" and t == change_at - 1:
            ws = W.set_period(ws, P2.to(device))
        sense = W.sense(ws, wcfg, g)
        st, rec = step_b(st, sense, surprise, frozen)
        NL[t] = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
        FI[t] = rec["hard"][:, 0]
        SG[t] = rec["sigma"][:, 0]
        E2[t] = rec["e"][:, 0] ** 2
        Z1[t] = rec["z2"][:, 0]
        Z4[t] = rec["z4"][:, 0]
        ws = W.advance(ws)
    cells = B * H * Wd
    out = dict(period=period, surprise=surprise, fires=float(FI.sum()) / cells,
               noise_fire_rate=float(FI[:, :, zone].float().mean()))
    if period == "grow":
        out["quality"] = float(NL[:, :, wave].mean())
        out["wave_fire_rate"] = float(FI[:, :, wave].float().mean())
        out["err_ratio_end"] = float(E2[T - 100:][:, :, wave].mean().sqrt()) / irreducible(wcfg.noise, 12.0)
        return out
    lag = wcfg.lag
    q, react, back, gave = [], [], [], []
    vis = {i: dict(z4=[], z1=[]) for i in range(len(pairs))}
    for x in range(Wd):
        rows = wave[:, x]
        if not bool(rows.any()):
            continue
        a0 = change_at + x * lag
        col = lambda A_: A_[..., x][..., rows]
        if a0 + VIS_WINDOW <= T:
            for i in range(len(pairs)):
                sl = slice(i * n_per, (i + 1) * n_per)
                vis[i]["z4"].append(float(col(Z4[a0:a0 + VIS_WINDOW])[:, sl].mean()))
                vis[i]["z1"].append(float(col(Z1[a0:a0 + VIS_WINDOW])[:, sl].mean()))
        if a0 + relearn > T:
            continue
        q.append(float(col(NL[a0:a0 + relearn]).mean()))
        react.append(float(col(FI[a0:a0 + 10]).any(0).any(-1).float().mean()))
        pre = col(E2[change_at - 200:change_at]).mean(0).sqrt()
        post = col(E2[a0 + relearn - 50:a0 + relearn]).mean(0).sqrt()
        back.append(float((post <= 2 * pre).float().mean()))
        gave.append(float((col(SG[a0 + relearn - 1]) > 2 * col(SG[change_at - 1])).float().mean()))
    out["quality"] = float(np.mean(q))
    out["steady_fire_rate_wave"] = float(FI[200:change_at][:, :, wave].float().mean())
    out["steady_fire_rate_noise"] = float(FI[200:change_at][:, :, zone].float().mean())
    out["steady_nll_wave"] = float(NL[200:change_at][:, :, wave].mean())
    out["react_cols"] = float(np.mean(react))
    out["err_back"] = float(np.mean(back))
    out["gaveup"] = float(np.mean(gave))
    out["visibility"] = {f"{pairs[i][0]:g}→{pairs[i][1]:g}": dict(z4=float(np.mean(v["z4"])), z1=float(np.mean(v["z1"])))
                         for i, v in vis.items()}
    return out


def plot(period, seed, ctrl_pts, b_pt, ref_pt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, name in enumerate(CONTROLS):
        pts = [(p["fires"], p["quality"]) for p in ctrl_pts if p["family"] == name]
        env = envelope(pts)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=SERIES[i + 1], alpha=0.35, linewidths=0)
        ax.plot([p[0] for p in env], [p[1] for p in env], color=SERIES[i + 1], linewidth=2, label=f"{name}（DNA なし、下の包み）")
    ax.scatter([b_pt[0]], [b_pt[1]], s=70, color=SERIES[0], zorder=5, label="B：4歩先の驚き")
    ax.scatter([ref_pt[0]], [ref_pt[1]], s=50, marker="s", color=SERIES[4], zorder=5, label="参考：1歩先の驚き（元の設計）")
    ax.set_xscale("log")
    ax.set_xlabel("発火数（1セルあたり、全セル）", color=INK_2, fontsize=9)
    ax.set_ylabel("NLL（波の区域。低いほど良い）", color=INK_2, fontsize=9)
    ttl = {"grow": "育つ時期（予測器0から、0〜400）", "adult": "大人の時期（学び直し：新しい波が届いてから300ステップ）"}[period]
    ax.set_title(f"{ttl}　ノイズ0.02、評価の乱数{seed}", loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    Physio2.sigma_min = SIGMA_MIN                            # S2-7 を丸ごと戻す（ノイズと σ の下限）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wcfg = W.WorldConfig(noise=NOISE)
    out = S2 / "runs" / "precheck_b"
    out.mkdir(parents=True, exist_ok=True)
    res = dict(world=wcfg.to_dict(), sigma_min=Physio2.sigma_min, k_ahead=K_AHEAD, seeds={})

    # 一致の確かめ：この回し方の元の設計の引き金 ＝ run_period ＋ 本物のセル（DNA なし）
    cell = AttentionCell(dna=False).to(dev).eval()
    cons = {}
    for period in ("grow", "adult"):
        a = run_period(cell, {"kind": "trigger"}, wcfg, period, eval_seed=EVAL_SEEDS[0], device=dev)
        b = run_period_b("z1", wcfg, period, EVAL_SEEDS[0], dev)
        cons[period] = dict(run_period=(a["fires"], a["quality"]), precheck=(b["fires"], b["quality"]),
                            same=abs(a["fires"] - b["fires"]) < 1e-6 and abs(a["quality"] - b["quality"]) < 1e-5)
    res["consistency"] = cons
    print("一致の確かめ：", json.dumps(cons, ensure_ascii=False), flush=True)
    if not all(c["same"] for c in cons.values()):
        (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("一致しない。判定の前に止める（DESIGN.md 13章）")

    pj = {"grow": "育つ時期", "adult": "大人の時期"}
    verdict_all = True
    for seed in EVAL_SEEDS:
        ctrl = curves(wcfg, seed, dev, families=CONTROLS)
        r = dict(controls=ctrl)
        for period in ("grow", "adult"):
            r[f"B_{period}"] = run_period_b("z4", wcfg, period, seed, dev)
            r[f"ref_{period}"] = run_period_b("z1", wcfg, period, seed, dev)
        r["static_adult"] = run_period_b("z4", wcfg, "adult", seed, dev, frozen=True)
        verd = {}
        for period in ("grow", "adult"):
            for name in CONTROLS:
                env = envelope([(p["fires"], p["quality"]) for p in ctrl if p["period"] == period and p["family"] == name])
                for who in ("B", "ref"):
                    pt = r[f"{who}_{period}"]
                    v, qc = compare((pt["fires"], pt["quality"]), env)
                    verd[f"{who}_{period}_{name}"] = dict(verdict=v, fires=pt["fires"], nll=pt["quality"], control_nll=qc)
            plot(period, seed, [p for p in ctrl if p["period"] == period],
                 (r[f"B_{period}"]["fires"], r[f"B_{period}"]["quality"]),
                 (r[f"ref_{period}"]["fires"], r[f"ref_{period}"]["quality"]), out / f"pareto_{period}_{seed}.png")
        r["verdicts"] = verd
        ok = all(verd[f"B_{p}_{n}"]["verdict"] == "上回る" for p in ("grow", "adult") for n in CONTROLS)
        r["promising"] = ok
        verdict_all &= ok
        res["seeds"][str(seed)] = r
        print(f"評価の乱数 {seed}：済み（{'見込みあり' if ok else '見込みなし'}）", flush=True)
    res["final"] = "見込みあり" if verdict_all else "見込みなし"
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 候補B（4歩先の驚き）の実装前の模擬", "",
             f"世界のノイズ {NOISE}、σ の下限 {SIGMA_MIN}、4歩先、DNA なし。合否は DESIGN.md 13章（走らせる前に決めた）。", "",
             "一致の確かめ（この回し方の1歩先の引き金 ＝ run_period ＋ 本物のセル）："
             + "、".join(f"{pj[k]} {'一致' if v['same'] else '不一致'}（発火 {v['precheck'][0]:.3f}、NLL {v['precheck'][1]:.4f}）" for k, v in cons.items()), "",
             f"**事前に決めた合否：{res['final']}**", "",
             "## 同じ発火数での比べ（7.5 の compare）", "",
             "| 評価の乱数 | 時期 | 驚き | 発火数（1セル） | NLL | 時計（同じ発火数） | サイコロ（同じ発火数） |", "|---|---|---|---:|---:|---|---|"]
    for seed, r in res["seeds"].items():
        for period in ("grow", "adult"):
            for who, jp in (("B", "B：4歩先"), ("ref", "参考：1歩先（元の設計）")):
                c1, c2 = r["verdicts"][f"{who}_{period}_時計"], r["verdicts"][f"{who}_{period}_サイコロ"]
                lines.append(f"| {seed} | {pj[period]} | {jp} | {c1['fires']:.2f} | {c1['nll']:.3f} | "
                             f"{c1['verdict']}（{c1['control_nll']:.3f}） | {c2['verdict']}（{c2['control_nll']:.3f}） |")
    lines += ["", "## 大人の時期の中身", "",
              "| 評価の乱数 | 驚き | 落ち着いた発火率（波／ノイズ） | 10歩以内の反応 | 誤差が戻った | あきらめ |", "|---|---|---|---:|---:|---:|"]
    for seed, r in res["seeds"].items():
        for who, jp in (("B", "B：4歩先"), ("ref", "参考：1歩先")):
            a = r[f"{who}_adult"]
            lines.append(f"| {seed} | {jp} | {a['steady_fire_rate_wave']:.4f}／{a['steady_fire_rate_noise']:.4f} | {a['react_cols']:.2f} | "
                         f"{a['err_back']:.2f} | {a['gaveup']:.2f} |")
    lines += ["", f"## 見える度合い（新しい波が届いてから{VIS_WINDOW}ステップの z² の平均、波の区域）", "",
              "静＝学び終えた予測器と σ・σ₄ を止めたまま、動＝B を回したまま。REVIEW.md §5 の表は窓が記録に残っていないので、数字は一致しなくてよい", "",
              "| 評価の乱数 | 組 | 4歩先 z₄²（静） | 4歩先 z₄²（動） | 1歩先 z²（静） | 1歩先 z²（動、B を回したまま） |", "|---|---|---:|---:|---:|---:|"]
    for seed, r in res["seeds"].items():
        for pair, st in r["static_adult"]["visibility"].items():
            dy = r["B_adult"]["visibility"][pair]
            lines.append(f"| {seed} | {pair} | {st['z4']:.1f} | {dy['z4']:.1f} | {st['z1']:.1f} | {dy['z1']:.1f} |")
    lines += ["", "図：`pareto_grow_<乱数>.png`、`pareto_adult_<乱数>.png`", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
