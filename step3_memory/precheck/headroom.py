"""
Step 3 の実装前の模擬：この世界に、記憶が役に立つ余地はあるか（step3_memory/DESIGN.md 7章・7.1。読み方は走らせる前に書いた）。

    .venv/Scripts/python.exe step3_memory/precheck/headroom.py [--device cuda] [--smoke 1]

凍結したレシピ（Step 2 の DNA なしの B。タグ step2-recipe-frozen）の引き金はそのままに、発火したときに書くものだけを変えて比べる。
Step 2 のコードは書き換えず、AttentionCell.step を呼んだあとで、発火したセルの係数を差し替える（予測 μ と覚えている4歩先の予測も出し直す）：
  床  ：凍結したレシピそのもの（w ← w ＋ tanh(w_ls − w)）
  天井：波の区域のセルが発火したら、その列がいま見ている周期の正確な係数を、そのまま書く（ノイズの区域は床と同じ）
  M1  ：保存先2つに、その組の A と B の正確な係数をはじめから入れておき（C は入れない）、発火したら最近の痕跡の上で
        J(u) = S − 2uᵀr ＋ uᵀRu を比べて、J(いちばん合う保存先) − J(w_ls) ≤ 6σ² ならその係数をそのまま書く。そうでなければ床と同じ
        （S は s² の痕跡、減衰0.9。安い式なので毎ステップの側。σ² はそのセルのいまの v）

世界：標準の世界（ノイズ0.02）と、ノイズだけ0.04 の世界（σ の下限は0.02 のまま）。学び終えた大人の状態（A）から始め、
滞在600ステップで A, B, A, B, A, C, A, B。組は (A, B, C) = (12, 24, 8) と (12, 20, 8)、各組に世界の実現8つ。
センサーの値は、どの回し方でも同じ乱数の並びで世界だけが決める（組にした比べ）。評価の乱数 300・301。

測るもの（切り替えごと、波の区域。窓は新しい波がその列に届いてから300ステップ）：
  発火数（その時期の1セルあたり）、窓の NLL（列で平均）、予測が当たるまでのステップ、窓の終わりの学び残し、
  M1 の思い出しの記録（発火のうち保存先を書いた割合、正しかった割合、C の時期の取り違え）
出力：step3_memory/runs/precheck_headroom/（summary.md、summary.json）。試運転は runs/smoke_precheck_headroom/
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "step2_attention"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from model import N_HIST, AttentionCell, Physio2, ahead, init_state, irreducible, ls_solution, nll  # noqa: E402
from protocol import exact_w, noise_gain  # noqa: E402
from sealed import check_dev_seed  # noqa: E402

DWELL = 600
SEQ = ("A", "B", "A", "B", "A", "C", "A", "B")
GROUPS = ((12.0, 24.0, 8.0), (12.0, 20.0, 8.0))
N_PER = 8
NOISES = (0.02, 0.04)
EVAL_SEEDS = (300, 301)
WINDOW = 300
END = (250, 300)            # 学び残しを測る区間（届いてからのステップ）
TAU = 6.0                   # M1 の思い出しの基準（J の差 ≤ TAU·σ²）
ACC = 1.5                   # 「予測が当たる」：列の平均の誤差の2乗の10ステップ平均 ≤ ACC × 当てられない誤差の2乗
VARIANTS = ("床", "天井", "M1")


def exact_w_t(P):
    """(…) の周期のテンソル → (…, 3) の正確な係数 [2c, −1, 1−c]（protocol.exact_w と同じ式）"""
    c = torch.cos(2 * math.pi / P)
    return torch.stack([2 * c, -torch.ones_like(c), 1 - c], -1)


def quad(R6, u):
    """uᵀRu（R は上三角の6つ：00,01,02,11,12,22）。u は (B,3,H,W)"""
    a, b, c, d, e, f = R6.unbind(1)
    u0, u1, u2 = u.unbind(1)
    return (u0 * u0 * a + 2 * u0 * u1 * b + 2 * u0 * u2 * c + u1 * u1 * d + 2 * u1 * u2 * e + u2 * u2 * f).unsqueeze(1)


def fit(S, R6, r, u):
    """最近の痕跡の上での当てはまり J(u) = S − 2uᵀr ＋ uᵀRu（(B,1,H,W)）"""
    return S - 2 * (u * r).sum(1, keepdim=True) + quad(R6, u)


def adult_init(wcfg, P1, g, dev):
    """学び終えた大人の状態（step2_attention/protocol.run_period の大人の時期と同じ始め方）。P1 は (B,) float64"""
    B = P1.shape[0]
    H = Wd = wcfg.size
    ws = W.reset(wcfg, P1.to(dev), n=B, generator=g, device=dev)
    st = init_state(B, H, Wd, dev)
    frames = []
    for _ in range(N_HIST):
        frames.append(W.sense(ws, wcfg, g))
        ws = W.advance(ws)
    hist = torch.cat(frames[::-1], 1)
    wv = torch.tensor([exact_w(float(p)) for p in P1], device=dev, dtype=torch.float32).view(B, 3, 1, 1).expand(B, 3, H, Wd).clone()
    irr = torch.tensor([irreducible(wcfg.noise, float(p)) for p in P1], device=dev, dtype=torch.float32)
    v = (irr ** 2).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
    mu = wv[:, 0:1] * hist[:, 0:1] + wv[:, 1:2] * hist[:, 1:2] + wv[:, 2:3]
    k = st.buf.shape[1]
    gk = torch.tensor([1 + sum(x * x for x in noise_gain(float(p), k)) for p in P1], device=dev, dtype=torch.float32)
    vk = (wcfg.noise ** 2 * gk).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
    hs = [hist[:, i:i + 1] for i in range(N_HIST)]
    slots = []
    for i in range(k):
        j = k - 1 - i
        slots.append(ahead(wv, hs[j], hs[j + 1], k) if j + 1 <= N_HIST - 1 else ahead(wv, hs[N_HIST - 2], hs[N_HIST - 1], i + N_HIST - 1))
    st = st._replace(w=wv, v=v, hist=hist, mu=mu, vk=vk, buf=torch.cat(slots, 1))
    return st, ws


def rewrite(st, w_new, mask):
    """mask のセルの係数を w_new にして、予測 μ と、いま覚えた4歩先の予測（buf の最後）を出し直す"""
    w = torch.where(mask, w_new, st.w)
    x0, x1 = st.hist[:, 0:1], st.hist[:, 1:2]                 # s(t)、s(t−1)（step のあとの状態）
    mu = w[:, 0:1] * x0 + w[:, 1:2] * x1 + w[:, 2:3]
    k = st.buf.shape[1]
    buf = torch.cat([st.buf[:, :-1], ahead(w, x0, x1, k)], 1)
    return st._replace(w=w, mu=mu, buf=buf)


@torch.no_grad()
def run(variant, wcfg, seed, dev):
    check_dev_seed(seed, "評価の乱数")
    T = len(SEQ) * DWELL
    B = N_PER * len(GROUPS)
    H = Wd = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in GROUPS for _ in range(N_PER)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in SEQ]
    n_sw = len(SEQ) - 1
    g = torch.Generator(device=dev).manual_seed(seed)
    st, ws = adult_init(wcfg, sched[0], g, dev)
    cell = AttentionCell(dna=False).to(dev)
    zone = W.zone_mask(wcfg, dev)
    wave = ~zone
    S = torch.zeros(B, 1, H, Wd, device=dev)
    slot_P = torch.stack([per["A"], per["B"]], 1).to(dev)                             # (B,2) 保存先の周期
    slot_w = exact_w_t(slot_P).float().permute(0, 2, 1)                               # (B,3,2)
    lag = wcfg.lag
    xs = torch.arange(Wd, device=dev)
    # 切り替えごとの集め物（セルごと）
    nll_sum = torch.zeros(n_sw, B, H, Wd, device=dev)
    num_end = torch.zeros_like(nll_sum)
    den_end = torch.zeros_like(nll_sum)
    fires = torch.zeros_like(nll_sum)
    E2col = torch.zeros(T, B, Wd, device=dev)                                         # 列の平均の誤差の2乗（波の行）
    wave_rows = wave.float()                                                          # (H,W)
    rows_per_col = wave_rows.sum(0)                                                   # (W,)
    rec_m1 = np.zeros((len(SEQ), 3))                                                  # 時期ごと：波の区域の発火、保存先を書いた、正しい保存先
    fires_total = torch.zeros(B, H, Wd, device=dev)
    t0 = time.time()
    for t in range(T):
        j = t // DWELL
        if t % DWELL == DWELL - 1 and j + 1 < len(SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)                                              # (B,W) いま見ている周期
        # 学び残しの材料（step の前の入力で、古い・新しい周期の正確な予測器）
        if j >= 1:
            x1, x2 = st.hist[:, 0:1], st.hist[:, 1:2]
            wo = exact_w_t(sched[j - 1].to(dev)).float().view(B, 3, 1, 1)
            wn = exact_w_t(sched[j].to(dev)).float().view(B, 3, 1, 1)
            m_old = wo[:, 0:1] * x1 + wo[:, 1:2] * x2 + wo[:, 2:3]
            m_new = wn[:, 0:1] * x1 + wn[:, 1:2] * x2 + wn[:, 2:3]
            num = (st.mu - m_new) ** 2
            den = (m_old - m_new) ** 2
        st, rec = cell.step(st, sense, policy={"kind": "trigger"})
        S = Physio2.gamma_R * S + sense ** 2
        fired = rec["hard"]                                                           # (B,1,H,W)
        if variant == "天井":
            w_star = exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, H, Wd)
            st = rewrite(st, w_star, fired & wave)
        elif variant == "M1":
            w_ls = ls_solution(st.R, st.r, Physio2.eps)
            J_ls = fit(S, st.R, st.r, w_ls)
            Js = torch.cat([fit(S, st.R, st.r, slot_w[:, :, i].view(B, 3, 1, 1).expand(B, 3, H, Wd)) for i in range(2)], 1)   # (B,2,H,W)
            J_best, i_best = Js.min(1, keepdim=True)
            sig2 = st.v.clamp(min=Physio2.sigma_min ** 2)
            accept = fired & (J_best - J_ls <= TAU * sig2)
            w_best = torch.gather(slot_w.unsqueeze(-1).unsqueeze(-1).expand(B, 3, 2, H, Wd), 2,
                                  i_best.unsqueeze(1).expand(B, 3, 1, H, Wd)).squeeze(2)
            st = rewrite(st, w_best, accept)
            P_best = torch.gather(slot_P.float().view(B, 2, 1, 1).expand(B, 2, H, Wd), 1, i_best)
            correct = accept & (P_best == colP.float().view(B, 1, 1, Wd))
            fw = fired & wave
            rec_m1[j] += (float(fw.sum()), float((accept & wave).sum()), float((correct & wave).sum()))
        # 記録（NLL は step の中の e と σ で。天井・M1 の差し替えは次のステップの予測から効く）
        NL = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]                   # (B,H,W)
        e2 = rec["e"][:, 0] ** 2
        E2col[t] = (e2 * wave_rows).sum(1) / rows_per_col
        fires_total += fired[:, 0].float()
        if j >= 1:
            sw = j - 1
            fires[sw] += fired[:, 0].float()
            rel = t - (j * DWELL + xs * lag)                                          # (W,) 届いてからのステップ
            in_win = ((rel >= 0) & (rel < WINDOW)).float().view(1, 1, Wd)
            in_end = ((rel >= END[0]) & (rel < END[1])).float().view(1, 1, Wd)
            nll_sum[sw] += NL * in_win
            num_end[sw] += num[:, 0] * in_end
            den_end[sw] += den[:, 0] * in_end
        ws = W.advance(ws)
    elapsed = time.time() - t0
    return dict(nll_sum=nll_sum, num_end=num_end, den_end=den_end, fires=fires, E2col=E2col, rec_m1=rec_m1,
                wave=wave, rows_per_col=rows_per_col, sched=sched, elapsed=elapsed, fires_total=fires_total)


def switch_names():
    out, seen = [], {}
    for j in range(1, len(SEQ)):
        a, b = SEQ[j - 1], SEQ[j]
        key = f"{a}→{b}"
        seen[key] = seen.get(key, 0) + 1
        novel = b not in SEQ[:j]
        out.append(dict(name=f"{key}（{seen[key]}回目）", key=key, nth=seen[key], novel=novel))
    return out


def summarize(res, wcfg):
    """組ごと・切り替えごとの数字"""
    wave, rpc = res["wave"], res["rows_per_col"]
    names = switch_names()
    out = []
    for gi, grp in enumerate(GROUPS):
        sl = slice(gi * N_PER, (gi + 1) * N_PER)
        rows = []
        for sw, nm in enumerate(names):
            P_new = float(res["sched"][sw + 1][sl][0])
            irr2 = irreducible(wcfg.noise, P_new) ** 2
            # 質：列ごとに（波の行・世界・300ステップで）平均して、列で平均
            q_col = (res["nll_sum"][sw][sl] * wave).sum((0, 1)) / (rpc * N_PER * WINDOW)
            quality = float(q_col.mean())
            fires = float((res["fires"][sw][sl] * wave).sum() / (wave.sum() * N_PER))
            ratio = (res["num_end"][sw][sl] / res["den_end"][sw][sl].clamp(min=1e-12))[:, wave]
            resid = float(ratio.median())
            # 予測が当たるまで（列ごと、組の世界で平均した誤差の2乗の10ステップ平均）
            taus = []
            for x in range(wcfg.size):
                a0 = (sw + 1) * DWELL + x * wcfg.lag
                tr = res["E2col"][a0:a0 + WINDOW, sl, x].mean(1)
                sm = torch.nn.functional.avg_pool1d(tr.view(1, 1, -1), 10, 1).view(-1)
                hit = (sm <= ACC * irr2).nonzero()
                taus.append(int(hit[0]) if len(hit) else WINDOW)
            rows.append(dict(switch=nm["name"], key=nm["key"], nth=nm["nth"], novel=nm["novel"], P_new=P_new,
                             fires=fires, quality=quality, t_acc=float(np.mean(taus)), resid_end=resid))
        out.append(dict(group=f"A={grp[0]:g}, B={grp[1]:g}, C={grp[2]:g}", rows=rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    assert Physio2.surprise_ahead == 4 and Physio2.sigma_min == 0.02 and Physio2.theta == 4.0, "凍結したレシピの値と違う"
    global DWELL
    name = "precheck_headroom"
    noises, seeds = NOISES, EVAL_SEEDS
    if args.smoke:
        DWELL = 160
        noises, seeds = (0.02,), (300,)
        name = "smoke_" + name
    out_dir = ROOT / "step3_memory" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    allres = {}
    for noise in noises:
        wcfg = W.WorldConfig(noise=noise)
        for seed in seeds:
            for var in VARIANTS:
                res = run(var, wcfg, seed, args.device)
                s = summarize(res, wcfg)
                m1 = res["rec_m1"]
                allres[f"{noise}|{seed}|{var}"] = dict(groups=s, m1=m1.tolist(), elapsed=res["elapsed"],
                                                       fires_total=float((res["fires_total"] * res["wave"]).sum() / (res["wave"].sum() * res["fires_total"].shape[0])))
                print(f"noise={noise} seed={seed} {var}: {res['elapsed']:.1f}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    report(allres, noises, seeds, out_dir)


def report(allres, noises, seeds, out_dir):
    L = ["# Step 3 の実装前の模擬：記憶が役に立つ余地（DESIGN.md 7.1）", "",
         f"滞在{DWELL}ステップ、並び {', '.join(SEQ)}。各組に世界の実現{N_PER}つ。窓は新しい波が届いてから{WINDOW}ステップ。",
         "差は切り替え1回あたりのナット（NLL の差×300。正なら床より良い）。取り分＝(床 − M1) ÷ (床 − 天井)", ""]
    verdict = {}
    for noise in noises:
        for seed in seeds:
            fl = allres[f"{noise}|{seed}|床"]["groups"]
            ce = allres[f"{noise}|{seed}|天井"]["groups"]
            m1 = allres[f"{noise}|{seed}|M1"]["groups"]
            for gi in range(len(fl)):
                L += [f"## ノイズ{noise}、評価の乱数{seed}、{fl[gi]['group']}", "",
                      "| 切り替え | 床：発火 | 床：NLL | 天井：発火 | 天井：NLL | M1：発火 | M1：NLL | 余地（天井−床、ナット/回） | M1 の得（ナット/回） | 取り分 | 当たるまで 床／天井／M1 | 学び残し 床／天井／M1 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
                for rf, rc, rm in zip(fl[gi]["rows"], ce[gi]["rows"], m1[gi]["rows"]):
                    head = (rf["quality"] - rc["quality"]) * WINDOW
                    gain = (rf["quality"] - rm["quality"]) * WINDOW
                    share = gain / head if abs(head) > 1e-9 else float("nan")
                    L.append(f"| {rf['switch']}{'（初めて）' if rf['novel'] else ''} | {rf['fires']:.2f} | {rf['quality']:.3f} | {rc['fires']:.2f} | {rc['quality']:.3f} | "
                             f"{rm['fires']:.2f} | {rm['quality']:.3f} | {head:.1f} | {gain:.1f} | {share:.2f} | "
                             f"{rf['t_acc']:.0f}／{rc['t_acc']:.0f}／{rm['t_acc']:.0f} | {rf['resid_end']:.3f}／{rc['resid_end']:.3f}／{rm['resid_end']:.3f} |")
                    if not rf["novel"]:
                        verdict.setdefault(noise, []).append(head)
                L.append("")
            rm1 = np.array(allres[f"{noise}|{seed}|M1"]["m1"])
            L += [f"M1 の思い出し（ノイズ{noise}、乱数{seed}、2つの組の合計。時期ごと：波の区域の発火、保存先を書いた割合、そのうち正しかった割合）", "",
                  "| 時期 | 周期 | 発火 | 保存先を書いた | 正しかった | 取り違え（発火に対する割合） |", "|---|---|---:|---:|---:|---:|"]
            for j, p in enumerate(SEQ):
                f_, a_, c_ = rm1[j]
                L.append(f"| {j} | {p} | {int(f_)} | {a_ / max(f_, 1):.2f} | {c_ / max(a_, 1):.2f} | {(a_ - c_) / max(f_, 1):.3f} |")
            L.append("")
    L += ["## 読み方（DESIGN.md 7.1、走らせる前に決めた）", ""]
    for noise, heads in verdict.items():
        mn = min(heads)
        v = "余地が大きい" if mn >= 10 else ("余地が小さい" if mn < 2 else "中間（そうの判断）")
        L.append(f"- ノイズ{noise}：戻りの切り替えの余地の最小 {mn:.1f} ナット/回 → **{v}**")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
