"""
Step 3 の睡眠の段の前の模擬：睡眠のどの部分が慣れを直すのか（step3_memory/DESIGN.md 9章。案と読み方は走らせる前に書いた。
2026-09-30、そうの判断：① σ₄ だけ戻す、② 係数だけ直す、③ 両方、に分けて測る）。

    .venv/Scripts/python.exe step3_memory/precheck/sleep.py [--device cuda] [--smoke 1]

凍結したレシピ（Step 2 の DNA なしの B）はそのまま。Step 2 のコードは書き換えず、AttentionCell.step のあとで、
静かなセルの見直し（睡眠の書き込み）だけを外から足す：
  見直しの材料：長い痕跡 R_L ← γ_L·R_L ＋ f fᵀ、r_L ← γ_L·r_L ＋ s·f（f = [s(t−1), s(t−2), 1]）。驚いて発火したら0に戻す
  いつ寝るか  ：最後に驚いて発火してから N_q = 100 ステップ発火していなければ、1回だけ見直す（次は、次の驚きの発火のあと）
  ② 係数だけ：w ← w ＋ tanh(w_L − w)（w_L は長い痕跡の最小二乗の解）。σ₄ は触らない
  ① σ₄ だけ ：σ₄² ← σ² × (1 ＋ α₄² ＋ β₄²) ÷ (1 ＋ w₁² ＋ w₂²)（1歩先の σ と自分の係数から見込める4歩先の外れ）。係数は触らない
  ③ 両方    ：② のあと、新しい係数で ①
  神様の版  ：①' σ₄ を、いまの周期の正確な係数での定常の大きさに、②' 係数を正確な係数に、③' 両方（波の区域だけ）
  床・天井  ：headroom.py と同じ（回帰の確かめ：大人から始めた床・天井は、headroom.py と同じ数字になるはず）
始め方は2つ：学び終えた大人の状態（headroom.py と同じ）と、予測器0から A で600ステップ育った状態。
世界・組・並び・評価の乱数は headroom.py と同じ。
出力：step3_memory/runs/precheck_sleep/（summary.md、details.md、summary.json）。試運転は runs/smoke_precheck_sleep/
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import headroom as H  # noqa: E402
from headroom import W, AttentionCell, Physio2, exact_w_t, init_state, irreducible, ls_solution, nll, noise_gain, rewrite  # noqa: E402

N_Q = 100
GAMMAS = (0.9, 0.97, 0.99, 0.995)
G_MAIN = 0.99
STARTS = ("adult", "grow")
REACT_OK, REACT_BAD, RESTLESS = 0.8, 0.3, 0.1


def variants():
    out = [dict(name="床", kind="floor"), dict(name="天井", kind="ceiling"), dict(name="①", kind="sig", g=None)]
    out += [dict(name=f"②(γ={g})", kind="coef", g=g) for g in GAMMAS]
    out += [dict(name=f"③(γ={g})", kind="both", g=g) for g in GAMMAS]
    out += [dict(name="①'", kind="sig_o", g=None), dict(name="②'", kind="coef_o", g=None), dict(name="③'", kind="both_o", g=None)]
    return out


def gain4(w):
    """1 ＋ α₄² ＋ β₄²（係数 w を4回まわしたとき、入力のノイズ n(t)・n(t−1) に掛かる係数。protocol.noise_gain と同じ漸化式）"""
    w1, w2 = w[:, 0:1], w[:, 1:2]
    a_prev, a = torch.zeros_like(w1), torch.ones_like(w1)
    b_prev, b = torch.ones_like(w1), torch.zeros_like(w1)
    for _ in range(4):
        a_prev, a = a, w1 * a + w2 * a_prev
        b_prev, b = b, w1 * b + w2 * b_prev
    return 1 + a * a + b * b


def gain1(w):
    """1 ＋ w₁² ＋ w₂²（1歩先の外れの分散は、入力のノイズの分散の何倍か）"""
    return 1 + w[:, 0:1] ** 2 + w[:, 1:2] ** 2


def sig4_exact(noise, P):
    a, b = noise_gain(P, 4)
    return max(noise * (1 + a * a + b * b) ** 0.5, Physio2.sigma_min)


@torch.no_grad()
def run(var, wcfg, seed, dev, start):
    H.check_dev_seed(seed, "評価の乱数")
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
    kind = var["kind"]
    sleeps_kind = kind in ("sig", "coef", "both", "sig_o", "coef_o", "both_o")
    uses_long = kind in ("coef", "both")
    RL = torch.zeros(B, 6, S_, S_, device=dev)
    rL = torch.zeros(B, 3, S_, S_, device=dev)
    q = torch.zeros(B, 1, S_, S_, device=dev)
    nk = torch.zeros(B, 1, S_, S_, device=dev)                                         # doubling：最後の驚きの発火からの見直しの回数（9.7）
    xs = torch.arange(S_, device=dev)
    lag = wcfg.lag
    Z = lambda dt=torch.float32: torch.zeros(n_sw, B, S_, S_, device=dev, dtype=dt)
    nll_sum, num_end, den_end, fires_ph, sleeps_ph, out_fires, sig_pre = Z(), Z(), Z(), Z(), Z(), Z(), Z()
    react = Z(torch.bool)
    fires_tot = torch.zeros(B, S_, S_, device=dev)
    sleeps_tot = torch.zeros(B, S_, S_, device=dev)
    t0 = time.time()
    for t in range(T):
        j = t // H.DWELL
        if t % H.DWELL == H.DWELL - 1 and j + 1 < len(H.SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        if j >= 1:
            x1, x2 = st.hist[:, 0:1], st.hist[:, 1:2]
            wo = exact_w_t(sched[j - 1].to(dev)).float().view(B, 3, 1, 1)
            wn = exact_w_t(sched[j].to(dev)).float().view(B, 3, 1, 1)
            m_old = wo[:, 0:1] * x1 + wo[:, 1:2] * x2 + wo[:, 2:3]
            m_new = wn[:, 0:1] * x1 + wn[:, 1:2] * x2 + wn[:, 2:3]
            num = (st.mu - m_new) ** 2
            den = (m_old - m_new) ** 2
        st, rec = cell.step(st, sense, policy={"kind": "trigger"})
        fired = rec["hard"]                                                            # 驚きの発火 (B,1,H,W)
        if kind == "ceiling":
            w_star = exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, S_, S_)
            st = rewrite(st, w_star, fired & wave)
        sleep_m = torch.zeros_like(fired)
        if sleeps_kind:
            if uses_long:                                                              # 長い痕跡（毎ステップの安い足し算）。驚いて発火したら0に戻す
                f1, f2 = st.hist[:, 1:2], st.hist[:, 2:3]
                one = torch.ones_like(f1)
                RL = var["g"] * RL + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
                rL = var["g"] * rL + sense * torch.cat([f1, f2, one], 1)
                RL = torch.where(fired, torch.zeros_like(RL), RL)
                rL = torch.where(fired, torch.zeros_like(rL), rL)
            q = torch.where(fired, torch.zeros_like(q), q + 1)
            if var.get("doubling"):                                                    # 9.7 の (は)：間隔を N_Q、2N_Q、4N_Q…と伸ばす（最後の驚きから N_Q、3N_Q、7N_Q…）
                nk = torch.where(fired, torch.zeros_like(nk), nk)
                sleep_m = q == N_Q * (torch.pow(2.0, nk + 1) - 1)
            else:
                sleep_m = ((q > 0) & (torch.remainder(q, N_Q) == 0)) if var.get("periodic") else (q == N_Q)   # periodic：静かな間は N_Q ごと（9.6 の追加の診断）
            if kind.endswith("_o"):
                sleep_m = sleep_m & wave
            if bool(sleep_m.any()):
                if kind in ("coef", "both"):
                    wL = ls_solution(RL, rL, Physio2.eps)
                    st = rewrite(st, st.w + torch.tanh(wL - st.w), sleep_m)
                if kind in ("coef_o", "both_o"):
                    st = rewrite(st, exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, S_, S_), sleep_m)
                if kind in ("sig", "both"):
                    v1 = st.v.clamp(min=Physio2.sigma_min ** 2)
                    vk_new = (v1 * gain4(st.w) / gain1(st.w)).clamp(min=Physio2.sigma_min ** 2)
                    st = st._replace(vk=torch.where(sleep_m, vk_new, st.vk))
                if kind in ("sig_o", "both_o"):
                    we = exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2)                      # (B,3,1,W)
                    vk_ex = (wcfg.noise ** 2 * gain4(we)).clamp(min=Physio2.sigma_min ** 2).expand(B, 1, S_, S_)
                    st = st._replace(vk=torch.where(sleep_m, vk_ex, st.vk))
        if var.get("doubling"):
            nk = nk + sleep_m.float()
        # 記録
        F = fired[:, 0]
        Sm = sleep_m[:, 0]
        fires_tot += F.float()
        sleeps_tot += Sm.float()
        sk = st.vk.clamp(min=Physio2.sigma_min ** 2).sqrt()[:, 0]
        if j + 1 < len(H.SEQ) and t >= (j + 1) * H.DWELL - 50:                             # 次の切り替えの直前50ステップの σ₄
            sig_pre[j] += sk / 50
        if j >= 1:
            sw = j - 1
            NL = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
            rel = (t - (j * H.DWELL + xs * lag)).view(1, 1, S_)
            in_win = (rel >= 0) & (rel < H.WINDOW)
            in_end = (rel >= H.END[0]) & (rel < H.END[1])
            nll_sum[sw] += NL * in_win.float()
            num_end[sw] += num[:, 0] * in_end.float()
            den_end[sw] += den[:, 0] * in_end.float()
            fires_ph[sw] += F.float()
            sleeps_ph[sw] += Sm.float()
            react[sw] |= F & in_win
            out_fires[sw] += (F & ~in_win).float()
        ws = W.advance(ws)
    return dict(nll_sum=nll_sum, num_end=num_end, den_end=den_end, fires_ph=fires_ph, sleeps_ph=sleeps_ph, out_fires=out_fires,
                react=react, sig_pre=sig_pre, fires_tot=fires_tot, sleeps_tot=sleeps_tot, sched=sched, wave=wave, elapsed=time.time() - t0,
                final=dict(core=st, RL=RL, rL=rL, q=q, k=nk))                  # 最後の状態（check_cell.py の回帰の確かめで比べるため。計算は変えていない）


def summarize(res, wcfg):
    wave = res["wave"]
    rpc = wave.float().sum(0)
    out = []
    for gi, grp in enumerate(H.GROUPS):
        sl = slice(gi * H.N_PER, (gi + 1) * H.N_PER)
        rows = []
        for sw in range(len(H.SEQ) - 1):
            P_old = float(res["sched"][sw][sl][0])
            P_new = float(res["sched"][sw + 1][sl][0])
            q_col = (res["nll_sum"][sw][sl] * wave).sum((0, 1)) / (rpc * H.N_PER * H.WINDOW)
            ratio = (res["num_end"][sw][sl] / res["den_end"][sw][sl].clamp(min=1e-12))[:, wave]
            rows.append(dict(
                switch=f"{H.SEQ[sw]}→{H.SEQ[sw + 1]}（{P_old:g}→{P_new:g}）",
                react=float(res["react"][sw][sl][:, wave].float().mean()),
                sig_ratio=float((res["sig_pre"][sw][sl][:, wave] / sig4_exact(wcfg.noise, P_old)).median()),
                fires=float(res["fires_ph"][sw][sl][:, wave].mean()),
                sleeps=float(res["sleeps_ph"][sw][sl][:, wave].mean()),
                restless=float(res["out_fires"][sw][sl][:, wave].mean()) / (max(H.DWELL - H.WINDOW, 1) / 100),
                quality=float(q_col.mean()),
                resid=float(ratio.median())))
        out.append(dict(group=f"A={grp[0]:g}, B={grp[1]:g}, C={grp[2]:g}", rows=rows,
                        fires_life=float(res["fires_tot"][sl].mean()), sleeps_life=float(res["sleeps_tot"][sl].mean())))
    return out


def judged(start, sw):
    """判定する切り替え：大人から始めたときは2回目以降、育ちから始めたときはすべて"""
    return sw >= 1 if start == "adult" else True


def judge(allres, name, seeds):
    rs, rest, fi, sl_, ql, sg = [], [], [], [], [], []
    for start in STARTS:
        for seed in seeds:
            for grp in allres[f"{start}|{seed}|{name}"]["groups"]:
                for sw, r in enumerate(grp["rows"]):
                    if judged(start, sw):
                        rs.append(r["react"])
                        rest.append(r["restless"])
                        fi.append(r["fires"])
                        sl_.append(r["sleeps"])
                        ql.append(r["quality"])
                        sg.append(r["sig_ratio"])
    mn = min(rs)
    verdict = "慣れが戻った" if mn >= REACT_OK else ("戻らない" if mn <= REACT_BAD else "一部")
    return dict(verdict=verdict, react_min=mn, react_mean=float(np.mean(rs)), restless=float(np.mean(rest)),
                restless_flag=float(np.mean(rest)) > RESTLESS, fires=float(np.mean(fi)), sleeps=float(np.mean(sl_)),
                quality=float(np.mean(ql)), sig_ratio=float(np.mean(sg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    assert Physio2.surprise_ahead == 4 and Physio2.sigma_min == 0.02 and Physio2.theta == 4.0, "凍結したレシピの値と違う"
    name = "precheck_sleep"
    seeds = H.EVAL_SEEDS
    vars_ = variants()
    if args.smoke:
        H.DWELL = 160
        seeds = (300,)
        vars_ = [v for v in vars_ if v["name"] in ("床", "①", "②(γ=0.99)", "③(γ=0.99)", "③'")]
        name = "smoke_" + name
    out_dir = H.ROOT / "step3_memory" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    wcfg = W.WorldConfig(noise=0.02)
    allres = {}
    for start in STARTS:
        for seed in seeds:
            for v in vars_:
                res = run(v, wcfg, seed, args.device, start)
                allres[f"{start}|{seed}|{v['name']}"] = dict(groups=summarize(res, wcfg), elapsed=res["elapsed"])
                print(f"{start} seed={seed} {v['name']}: {res['elapsed']:.1f}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    report(allres, vars_, seeds, out_dir)


def report(allres, vars_, seeds, out_dir):
    J = {v["name"]: judge(allres, v["name"], seeds) for v in vars_}
    L = ["# Step 3 の睡眠の段の前の模擬（DESIGN.md 9章。読み方は走らせる前に決めた）", "",
         f"世界：標準（ノイズ0.02）。組 {', '.join('/'.join(f'{p:g}' for p in grp) for grp in H.GROUPS)}、並び {', '.join(H.SEQ)}、滞在{H.DWELL}。"
         f"評価の乱数 {', '.join(map(str, seeds))}。始め方：大人（学び終えた状態）と育ち（予測器0から）。寝るのは最後の驚きの発火から{N_Q}ステップ静かなとき（1回だけ）", "",
         "判定する切り替え：大人は2回目以降、育ちはすべて（組2つ×乱数×始め方2つ）。反応＝窓の中で1回でも驚いて発火した波の区域のセルの割合。",
         f"慣れが戻った＝判定する切り替えのすべてで反応 ≥ {REACT_OK}、戻らない＝どれかで ≤ {REACT_BAD}。落ち着かない＝窓の外の驚きの発火が1セル・100ステップあたり {RESTLESS} 回を超える（平均）", "",
         "| 版 | 判定 | 反応（最小） | 反応（平均） | 落ち着かなさ | 驚きの発火／切り替え | 睡眠の書き込み／切り替え | 直前の σ₄ 比（平均） | 窓の NLL（平均） |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v in vars_:
        j = J[v["name"]]
        L.append(f"| {v['name']} | {j['verdict']}{'・落ち着かない' if j['restless_flag'] else ''} | {j['react_min']:.2f} | {j['react_mean']:.2f} | {j['restless']:.3f} | "
                 f"{j['fires']:.2f} | {j['sleeps']:.2f} | {j['sig_ratio']:.2f} | {j['quality']:.3f} |")
    L += ["", "## 読み方（DESIGN.md 9.3）", ""]
    names = {v["name"] for v in vars_}
    if {"①", f"②(γ={G_MAIN})", f"③(γ={G_MAIN})"} <= names:
        ok = lambda n: J[n]["verdict"] == "慣れが戻った"
        bad = lambda n: J[n]["verdict"] == "戻らない"
        o1, o2, o3 = "①", f"②(γ={G_MAIN})", f"③(γ={G_MAIN})"
        if ok(o2) and bad(o1):
            msg = "効くのは係数を直すこと（σ₄ は係数が正確になれば自然に戻る）"
        elif ok(o1) and bad(o2):
            msg = "効くのは σ₄ を置き直すこと（驚く力を戻せば、不正確な係数でも学び直せる）"
        elif ok(o3) and not ok(o1) and not ok(o2):
            msg = "両方が要る（③ だけ戻った）"
        elif ok(o1) and ok(o2):
            msg = "① と ② のどちらでも足りる。睡眠の段には、費用の少ないほうを案にする"
        elif not (ok(o1) or ok(o2) or ok(o3)):
            gods = [n for n in ("①'", "②'", "③'") if n in J and J[n]["verdict"] == "慣れが戻った"]
            msg = ("この形の睡眠では慣れは戻らない → 予備案へ" if not gods else
                   f"局所の版では戻らないが、神様の版（{', '.join(gods)}）では戻る → 部分は効くが、局所の見積もりが足りない（予備案の前にそうに出す）")
        else:
            msg = "決めた読み方のどの場合にも当てはまらない（そのまま報告する）"
        L.append(f"- 局所の版（γ_L={G_MAIN}）：① {J[o1]['verdict']}、② {J[o2]['verdict']}、③ {J[o3]['verdict']} → **{msg}**")
        for fam in ("②", "③"):
            okg = [g for g in GAMMAS if f"{fam}(γ={g})" in J and J[f"{fam}(γ={g})"]["verdict"] == "慣れが戻った"]
            L.append(f"- {fam} の見直しの材料：「慣れが戻った」を満たすいちばん短い窓は " + (f"γ_L={min(okg)}（約{round(1 / (1 - min(okg)))}ステップ）" if okg else "なし"))
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    D = ["# 睡眠の模擬の細かい数字（切り替えごと、波の区域）", "",
         "反応／直前の σ₄ 比／驚きの発火／睡眠の書き込み／落ち着かなさ（窓の外、1セル・100ステップ）／窓の NLL／窓の終わりの学び残し", ""]
    for start in STARTS:
        for seed in seeds:
            base = allres[f"{start}|{seed}|{vars_[0]['name']}"]["groups"]
            for gi in range(len(H.GROUPS)):
                D += [f"## {'大人' if start == 'adult' else '育ち'}から、評価の乱数{seed}、{base[gi]['group']}", "",
                      "| 版 | " + " | ".join(r["switch"] for r in base[gi]["rows"]) + " | 一生の驚き／睡眠（全セル） |",
                      "|---|" + "---|" * (len(H.SEQ) - 1) + "---|"]
                for v in vars_:
                    gg = allres[f"{start}|{seed}|{v['name']}"]["groups"][gi]
                    cells = [f"{r['react']:.2f}／{r['sig_ratio']:.2f}／{r['fires']:.2f}／{r['sleeps']:.2f}／{r['restless']:.3f}／{r['quality']:.3f}／{r['resid']:.2f}" for r in gg["rows"]]
                    D.append(f"| {v['name']} | " + " | ".join(cells) + f" | {gg['fires_life']:.1f}／{gg['sleeps_life']:.1f} |")
                D.append("")
    (out_dir / "details.md").write_text("\n".join(D) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
