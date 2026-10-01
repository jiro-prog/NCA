"""
思い出し方の模擬（step3_memory/DESIGN.md 12.7。候補と読み方は走らせる前に書いた。2026-10-02、そうの判断）。

    .venv/Scripts/python.exe step3_memory/precheck/recall.py [--device cuda] [--smoke 1]

記憶の段＝眠る段（cell.Cell3(sleep=True)）＋保存先 K=4（帳簿は storage_check.Book）＋しまう決まり（見直しで落ち着いていたら、しまう：12.6）
＋忘れ方（いちばん長く使っていないものを入れ替える）＋思い出し方。眠る段のセルに、外から帳簿と思い出しの書き換えを足す（睡眠の段の模擬と同じやり方）。
思い出し（驚いて発火したとき。その発火の書き換えの代わりに、保存先の係数をそのまま書く。書き込みは1回）：
  D(u) = (u − w_ls)ᵀR(u − w_ls)（R・r は最近の痕跡、w_ls はその最小二乗の解）、u* はいちばん合う保存先、w_今 は驚いた瞬間の書き換える前の係数、σ² はそのセルの v
  (お1) D(u*) ≤ 6σ²　(お2) D(u*) ＋ 6σ² ≤ D(w_今)　(お3) 両方
参考：眠る段（思い出しなし。主張2 の比べる相手）、天井＋睡眠（驚いて発火したら正確な係数を書く）。
世界は開発のまま（標準、組 12/24/8・12/20/8、並び A, B, A, B, A, C, A, B、滞在600、評価の乱数300・301、始め方は大人と育ち）。
出力：step3_memory/runs/precheck_recall/（summary.md、summary.json）。試運転は runs/smoke_precheck_recall/
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import storage_check as SC  # noqa: E402
from storage_check import C, E, W, Physio2, pred_gap, quad  # noqa: E402

VARIANTS = ("眠る段", "天井＋睡眠", "お1", "お2", "お3")
MEMORY = ("お1", "お2", "お3")
TAU = SC.TAU
W_MARGIN, N_MARGIN = 0.5, 2.0          # 主張2 の線（10.2）：書き込み0.5回、窓のナット2
STARTS = ("adult", "grow")
SEEDS = (300, 301)


@torch.no_grad()
def run(var, wcfg, seed, start, dev, plan=E.Plan()):
    B = plan.n_per * len(plan.groups)
    S_ = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in plan.groups for _ in range(plan.n_per)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in plan.seq]
    T = len(plan.seq) * plan.dwell
    n_sw = len(plan.seq) - 1
    first_phase = {p: plan.seq.index(p) for p in "ABC"}
    g = torch.Generator(device=dev).manual_seed(seed)
    if start == "adult":
        core, ws = E.adult_core(wcfg, sched[0], g, dev)
        st = C.wrap(core)
    else:
        ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
        st = C.init_state(B, S_, S_, dev)
    cell = C.Cell3(sleep=True, device=dev)
    book = SC.Book(B, S_, S_, dev) if var in MEMORY else None
    wave = ~W.zone_mask(wcfg, dev)
    xs = torch.arange(S_, device=dev)
    lag = wcfg.lag
    PA = torch.tensor([grp[0] for grp in plan.groups for _ in range(plan.n_per)], device=dev).view(B, 1, 1)
    PB = torch.tensor([grp[1] for grp in plan.groups for _ in range(plan.n_per)], device=dev).view(B, 1, 1)
    PC = torch.tensor([grp[2] for grp in plan.groups for _ in range(plan.n_per)], device=dev).view(B, 1, 1)
    Z = lambda dt=torch.float32: torch.zeros(n_sw, B, S_, S_, device=dev, dtype=dt)
    nll_sum, fires_ph, sleeps_ph, recalls_ph = Z(), Z(), Z(), Z()
    react = Z(torch.bool)
    mis_novel = torch.zeros(B, S_, S_, device=dev)       # 初めてのパターンの時期の取り違え
    mis_return = torch.zeros(B, S_, S_, device=dev)      # 戻りの時期の取り違え
    rec_ok = torch.zeros(B, S_, S_, device=dev)          # 正しい保存先を思い出した回数
    mis_noop = torch.zeros(B, S_, S_, device=dev)        # 取り違えのうち、いまの係数と見分けがつかない保存先を書き直したもの（予測はほぼ変わらない）
    mis_change = torch.zeros(B, S_, S_, device=dev)      # 取り違えのうち、いまの係数と違う保存先を書いたもの（予測が変わる）
    nth_hist = torch.zeros(4, device=dev)                # 何回目の発火で思い出したか（1・2・3・4以上。波の区域）
    fire_since = torch.zeros(B, S_, S_, device=dev)      # その列に新しい波が届いてからの驚きの発火の数
    comp_sum = torch.zeros((), device=dev)               # 驚きの発火のときに比べた保存先の数の合計
    comp_n = torch.zeros((), device=dev)                 # 比べた驚きの発火の数
    for t in range(T):
        j = t // plan.dwell
        if t % plan.dwell == plan.dwell - 1 and j + 1 < len(plan.seq):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg).float().view(B, 1, S_)            # (B,1,W) その列がいま見ている周期
        prev = st
        st, rec = cell.step(st, sense)
        fired = rec["surprise"][:, 0]
        rel = (t - (j * plan.dwell + xs * lag)).view(1, 1, S_)
        if j >= 1:
            fire_since = torch.where((rel == 0).expand_as(fire_since), torch.zeros_like(fire_since), fire_since)
        fire_since = fire_since + fired.float()
        recalled = torch.zeros_like(fired)
        if var == "天井＋睡眠":
            w_star = E.exact_w_t(colP.view(B, S_).double()).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, S_, S_)
            st = st._replace(core=C.rewrite(st.core, w_star, (fired & wave).unsqueeze(1)))
        if book is not None:
            book.tick()
            occ = book.n > 0
            if bool((fired & occ.any(1)).any()):
                R, r = st.core.R, st.core.r
                w_ls = C.ls_solution(R, r, Physio2.eps)
                D = torch.stack([quad(R, book.slot[:, i] - w_ls) for i in range(SC.K)], 1)
                D = torch.where(occ, D, torch.full_like(D, float("inf")))
                D_best, i_best = D.min(1)
                D_cur = quad(R, prev.core.w - w_ls)
                sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
                c1 = D_best <= TAU * sig2
                c2 = D_best + TAU * sig2 <= D_cur
                cond = {"お1": c1, "お2": c2, "お3": c1 & c2}[var]
                recalled = fired & cond & occ.any(1)
                if bool(recalled.any()):
                    u = torch.gather(book.slot, 1, i_best.view(B, 1, 1, S_, S_).expand(B, 1, 3, S_, S_)).squeeze(1)
                    st = st._replace(core=C.rewrite(st.core, u, recalled.unsqueeze(1)))
                    onehot = torch.nn.functional.one_hot(i_best, SC.K).permute(0, 3, 1, 2).bool() & recalled.unsqueeze(1)
                    book.age = torch.where(onehot, torch.zeros_like(book.age), book.age)
                    # 取り違えの数え：思い出した保存先が当たるパターンと、その列がいま見ているパターン
                    gA = pred_gap_t(u, PA)
                    gB = pred_gap_t(u, PB)
                    gC = pred_gap_t(u, PC)
                    gg = torch.stack([gA, gB, gC], 0)
                    best_g, best_p = gg.min(0)
                    slot_p = torch.where(best_g <= SC.MATCH, best_p, torch.full_like(best_p, -1))
                    cur_p = torch.where(colP == PA, 0, torch.where(colP == PB, 1, 2)).expand(B, S_, S_)
                    j_col = torch.where(rel >= 0, torch.full_like(rel, j), torch.full_like(rel, j - 1)).expand(B, S_, S_)
                    fp = torch.tensor([first_phase["A"], first_phase["B"], first_phase["C"]], device=dev)[cur_p]
                    novel = fp == j_col
                    wrong = recalled & wave & (slot_p != cur_p)
                    ok = recalled & wave & (slot_p == cur_p)
                    noop = quad(R, u - prev.core.w) <= TAU * sig2      # いまの係数と、最近の材料の上で見分けがつかない
                    if j >= 1:
                        mis_noop += (wrong & noop).float()
                        mis_change += (wrong & ~noop).float()
                        mis_novel += (wrong & novel).float()
                        mis_return += (wrong & ~novel).float()
                        rec_ok += ok.float()
                        k_ = fire_since[recalled & wave].clamp(max=4).long() - 1
                        nth_hist += torch.bincount(k_, minlength=4).float()[:4]
            comp_sum += (occ.sum(1).float() * fired.float()).sum()
            comp_n += fired.float().sum()
            # しまう：見直しで落ち着いていたら（12.6）
            sl = rec["sleep"][:, 0]
            if bool(sl.any()):
                sig2s = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
                wL = C.ls_solution(st.RL, st.rL, Physio2.eps)
                settled = quad(st.RL, prev.core.w - wL) <= TAU * sig2s
                book.store(sl & settled, st.core.w, st.RL, sig2s)
        if j >= 1:
            sw = j - 1
            NL = E.nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
            in_win = (rel >= 0) & (rel < plan.window)
            nll_sum[sw] += NL * in_win.float()
            fires_ph[sw] += fired.float()
            sleeps_ph[sw] += rec["sleep"][:, 0].float()
            recalls_ph[sw] += recalled.float()
            react[sw] |= fired & in_win
        ws = W.advance(ws)
    out = dict(nll_sum=nll_sum, fires_ph=fires_ph, sleeps_ph=sleeps_ph, recalls_ph=recalls_ph, react=react, wave=wave,
               mis_novel=mis_novel, mis_return=mis_return, rec_ok=rec_ok, nth_hist=nth_hist.tolist(), mis_noop=mis_noop, mis_change=mis_change,
               comp=(float(comp_sum) / max(float(comp_n), 1.0)) if book is not None else None)
    if book is not None:
        out.update(slot_n=book.n, ever_full=book.ever_full, evict=book.evict)
    # 最後の状態（check_cell.py の回帰の確かめで、本物のセルの記憶の段と比べるため。計算は変えていない）
    out["final"] = dict(core=st.core, RL=st.RL, rL=st.rL, q=st.q, k=st.k,
                        slot=None if book is None else book.slot, n=None if book is None else book.n, age=None if book is None else book.age)
    return out


def pred_gap_t(u, P):
    """u (B,3,H,W) と周期 P (B,1,1) → (B,H,W)：きれいな波の上での1歩先の予測の差（storage_check.pred_gap と同じ式）"""
    c = torch.cos(2 * np.pi / P)
    dw1, dw2, db = u[:, 0] - 2 * c, u[:, 1] + 1.0, u[:, 2] - (1 - c)
    m, a = 0.5, 0.4
    return torch.sqrt((dw1 * m + dw2 * m + db) ** 2 + a * a / 2 * (dw1 ** 2 + dw2 ** 2 + 2 * dw1 * dw2 * c))


def per_switch(res, plan, gi):
    """組 gi の切り替えごとの数字：書き込み（全セル、驚き＋睡眠。思い出しは驚きの発火に含まれる）、窓のナット（波の区域）、思い出し、見逃し"""
    sl = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
    wave = res["wave"]
    rpc = wave.float().sum(0)
    rows = []
    for sw in range(len(plan.seq) - 1):
        q_col = (res["nll_sum"][sw][sl] * wave).sum((0, 1)) / (rpc * plan.n_per * plan.window)
        rows.append(dict(writes=float((res["fires_ph"][sw][sl] + res["sleeps_ph"][sw][sl]).mean()),
                         nats=float(q_col.mean()) * plan.window,
                         recalls=float(res["recalls_ph"][sw][sl].mean()),
                         sleeps=float(res["sleeps_ph"][sw][sl].mean()),
                         miss=int((~res["react"][sw][sl][:, wave]).sum())))
    return rows


def kinds(plan, start):
    """切り替えの種類（判定するもの）：戻り＝前に滞在したパターンへ、初めて＝一度も滞在していないパターンへ。大人から始めたときは1回目を除く"""
    out = {}
    for sw in range(len(plan.seq) - 1):
        if start == "adult" and sw == 0:
            continue
        p = plan.seq[sw + 1]
        out[sw] = "初めて" if p not in plan.seq[:sw + 1] else "戻り"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    plan = E.Plan()
    seeds, name = SEEDS, "precheck_recall"
    if args.smoke:
        plan = E.Plan(dwell=200, window=100, end=(80, 100))
        seeds, name = (300,), "smoke_precheck_recall"
    wcfg = W.WorldConfig(noise=0.02)
    res = {}
    for start in STARTS:
        for seed in seeds:
            for v in VARIANTS:
                r = run(v, wcfg, seed, start, args.device, plan)
                res[(start, seed, v)] = r
                print(start, seed, v, "done", flush=True)
    # 主張2 の当てはめ（10.2・10.3 の形）
    L = ["# 思い出し方の模擬（DESIGN.md 12.7。候補と読み方は走らせる前に決めた）", "",
         "書き込み＝驚きの発火＋睡眠の書き込み（全セルの1セル・1切り替えあたり。思い出しは驚きの発火の中の内訳で、二重に数えない）。窓のナット＝窓の NLL の平均×300（波の区域）。",
         "戻り：(A) 書き込み ≤ 眠る段 − 0.5 かつ ナット ≤ 眠る段＋2、または (B) ナット ≤ 眠る段 − 2 かつ 書き込み ≤ 眠る段（条件ごとに、戻りの切り替えの平均で）。",
         "守り：本（始め方×乱数）ごとに、初めての切り替えを合わせて、ナットの差の合計 ≤ 2×数、書き込みの差の合計 ≤ 0.5×数", ""]
    summary = {}
    for v in VARIANTS:
        if v == "眠る段":
            continue
        ret_ok_all, guard_ok_all = True, True
        rows_r, rows_g = [], []
        for start in STARTS:
            for seed in seeds:
                kd = kinds(plan, start)
                g_dn, g_dw, g_n = 0.0, 0.0, 0
                for gi, grp in enumerate(plan.groups):
                    rs = per_switch(res[(start, seed, "眠る段")], plan, gi)
                    rm = per_switch(res[(start, seed, v)], plan, gi)
                    ret = [sw for sw, k in kd.items() if k == "戻り"]
                    ws_ = np.mean([rs[sw]["writes"] for sw in ret]); wm = np.mean([rm[sw]["writes"] for sw in ret])
                    ns = np.mean([rs[sw]["nats"] for sw in ret]); nm = np.mean([rm[sw]["nats"] for sw in ret])
                    okA = (wm <= ws_ - W_MARGIN) and (nm <= ns + N_MARGIN)
                    okB = (nm <= ns - N_MARGIN) and (wm <= ws_)
                    ok = okA or okB
                    ret_ok_all &= ok
                    rows_r.append(f"| {v} | {'大人' if start == 'adult' else '育ち'} | {seed} | {'/'.join(f'{p:g}' for p in grp)} | {ws_:.2f} → {wm:.2f}（{wm - ws_:+.2f}） | "
                                  f"{ns:.1f} → {nm:.1f}（{nm - ns:+.1f}） | {'A' if okA else ''}{'B' if okB else ''}{'' if ok else '満たさない'} |")
                    for sw, k in kd.items():
                        if k == "初めて":
                            g_dn += rm[sw]["nats"] - rs[sw]["nats"]
                            g_dw += rm[sw]["writes"] - rs[sw]["writes"]
                            g_n += 1
                gok = (g_dn <= N_MARGIN * g_n) and (g_dw <= W_MARGIN * g_n)
                guard_ok_all &= gok
                rows_g.append(f"| {v} | {'大人' if start == 'adult' else '育ち'} | {seed} | {g_n} | {g_dn:+.1f}（≤ {N_MARGIN * g_n:g}） | {g_dw:+.2f}（≤ {W_MARGIN * g_n:g}） | {'満たす' if gok else '満たさない'} |")
        mis = sum(float(res[(s, sd, v)]["mis_novel"][:, res[(s, sd, v)]["wave"]].sum() + res[(s, sd, v)]["mis_return"][:, res[(s, sd, v)]["wave"]].sum())
                  for s in STARTS for sd in seeds) if v in MEMORY else None
        summary[v] = dict(returns=bool(ret_ok_all), guard=bool(guard_ok_all), misrecalls=mis)
        L += [f"## {v}", "", "| 版 | 始め方 | 評価の乱数 | 組 | 戻りの書き込み（眠る段 → この版） | 戻りの窓のナット | 戻り |", "|---|---|---:|---|---|---|---|"] + rows_r
        L += ["", "| 版 | 始め方 | 評価の乱数 | 初めての切り替えの数 | ナットの差の合計 | 書き込みの差の合計 | 守り |", "|---|---|---:|---:|---|---|---|"] + rows_g + [""]
    # 取り違え・思い出し・比べた数・保存先
    L += ["## 取り違えと思い出しの記録（波の区域、すべての条件の合計）", "",
          "| 版 | 取り違え（初めての時期／戻りの時期） | 正しく思い出した | 何回目の発火で思い出したか（1／2／3／4以上） | 驚きの発火1回あたりに比べた保存先の数 | いっぱいになったことのあるセル | 入れ替え | 見逃しの数 |",
          "|---|---|---:|---|---:|---:|---:|---:|"]
    for v in VARIANTS:
        tot_miss = sum(sum(r["miss"] for r in per_switch(res[(s, sd, v)], plan, gi)[(1 if s == "adult" else 0):]) for s in STARTS for sd in seeds for gi in range(len(plan.groups)))
        if v in MEMORY:
            mn = sum(float(res[(s, sd, v)]["mis_novel"][:, res[(s, sd, v)]["wave"]].sum()) for s in STARTS for sd in seeds)
            mr = sum(float(res[(s, sd, v)]["mis_return"][:, res[(s, sd, v)]["wave"]].sum()) for s in STARTS for sd in seeds)
            okc = sum(float(res[(s, sd, v)]["rec_ok"][:, res[(s, sd, v)]["wave"]].sum()) for s in STARTS for sd in seeds)
            nth = np.sum([res[(s, sd, v)]["nth_hist"] for s in STARTS for sd in seeds], 0)
            comp = np.mean([res[(s, sd, v)]["comp"] for s in STARTS for sd in seeds])
            full = sum(int(res[(s, sd, v)]["ever_full"][:, res[(s, sd, v)]["wave"]].sum()) for s in STARTS for sd in seeds)
            ev = sum(int(res[(s, sd, v)]["evict"][:, res[(s, sd, v)]["wave"]].sum()) for s in STARTS for sd in seeds)
            L.append(f"| {v} | {int(mn)}／{int(mr)} | {int(okc)} | {'／'.join(str(int(x)) for x in nth)} | {comp:.2f} | {full} | {ev} | {tot_miss} |")
        else:
            L.append(f"| {v} | — | — | — | — | — | — | {tot_miss} |")
    # 読み方
    passing = [v for v in MEMORY if summary[v]["returns"] and summary[v]["guard"]]
    if passing:
        best = min(summary[v]["misrecalls"] for v in passing)
        tied = [v for v in passing if summary[v]["misrecalls"] == best]
        choice = "お2" if "お2" in tied else ("お1" if "お1" in tied else tied[0])
        verdict = f"**{choice} を採る**（主張2 の条件を満たした候補：{', '.join(passing)}。取り違えの数：" + "、".join(f"{v} {int(summary[v]['misrecalls'])}" for v in passing) + "）"
    else:
        verdict = "**どの候補も主張2 の条件を満たさない → そうに出す**"
    L += ["", "## 読み方（12.7）", ""] + [f"- {v}：戻り {'満たす' if summary[v]['returns'] else '満たさない'}、守り {'満たす' if summary[v]['guard'] else '満たさない'}" for v in MEMORY]
    L += [f"- 参考の天井＋睡眠：戻り {'満たす' if summary['天井＋睡眠']['returns'] else '満たさない'}、守り {'満たす' if summary['天井＋睡眠']['guard'] else '満たさない'}", "", f"- 判定：{verdict}"]
    out_dir = SC.ROOT / "step3_memory" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    js = {f"{k[0]}|{k[1]}|{k[2]}": dict(groups=[per_switch(r, plan, gi) for gi in range(len(plan.groups))]) for k, r in res.items()}
    (out_dir / "summary.json").write_text(json.dumps(dict(summary=summary, runs=js), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
