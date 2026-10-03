"""
取り違えと落ち着かなさを、いま見ている模様ごとに数える安い模擬（step3_memory/DESIGN.md 15章。2026-10-03・04、そうの判断。読み方と線は走らせる前に書いた）。
凍結した記憶の段（cell.Cell3(sleep=True, memory=True)、タグ step3-recipe-frozen）をそのまま回して数えるだけ。仕組みは足さない。凍結したファイルは触らない。
回し方は evaluate.run と同じ（育ちから、乱数の使い方も同じ）。取り違えの分け方も evaluate.run と同じ（思い出した係数が当たる模様と、その列がいま見ている模様を比べる）。

    .venv/Scripts/python.exe step3_memory/precheck/root_check.py --check 1            # 回帰の確かめ：evaluate.run の合計と1ビットも同じか（開発の周期、短い）
    .venv/Scripts/python.exe step3_memory/precheck/root_check.py --noise 0.02 --period 48   # 区切って回す（ノイズと B の周期を1つ、評価の乱数2本。--period なしならそのノイズの周期を全部）
    .venv/Scripts/python.exe step3_memory/precheck/root_check.py --judge 1            # 読み方を当てはめて runs/root_check/summary.md に書く

条件（15章）：標準の世界（32×32、波の速さ0.25、右下8×8 のノイズの区域）、滞在1800、窓300、並び A, B, A, B, A, C, A, B、A=12・C=8、
ノイズ0.02 で B ∈ {24, 30, 38, 48}、0.025 で B ∈ {24, 30, 38, 48}、0.03 で B ∈ {30, 48}。評価の乱数320・321、世界の実現8つ。
数えるもの（B を見ている列、波の区域）：思い出しの総数、害のある取り違え（evaluate.run の mis_change）、書き直し（mis_noop）、正しく思い出した数（rec_ok）、
B の波が届いてから300ステップより後（窓の外）の驚きの発火と、その区域のセル×ステップの数。A と C についても同じ数を記録する（判定には使わない）
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "step2_attention"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import cell as C  # noqa: E402
import evaluate as E  # noqa: E402
import world as W  # noqa: E402
from model import Physio2  # noqa: E402

SEQ = ("A", "B", "A", "B", "A", "C", "A", "B")
A_P, C_P = 12.0, 8.0
GRID = {0.02: (24.0, 30.0, 38.0, 48.0), 0.025: (24.0, 30.0, 38.0, 48.0), 0.03: (30.0, 48.0)}
PAIRS = (((48.0, 0.02), (38.0, 0.025)), ((38.0, 0.02), (30.0, 0.025)), ((30.0, 0.02), (24.0, 0.025)), ((38.0, 0.025), (30.0, 0.03)))
SEEDS = (320, 321)
DWELL, WINDOW, N_PER = 1800, 300, 8
MIN_RECALLS = 1000
RHO_LINE = -0.8
OUT = ROOT / "runs" / "root_check"


def v4(P, noise, amp=0.4):
    """4歩の変化の大きさ（きれいな波の s(t+4) − s(t) の二乗平均の平方根）÷ ノイズ"""
    return math.sqrt(2) * amp * abs(math.sin(4 * math.pi / P)) / noise


@torch.no_grad()
def count(wcfg, group, seed, n_per, dwell, window, dev="cpu"):
    """evaluate.run（育ち、ずれなし、天井なし）と同じ回し方で、模様ごとの数を返す。回帰の確かめ用に、evaluate.run と比べる合計も返す"""
    B = n_per
    S = wcfg.size
    per = {name: torch.tensor([group[i]] * B, dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in SEQ]
    T = len(SEQ) * dwell
    g = torch.Generator(device=dev).manual_seed(seed)
    ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
    st = C.init_state(B, S, S, dev)
    cell = C.Cell3(sleep=True, memory=True, device=dev)
    wave = ~W.zone_mask(wcfg, dev)
    xs = torch.arange(S, device=dev)
    lag = wcfg.lag
    PA, PB, PC = (torch.tensor([group[i]] * B, device=dev).view(B, 1, 1) for i in range(3))
    z = lambda: [0, 0, 0]  # noqa: E731
    rec_all, harm, noop_n, ok_n, fq, cq = z(), z(), z(), z(), z(), z()
    tot = dict(mis_change=0, mis_noop=0, rec_ok=0, fires=0, recalls=0)
    t0 = time.time()
    for t in range(T):
        j = t // dwell
        if t % dwell == dwell - 1 and j + 1 < len(SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        prev = st
        st, rec = cell.step(st, sense, jitter_p=0.0, gen=None)
        if j >= 1:
            fired = rec["surprise"][:, 0]
            rc = rec["recall"][:, 0]
            tot["fires"] += int(fired.sum())
            tot["recalls"] += int(rc.sum())
            rel = (t - (j * dwell + xs * lag)).view(1, 1, S)
            j_col = torch.where(rel >= 0, torch.full_like(rel, j), torch.full_like(rel, j - 1))
            since = t - (j_col * dwell + xs.view(1, 1, S) * lag)            # その列にいまの模様の波が届いてからのステップ数
            cP = colP.float().view(B, 1, S)
            cur_p = torch.where(cP == PA, 0, torch.where(cP == PB, 1, 2)).expand(B, S, S)
            quiet = (since >= window).expand(B, S, S) & wave
            for k in range(3):
                m = cur_p == k
                fq[k] += int((fired & quiet & m).sum())
                cq[k] += int((quiet & m).sum())
            if bool(rc.any()):
                u = st.core.w
                gg = torch.stack([E.pred_gap_t(u, PA), E.pred_gap_t(u, PB), E.pred_gap_t(u, PC)], 0)
                best_g, best_p = gg.min(0)
                slot_p = torch.where(best_g <= 0.02, best_p, torch.full_like(best_p, -1))
                sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
                noop = C.quad(st.core.R, u - prev.core.w) <= C.Physio3.tau * sig2
                wrong = rc & wave & (slot_p != cur_p)
                ok = rc & wave & (slot_p == cur_p)
                tot["mis_change"] += int((wrong & ~noop).sum())
                tot["mis_noop"] += int((wrong & noop).sum())
                tot["rec_ok"] += int(ok.sum())
                for k in range(3):
                    m = cur_p == k
                    rec_all[k] += int((rc & wave & m).sum())
                    harm[k] += int((wrong & ~noop & m).sum())
                    noop_n[k] += int((wrong & noop & m).sum())
                    ok_n[k] += int((ok & m).sum())
        ws = W.advance(ws)
    pats = {}
    for k, name in enumerate("ABC"):
        pats[name] = dict(period=group[k], v4=v4(group[k], wcfg.noise), recalls=rec_all[k], harmful=harm[k], noop=noop_n[k], ok=ok_n[k],
                          quiet_fires=fq[k], quiet_cellsteps=cq[k], restless=(100.0 * fq[k] / cq[k] if cq[k] else None))
    return dict(noise=wcfg.noise, group=list(group), seed=seed, n_per=n_per, dwell=dwell, window=window, patterns=pats, totals=tot, elapsed=time.time() - t0)


def check():
    """回帰の確かめ：開発の周期の短い条件で、evaluate.run と合計が1ビットも同じか"""
    wcfg = W.WorldConfig(noise=0.02)
    group = (12.0, 24.0, 8.0)
    plan = E.Plan(groups=(group,), n_per=4, dwell=600, window=300, end=(250, 300))
    ref = E.run(C.Cell3(sleep=True, memory=True), wcfg, 320, "grow", plan=plan, dev="cpu")
    wv = ref["wave"]
    mine = count(wcfg, group, 320, n_per=4, dwell=600, window=300)
    want = dict(mis_change=int(ref["mis_change"][:, wv].sum()), mis_noop=int(ref["mis_noop"][:, wv].sum()), rec_ok=int(ref["rec_ok"][:, wv].sum()),
                fires=int(ref["fires_ph"].sum()), recalls=int(ref["recalls_ph"].sum()))
    got = mine["totals"]
    same = all(want[k] == got[k] for k in want)
    split = all(sum(mine["patterns"][p][k] for p in "ABC") == got[g] for k, g in (("harmful", "mis_change"), ("noop", "mis_noop"), ("ok", "rec_ok")))
    lines = ["# 回帰の確かめ（root_check.py と evaluate.run。ノイズ0.02、組 12/24/8、世界の実現4つ、滞在600、評価の乱数320）", "",
             "| 数 | evaluate.run | root_check.py |", "|---|---:|---:|"]
    lines += [f"| {k} | {want[k]} | {got[k]} |" for k in want]
    lines += ["", f"- 合計が1ビットも同じ：{'はい' if same else 'いいえ'}。模様ごとに分けた数を足すと合計に戻る：{'はい' if split else 'いいえ'}",
              f"- かかった時間：root_check.py {mine['elapsed']:.0f}秒（{len(SEQ) * 600}ステップ×世界の実現4つ）"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not (same and split):
        raise SystemExit("回帰の確かめが通らない")


def run_noise(noise, seeds=SEEDS, only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    wcfg = W.WorldConfig(noise=noise)
    for P in GRID[noise]:
        if only is not None and P != only:
            continue
        for seed in seeds:
            path = OUT / f"n{noise:g}_B{P:g}_s{seed}.json"
            if path.exists():
                print("すでにある：", path.name, flush=True)
                continue
            res = count(wcfg, (A_P, P, C_P), seed, N_PER, DWELL, WINDOW)
            path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"ノイズ{noise:g} B={P:g} 評価の乱数{seed}: {res['elapsed']:.0f}秒", flush=True)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else None


def judge():
    conds = [(P, n) for n, Ps in GRID.items() for P in Ps]
    data = {}
    for (P, n) in conds:
        for seed in SEEDS:
            path = OUT / f"n{n:g}_B{P:g}_s{seed}.json"
            data[(P, n, seed)] = json.loads(path.read_text(encoding="utf-8"))["patterns"]["B"]
    L = ["# 取り違えと落ち着かなさは、同じ物差し（v₄）で並ぶか（DESIGN.md 15章。読み方と線は走らせる前に決めた）", "",
         "v₄ ＝ 4歩の変化の大きさ ÷ ノイズ ＝ √2·0.4·|sin(4π/P)| ÷ σ_n。数は B を見ている列（波の区域）、世界の実現8つ、滞在1800。凍結した記憶の段", "",
         "| ノイズ | B の周期 | v₄ | 評価の乱数 | 思い出しの総数（分母） | 害のある取り違え（分子） | 取り違えの割合 | 書き直し | 正しく思い出した | 窓の外の驚き | 窓の外のセル×ステップ | 落ち着かなさ（1セル・100歩） |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (P, n) in sorted(conds, key=lambda c: -v4(*c)):
        for seed in SEEDS:
            d = data[(P, n, seed)]
            ratio = d["harmful"] / d["recalls"] if d["recalls"] else None
            L.append(f"| {n:g} | {P:g} | {v4(P, n):.2f} | {seed} | {d['recalls']} | {d['harmful']} | {'—' if ratio is None else f'{ratio:.3f}'}"
                     f"{'（判定から外す）' if d['recalls'] < MIN_RECALLS else ''} | {d['noop']} | {d['ok']} | {d['quiet_fires']} | {d['quiet_cellsteps']} | {d['restless']:.3f} |")
    verdict = {}
    L += ["", "## 読み方の当てはめ", ""]
    for metric, label in (("misrecall", "取り違えの割合"), ("restless", "落ち着かなさ")):
        per_seed = {}
        for seed in SEEDS:
            vals, excluded = {}, []
            for (P, n) in conds:
                d = data[(P, n, seed)]
                if metric == "misrecall":
                    if d["recalls"] < MIN_RECALLS:
                        excluded.append((P, n))
                        continue
                    vals[(P, n)] = d["harmful"] / d["recalls"]
                else:
                    vals[(P, n)] = d["restless"]
            keys = list(vals)
            rho = spearman([v4(*k) for k in keys], [vals[k] for k in keys]) if len(keys) >= 2 else None
            width = (max(vals.values()) - min(vals.values())) if vals else 0.0
            pair_rows, missing = [], 0
            for a, b in PAIRS:
                if a not in vals or b not in vals:
                    missing += 1
                    pair_rows.append((a, b, None, None))
                    continue
                diff = abs(vals[a] - vals[b])
                pair_rows.append((a, b, diff, width > 0 and diff <= width / 4))
            undecidable = metric == "misrecall" and (len(excluded) >= 3 or missing >= 2)
            ok_rank = rho is not None and rho <= RHO_LINE
            ok_pairs = width > 0 and all(r[3] for r in pair_rows if r[2] is not None)
            lined = (not undecidable) and ok_rank and ok_pairs
            per_seed[seed] = dict(rho=rho, width=width, pairs=pair_rows, excluded=excluded, undecidable=undecidable, ok_rank=ok_rank, ok_pairs=ok_pairs, lined=lined)
            L.append(f"- **{label}・評価の乱数{seed}**：順位相関 {('—' if rho is None else f'{rho:+.3f}')}（線 ≤ {RHO_LINE}）→ {'満たす' if ok_rank else '満たさない'}。"
                     f"幅 {width:.4f}（4分の1 は {width / 4:.4f}）。外した条件 {len(excluded)}{'（' + '、'.join(f'B={p:g}・ノイズ{nn:g}' for p, nn in excluded) + '）' if excluded else ''}")
            for a, b, diff, ok in pair_rows:
                L.append(f"  - そろえた組 B={a[0]:g}・ノイズ{a[1]:g}（v₄ {v4(*a):.2f}）と B={b[0]:g}・ノイズ{b[1]:g}（v₄ {v4(*b):.2f}）："
                         + ("欠けた" if diff is None else f"{vals[a]:.4f} と {vals[b]:.4f}、差 {diff:.4f} → {'満たす' if ok else '満たさない'}"))
            L.append(f"  - **{'決められない' if undecidable else ('並ぶ' if lined else '並ばない')}**")
        lined_all = all(per_seed[s]["lined"] for s in SEEDS)
        undecidable_any = any(per_seed[s]["undecidable"] for s in SEEDS)
        verdict[metric] = dict(per_seed=per_seed, lined=lined_all, undecidable=undecidable_any)
        L.append(f"- **{label}：{'決められない' if undecidable_any else ('同じ物差し（v₄）で並ぶ（評価の乱数2本とも）' if lined_all else '並ばない')}**")
        L.append("")
    both = verdict["misrecall"]["lined"] and verdict["restless"]["lined"]
    L.append(f"## 判定：**{'両方が同じ物差し（v₄）で並ぶ → 次の問いを1つにする' if both else '片方だけ並ぶ、またはどちらも並ばない（決められないを含む）→ 別々に、取り違えから進める'}**")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    js = {f"{P:g}_{n:g}_{s}": d for (P, n, s), d in data.items()}
    (OUT / "summary.json").write_text(json.dumps(dict(data=js, verdict={m: dict(lined=v["lined"], undecidable=v["undecidable"],
                                                                                 per_seed={str(s): dict(rho=x["rho"], width=x["width"], lined=x["lined"], excluded=[list(e) for e in x["excluded"]],
                                                                                                        pairs=[[list(a), list(b), dd, oo] for a, b, dd, oo in x["pairs"]])
                                                                                           for s, x in v["per_seed"].items()}) for m, v in verdict.items()},
                                                       both=both), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--period", type=float, default=None)   # 区切って回すとき：B の周期を1つだけ
    ap.add_argument("--judge", type=int, default=0)
    args = ap.parse_args()
    if args.check:
        check()
    if args.noise is not None:
        run_noise(args.noise, only=args.period)
    if args.judge:
        judge()


if __name__ == "__main__":
    main()
