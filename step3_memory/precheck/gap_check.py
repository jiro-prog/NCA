"""
取り違えの物差しの確かめ (b)：思い出した保存先が、正確な係数からどれだけ離れていたかを数える（step3_memory/DESIGN.md 16章。2026-10-04、読み方は走らせる前に書いた）。
凍結した記憶の段をそのまま回して数えるだけ（仕組みは足さない）。回し方と取り違えの分け方は root_check.py（＝evaluate.run）と同じで、
思い出し・害のある取り違え・書き直し・正しく思い出した数が、root_check の同じ条件・乱数の数と1ビットも同じことを確かめる。

    .venv/Scripts/python.exe step3_memory/precheck/gap_check.py --noise 0.025 --period 38   # 区切って回す（評価の乱数2本）
    .venv/Scripts/python.exe step3_memory/precheck/gap_check.py --judge 1                   # 読み方を当てはめる

条件：ノイズ0.02・0.025 × 周期38・24（B）。ほかは15章と同じ（滞在1800、A=12・C=8、世界の実現8つ、評価の乱数320・321）。
B を見ている列（波の区域）での思い出しごとに、思い出した保存先と正確な B の係数の差 g（きれいな波の上での1歩先の予測の差）を、
正しく思い出した・書き直し・害のある取り違えごとに、細かい区間（0.0005 刻み、0.3 まで）で数える。思い出す前の係数の g も数える
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
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import cell as C  # noqa: E402
import evaluate as E  # noqa: E402
import root_check as RC  # noqa: E402
import world as W  # noqa: E402
from model import Physio2  # noqa: E402

CONDS = ((38.0, 0.02), (38.0, 0.025), (24.0, 0.02), (24.0, 0.025))
STEP, TOP = 0.0005, 0.3
NB = int(round(TOP / STEP))
COARSE = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, float("inf")))
OUT = ROOT / "runs" / "gap_check"
LS = ROOT / "runs" / "ls_bias" / "summary.json"


def hist_add(h, g):
    """g（1次元のテンソル）を細かい区間に足す。最後の区間は TOP 以上"""
    idx = torch.clamp((g / STEP).long(), max=NB)
    b = torch.bincount(idx, minlength=NB + 1)
    for i in torch.nonzero(b).flatten().tolist():
        h[i] += int(b[i])


@torch.no_grad()
def count_gaps(wcfg, group, seed, n_per=RC.N_PER, dwell=RC.DWELL, window=RC.WINDOW, dev="cpu"):
    B = n_per
    S = wcfg.size
    per = {name: torch.tensor([group[i]] * B, dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in RC.SEQ]
    T = len(RC.SEQ) * dwell
    g = torch.Generator(device=dev).manual_seed(seed)
    ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
    st = C.init_state(B, S, S, dev)
    cell = C.Cell3(sleep=True, memory=True, device=dev)
    wave = ~W.zone_mask(wcfg, dev)
    xs = torch.arange(S, device=dev)
    lag = wcfg.lag
    PA, PB, PC = (torch.tensor([group[i]] * B, device=dev).view(B, 1, 1) for i in range(3))
    cls_names = ("ok", "noop", "harmful")
    hist = {c: [0] * (NB + 1) for c in cls_names}
    hist_prev = {c: [0] * (NB + 1) for c in cls_names}
    labels = {c: {"A": 0, "B": 0, "C": 0, "none": 0} for c in cls_names}
    n = dict(recalls=0, harmful=0, noop=0, ok=0)
    t0 = time.time()
    for t in range(T):
        j = t // dwell
        if t % dwell == dwell - 1 and j + 1 < len(RC.SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        prev = st
        st, rec = cell.step(st, sense, jitter_p=0.0, gen=None)
        if j >= 1:
            rc = rec["recall"][:, 0]
            if bool(rc.any()):
                cP = colP.float().view(B, 1, S)
                cur_p = torch.where(cP == PA, 0, torch.where(cP == PB, 1, 2)).expand(B, S, S)
                u = st.core.w
                gg = torch.stack([E.pred_gap_t(u, PA), E.pred_gap_t(u, PB), E.pred_gap_t(u, PC)], 0)
                best_g, best_p = gg.min(0)
                slot_p = torch.where(best_g <= 0.02, best_p, torch.full_like(best_p, -1))
                sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
                noop = C.quad(st.core.R, u - prev.core.w) <= C.Physio3.tau * sig2
                wrong = rc & wave & (slot_p != cur_p)
                ok = rc & wave & (slot_p == cur_p)
                mB = cur_p == 1
                g_prev = E.pred_gap_t(prev.core.w, PB)
                for c, m in (("ok", ok & mB), ("noop", wrong & noop & mB), ("harmful", wrong & ~noop & mB)):
                    if bool(m.any()):
                        hist_add(hist[c], gg[1][m])
                        hist_add(hist_prev[c], g_prev[m])
                        sp = slot_p[m]
                        for name, val in (("A", 0), ("B", 1), ("C", 2), ("none", -1)):
                            labels[c][name] += int((sp == val).sum())
                n["recalls"] += int((rc & wave & mB).sum())
                n["harmful"] += int((wrong & ~noop & mB).sum())
                n["noop"] += int((wrong & noop & mB).sum())
                n["ok"] += int((ok & mB).sum())
        ws = W.advance(ws)
    return dict(noise=wcfg.noise, group=list(group), seed=seed, counts=n, hist=hist, hist_prev=hist_prev, labels=labels, elapsed=time.time() - t0)


def run_cond(P, noise):
    OUT.mkdir(parents=True, exist_ok=True)
    wcfg = W.WorldConfig(noise=noise)
    for seed in RC.SEEDS:
        path = OUT / f"n{noise:g}_B{P:g}_s{seed}.json"
        if path.exists():
            print("すでにある：", path.name, flush=True)
            continue
        res = count_gaps(wcfg, (RC.A_P, P, RC.C_P), seed)
        ref = json.loads((RC.OUT / f"n{noise:g}_B{P:g}_s{seed}.json").read_text(encoding="utf-8"))["patterns"]["B"]
        same = all(res["counts"][k] == ref[k] for k in ("recalls", "harmful", "noop", "ok"))
        res["same_as_root_check"] = same
        path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"ノイズ{noise:g} B={P:g} 評価の乱数{seed}: {res['elapsed']:.0f}秒、15章の数と1ビットも同じ：{'はい' if same else 'いいえ'}", flush=True)
        if not same:
            raise SystemExit("回帰の確かめが通らない")


def coarse(h):
    out = []
    for lo, hi in COARSE:
        out.append(sum(c for i, c in enumerate(h) if lo <= i * STEP < hi or (i == NB and hi == float("inf"))))
    return out


def median(h):
    tot = sum(h)
    if tot == 0:
        return None
    acc = 0
    for i, c in enumerate(h):
        acc += c
        if acc * 2 >= tot:
            return (i + 0.5) * STEP
    return None


def judge():
    ga = {(r["P"], r["noise"]): r["g_a"] for r in json.loads(LS.read_text(encoding="utf-8"))}
    L = ["# 取り違えの物差しの確かめ (b)：思い出した保存先と正確な係数の差（DESIGN.md 16章。読み方は走らせる前に決めた）", "",
         "B を見ている列（波の区域）での思い出し。g＝思い出した保存先と正確な B の係数の、きれいな波の上での1歩先の予測の差（線0.02）。数は評価の乱数ごと", ""]
    verdict = {}
    for P, noise in CONDS:
        for seed in RC.SEEDS:
            d = json.loads((OUT / f"n{noise:g}_B{P:g}_s{seed}.json").read_text(encoding="utf-8"))
            h_all = [sum(d["hist"][c][i] for c in ("ok", "noop", "harmful")) for i in range(NB + 1)]
            med = median(h_all)
            g_a = ga[(P, noise)]
            hc = coarse(d["hist"]["harmful"])
            tot_h = sum(hc)
            near, far = hc[1], hc[3]
            if tot_h == 0:
                read = "害のある取り違えがない"
            elif near * 2 > tot_h:
                read = "数え方のずれ"
            elif far * 2 > tot_h:
                read = "本当の取り違え"
            else:
                read = "どちらとも言えない"
            same_side = med is not None and ((med > 0.02) == (g_a > 0.02))
            close = med is not None and abs(med - g_a) <= g_a / 2
            agree = same_side and close
            verdict[(P, noise, seed)] = dict(read=read, agree=agree)
            L.append(f"## ノイズ{noise:g}・周期{P:g}・評価の乱数{seed}（15章の数と1ビットも同じ：{'はい' if d['same_as_root_check'] else 'いいえ'}）")
            L.append("")
            L.append(f"- 思い出し {d['counts']['recalls']}（正しく {d['counts']['ok']}、書き直し {d['counts']['noop']}、害のある取り違え {d['counts']['harmful']}）")
            L.append("| 種類 | 0〜0.02 | 0.02〜0.05 | 0.05〜0.1 | 0.1 以上 | 中央値 | 思い出す前の係数の中央値 | 保存先が当たった模様（A／B／C／どれにも当たらない） |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for c, name in (("ok", "正しく思い出した"), ("noop", "書き直し"), ("harmful", "害のある取り違え")):
                cc = coarse(d["hist"][c])
                m1, m2 = median(d["hist"][c]), median(d["hist_prev"][c])
                lb = d["labels"][c]
                L.append(f"| {name} | {cc[0]} | {cc[1]} | {cc[2]} | {cc[3]} | {'—' if m1 is None else f'{m1:.4f}'} | {'—' if m2 is None else f'{m2:.4f}'} | {lb['A']}／{lb['B']}／{lb['C']}／{lb['none']} |")
            L.append("")
            L.append(f"- **害のある取り違えの読み：{read}**（0.02〜0.05 が {near}／{tot_h}、0.1 以上が {far}／{tot_h}）")
            L.append(f"- **(a) と (b)**：思い出した保存先すべての g の中央値 {('—' if med is None else f'{med:.4f}')}、(a) の予想 g_a {g_a:.4f} → 線の同じ側：{'はい' if same_side else 'いいえ'}、"
                     f"差が g_a の半分以下：{'はい' if close else 'いいえ'} → **{'合う' if agree else '合わない'}**")
            L.append("")
    key = [(38.0, 0.025, s) for s in RC.SEEDS]
    if all(verdict[k]["read"] == "数え方のずれ" and verdict[k]["agree"] for k in key):
        summ = "ノイズ0.025・周期38 は（評価の乱数2本とも）数え方のずれで、(a) と (b) が合う → 「正しく思い出した数」が落ちたのは数え方のずれ。課題の順番を見直す"
    elif all(verdict[k]["read"] == "本当の取り違え" for k in key):
        summ = "ノイズ0.025・周期38 は本当の取り違え → 課題1 のまま"
    else:
        summ = "それ以外 → 数を並べて、そうと決める"
    k24 = [(24.0, 0.025, s) for s in RC.SEEDS]
    L.append(f"## まとめ（読み方どおり）：**{summ}**")
    L.append("")
    L.append("- ノイズ0.025・周期24（別に読む点）：" + "、".join(f"評価の乱数{s} は{verdict[(24.0, 0.025, s)]['read']}（(a) と (b) は{'合う' if verdict[(24.0, 0.025, s)]['agree'] else '合わない'}）" for _, _, s in k24))
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({f"{P:g}_{n:g}_{s}": v for (P, n, s), v in verdict.items()} | {"summary": summ}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--judge", type=int, default=0)
    args = ap.parse_args()
    if args.noise is not None:
        run_cond(args.period, args.noise)
    if args.judge:
        judge()


if __name__ == "__main__":
    main()
