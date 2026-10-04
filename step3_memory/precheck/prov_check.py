"""
出どころの物差しで、本当の取り違えを数え直す（step3_memory/DESIGN.md 17章。2026-10-04、そうの判断。読み方は数え直す前に書いた）。
凍結した記憶の段をそのまま回して数えるだけ（仕組みは足さない。セルは変えず、評価の側で保存先ごとの「出どころの表」を持つ）。
回し方は evaluate.run と同じ。古い物差し（正確な係数に0.02 以内で当たるか）の数も一緒に数え、
15章の条件は runs/root_check の数と、封印の条件は封印テストの記録（runs/sealed_test3/summary.json の記憶の段の見張り）と、1ビットも同じことを確かめる。

    .venv/Scripts/python.exe step3_memory/precheck/prov_check.py --set check --noise 0.025 --period 38   # 15章の条件（CPU。評価の乱数2本）
    .venv/Scripts/python.exe step3_memory/precheck/prov_check.py --set sealed --world T0 --seed 9301    # 封印の条件（GPU。封印テストと同じ世界）
    .venv/Scripts/python.exe step3_memory/precheck/prov_check.py --judge 1

出どころの表：保存先ごとに、しまった回数を、そのときその列が見ていた模様（A・B・C）ごとに数える（最初の時期から）。新しくしまう・入れ替えるときはその模様だけ1回、寄せるときはその模様に1つ足す。
ラベル：いちばん多い模様の回数が全体の3分の2 以上ならその模様、そうでなければ「混ざった保存先」（2分の1（半分より多い）・4分の3（以上）でも数える）。
分け方：ラベルがいま見ている模様と同じ → 正しく思い出した。違う → 取り違え（いまの係数と見分けがつかないものは書き直し、見分けがつくものは本当の害のある取り違え）。混ざった保存先は別に数える
"""
import argparse
import json
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
import sealed3 as S3  # noqa: E402
import world as W  # noqa: E402
from model import Physio2  # noqa: E402

OUT = ROOT / "runs" / "prov_check"
SEALED = ROOT / "runs" / "sealed_test3" / "summary.json"
LINES = {"2/3": (2, 3, False), "1/2": (1, 2, True), "3/4": (3, 4, False)}   # (a, b, 厳しい不等号か)：いちばん多い回数 × b ≥ a × 全体（1/2 は ＞）
OLD = ("ok", "noop", "harmful")
NEW = ("ok", "noop", "harmful", "mixed")


def label(tally, line):
    """tally (B,3,S,S) → ラベル (B,S,S)：0/1/2 は模様、-2 は混ざった保存先"""
    a, b, strict = LINES[line]
    total = tally.sum(1)
    top, arg = tally.max(1)
    ok = (top * b > a * total) if strict else (top * b >= a * total)
    return torch.where(ok, arg, torch.full_like(arg, -2))


@torch.no_grad()
def count_prov(wcfg, groups, n_per, seed, dwell, window, scope, dev="cpu", sealed=False):
    """scope="B"：B を見ている列（15章）。scope="all"：すべての模様（封印テストの見張りと同じ範囲）。組ごとに数える"""
    if not sealed:
        from sealed import check_dev_seed
        check_dev_seed(seed, "評価の乱数")
    B = n_per * len(groups)
    S = wcfg.size
    K = C.Physio3.k_slots
    per = {name: torch.tensor([grp[i] for grp in groups for _ in range(n_per)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in RC.SEQ]
    T = len(RC.SEQ) * dwell
    g = torch.Generator(device=dev).manual_seed(seed)
    ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
    st = C.init_state(B, S, S, dev)
    cell = C.Cell3(sleep=True, memory=True, device=dev)
    wave = ~W.zone_mask(wcfg, dev)
    PA, PB, PC = (torch.tensor([grp[i] for grp in groups for _ in range(n_per)], device=dev).view(B, 1, 1) for i in range(3))
    prov = torch.zeros(B, K, 3, S, S, dtype=torch.long, device=dev)
    grp_of = torch.arange(B, device=dev) // n_per
    G = len(groups)
    cnt = [dict(recalls=0, old={c: 0 for c in OLD}, new={ln: {c: 0 for c in NEW} for ln in LINES},
                cross={f"{o}->{n}": 0 for o in OLD for n in NEW}, pair={f"{a}<-{b}": 0 for a in "ABC" for b in "ABCX"}) for _ in range(G)]
    old_tot = dict(mis_change=0, mis_noop=0, rec_ok=0)
    t0 = time.time()
    for t in range(T):
        j = t // dwell
        if t % dwell == dwell - 1 and j + 1 < len(RC.SEQ):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        prev = st
        st, rec = cell.step(st, sense, jitter_p=0.0, gen=None)
        cP = colP.float().view(B, 1, S)
        cur_p = torch.where(cP == PA, 0, torch.where(cP == PB, 1, 2)).expand(B, S, S)
        # しまった保存先の出どころ
        merged, fresh = rec["merged"][:, 0], (rec["stored_new"] | rec["evicted"])[:, 0]
        sm = merged | fresh
        if bool(sm.any()):
            changed = (st.mslot != prev.mslot).any(2) | (st.mn != prev.mn)
            ch = changed & sm.unsqueeze(1)
            if not torch.equal(ch.sum(1), sm.long()):
                raise SystemExit(f"t={t}：しまった保存先がちょうど1つに決まらない")
            oh = torch.nn.functional.one_hot(cur_p, 3).permute(0, 3, 1, 2)                     # (B,3,S,S)
            prov = prov + ((ch & merged.unsqueeze(1)).unsqueeze(2) & oh.unsqueeze(1).bool()).long()
            prov = torch.where((ch & fresh.unsqueeze(1)).unsqueeze(2), oh.unsqueeze(1).expand_as(prov), prov)
        if j >= 1:
            rc = rec["recall"][:, 0]
            if bool(rc.any()):
                rsel = (st.mn > 0) & (st.mage == 0) & rc.unsqueeze(1)
                if not torch.equal(rsel.sum(1), rc.long()):
                    raise SystemExit(f"t={t}：思い出した保存先がちょうど1つに決まらない")
                tally = (prov * rsel.unsqueeze(2)).sum(1)                                      # (B,3,S,S)
                labs = {ln: label(tally, ln) for ln in LINES}
                u = st.core.w
                gg = torch.stack([E.pred_gap_t(u, PA), E.pred_gap_t(u, PB), E.pred_gap_t(u, PC)], 0)
                best_g, best_p = gg.min(0)
                slot_p = torch.where(best_g <= 0.02, best_p, torch.full_like(best_p, -1))
                sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
                noop = C.quad(st.core.R, u - prev.core.w) <= C.Physio3.tau * sig2
                base = rc & wave
                old_wrong = base & (slot_p != cur_p)
                old_tot["mis_change"] += int((old_wrong & ~noop).sum())
                old_tot["mis_noop"] += int((old_wrong & noop).sum())
                old_tot["rec_ok"] += int((base & (slot_p == cur_p)).sum())
                sc = base & (cur_p == 1) if scope == "B" else base
                old_cls = {"ok": sc & (slot_p == cur_p), "noop": sc & (slot_p != cur_p) & noop, "harmful": sc & (slot_p != cur_p) & ~noop}
                new_cls = {}
                for ln, lab in labs.items():
                    mixed = lab == -2
                    new_cls[ln] = {"ok": sc & ~mixed & (lab == cur_p), "noop": sc & ~mixed & (lab != cur_p) & noop,
                                   "harmful": sc & ~mixed & (lab != cur_p) & ~noop, "mixed": sc & mixed}
                for gi in range(G):
                    gm = (grp_of == gi).view(B, 1, 1)
                    c = cnt[gi]
                    c["recalls"] += int((sc & gm).sum())
                    for k, m in old_cls.items():
                        c["old"][k] += int((m & gm).sum())
                    for ln in LINES:
                        for k, m in new_cls[ln].items():
                            c["new"][ln][k] += int((m & gm).sum())
                    for o, mo in old_cls.items():
                        for n_, mn in new_cls["2/3"].items():
                            c["cross"][f"{o}->{n_}"] += int((mo & mn & gm).sum())
                    lab = labs["2/3"]
                    for ai, an in enumerate("ABC"):
                        for bi, bn in ((0, "A"), (1, "B"), (2, "C"), (-2, "X")):
                            c["pair"][f"{an}<-{bn}"] += int((sc & gm & (cur_p == ai) & (lab == bi)).sum())
        ws = W.advance(ws)
    # 最後に混ざった保存先の数（波の区域、埋まっている保存先）
    occ = st.mn > 0
    mixed_end = []
    for gi in range(G):
        gm = (grp_of == gi).view(B, 1, 1, 1)
        tot_occ = int((occ & gm & wave).sum())
        tal = prov.permute(0, 1, 3, 4, 2)                                                       # (B,K,S,S,3)
        total = tal.sum(-1)
        top = tal.max(-1).values
        mx = occ & (top * 3 < 2 * total) & gm & wave
        mixed_end.append(dict(occupied=tot_occ, mixed=int(mx.sum())))
    return dict(noise=wcfg.noise, size=wcfg.size, groups=[list(x) for x in groups], seed=seed, dwell=dwell, scope=scope,
                counts=cnt, old_totals=old_tot, mixed_end=mixed_end, elapsed=time.time() - t0)


def run_check(noise, period):
    OUT.mkdir(parents=True, exist_ok=True)
    wcfg = W.WorldConfig(noise=noise)
    for seed in RC.SEEDS:
        path = OUT / f"check_n{noise:g}_B{period:g}_s{seed}.json"
        if path.exists():
            print("すでにある：", path.name, flush=True)
            continue
        res = count_prov(wcfg, [(RC.A_P, period, RC.C_P)], RC.N_PER, seed, RC.DWELL, RC.WINDOW, "B", dev="cpu")
        ref = json.loads((RC.OUT / f"n{noise:g}_B{period:g}_s{seed}.json").read_text(encoding="utf-8"))["patterns"]["B"]
        c = res["counts"][0]
        same = c["recalls"] == ref["recalls"] and c["old"]["ok"] == ref["ok"] and c["old"]["noop"] == ref["noop"] and c["old"]["harmful"] == ref["harmful"]
        res["same_as_before"] = same
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"ノイズ{noise:g} B={period:g} 評価の乱数{seed}: {res['elapsed']:.0f}秒、古い物差しの数が15章と1ビットも同じ：{'はい' if same else 'いいえ'}", flush=True)
        if not same:
            raise SystemExit("回帰の確かめが通らない")


def run_sealed(world, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    spec = S3.SEALED_WORLDS[world]
    wcfg = W.WorldConfig(size=spec["size"], zone=spec["zone"], v=spec["v"], noise=spec["noise"])
    path = OUT / f"sealed_{world}_s{seed}.json"
    if path.exists():
        print("すでにある：", path.name, flush=True)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = count_prov(wcfg, list(S3.SEALED_GROUPS), 8, seed, spec["dwell"], 300, "all", dev=dev, sealed=True)
    mw = json.loads(SEALED.read_text(encoding="utf-8"))["results"][world][str(seed)]["memory_watch"]
    same = res["old_totals"]["mis_change"] == mw["mis_change"] and res["old_totals"]["mis_noop"] == mw["mis_noop"] and res["old_totals"]["rec_ok"] == mw["rec_ok"]
    res["same_as_before"] = same
    res["device"] = dev
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{world} 評価の乱数{seed}（{dev}）: {res['elapsed']:.0f}秒、古い物差しの数が封印テストの記録と1ビットも同じ：{'はい' if same else 'いいえ'}"
          f"（害のある {res['old_totals']['mis_change']}/{mw['mis_change']}、書き直し {res['old_totals']['mis_noop']}/{mw['mis_noop']}、正しい {res['old_totals']['rec_ok']}/{mw['rec_ok']}）", flush=True)
    if not same:
        raise SystemExit("回帰の確かめが通らない")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "—"


def judge():
    rows = []
    for (P, n) in [(P, n) for n, Ps in RC.GRID.items() for P in Ps]:
        cs = [json.loads((OUT / f"check_n{n:g}_B{P:g}_s{s}.json").read_text(encoding="utf-8")) for s in RC.SEEDS]
        rows.append((f"15章 ノイズ{n:g}・B={P:g}（B を見ている列）", [x["counts"][0] for x in cs], [x["mixed_end"][0] for x in cs], all(x["same_as_before"] for x in cs)))
    have_sealed = all((OUT / f"sealed_{w}_s{s}.json").exists() for w in S3.SEALED_WORLDS for s in S3.SEALED_SEEDS)
    if have_sealed:
        for w in S3.SEALED_WORLDS:
            cs = [json.loads((OUT / f"sealed_{w}_s{s}.json").read_text(encoding="utf-8")) for s in S3.SEALED_SEEDS]
            for gi, grp in enumerate(S3.SEALED_GROUPS):
                rows.append((f"封印 {w}・組 {'/'.join(f'{x:g}' for x in grp)}（すべての模様）", [x["counts"][gi] for x in cs], [x["mixed_end"][gi] for x in cs], all(x["same_as_before"] for x in cs)))

    def pool(cnts, key, sub=None, ln=None):
        if key == "recalls":
            return sum(c["recalls"] for c in cnts)
        if key == "old":
            return sum(c["old"][sub] for c in cnts)
        return sum(c["new"][ln][sub] for c in cnts)
    L = ["# 出どころの物差しで数え直した取り違え（DESIGN.md 17章。読み方は数え直す前に決めた）", "",
         "条件ごとに評価の乱数を合わせた数（15章の条件は2本、封印の条件は4本）。波の区域、最初の時期は数えない。出どころのラベルの線は3分の2", "",
         "| 条件 | 古い物差しが前と1ビットも同じ | 思い出しの総数 | 正しく（出どころ） | 書き直し（出どころの違う） | **本当の害のある取り違え** | 混ざった保存先 | 古い物差しの害のある取り違え | 古い物差しの正しく |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    worst, mixed_max = 0.0, 0.0
    for name, cnts, mends, same in rows:
        r = pool(cnts, "recalls")
        ok, nop, harm, mix = (pool(cnts, "new", k, "2/3") for k in ("ok", "noop", "harmful", "mixed"))
        oh, oo = pool(cnts, "old", "harmful"), pool(cnts, "old", "ok")
        worst = max(worst, harm / r if r else 0.0)
        mixed_max = max(mixed_max, mix / r if r else 0.0)
        L.append(f"| {name} | {'はい' if same else 'いいえ'} | {r} | {ok}（{pct(ok, r)}） | {nop}（{pct(nop, r)}） | **{harm}（{pct(harm, r)}）** | {mix}（{pct(mix, r)}） | {oh}（{pct(oh, r)}） | {oo}（{pct(oo, r)}） |")
    L += ["", f"- **混ざった保存先を思い出した割合の最大：{100 * mixed_max:.2f}%**（線1%）→ " +
          ("1% 未満なので、ラベルの線の選び方（2分の1〜4分の3）は結論に影響しないとみなす" if mixed_max < 0.01 else "1% 以上の条件があるので、2分の1・4分の3 でも数えて並べる（下の表。判定は3分の2 のまま）")]
    if mixed_max >= 0.01:
        L += ["", "| 条件 | 線 | 正しく | 書き直し | 本当の害のある取り違え | 混ざった |", "|---|---|---:|---:|---:|---:|"]
        for name, cnts, _, _ in rows:
            r = pool(cnts, "recalls")
            for ln in ("1/2", "2/3", "3/4"):
                ok, nop, harm, mix = (pool(cnts, "new", k, ln) for k in ("ok", "noop", "harmful", "mixed"))
                L.append(f"| {name} | {ln} | {pct(ok, r)} | {pct(nop, r)} | {pct(harm, r)} | {pct(mix, r)} |")
    L += ["", f"- **本当の害のある取り違えの割合の最大：{100 * worst:.2f}%**（線5%）→ " +
          ("**どれかの条件で5% を超えた → 課題の順番をもう一度そうと相談する**" if worst > 0.05 else "**どの条件でも5% 以下 → 落ち着かなさ（書き込みの費用）を次の問いにする**")]
    L += ["", "## 記録（判定に使わない）", "", "最後に混ざった保存先の数（埋まっている保存先、波の区域）と、古い物差しとの突き合わせ（どちらで何に分かれたか）、出どころのラベル（行：いま見ている模様、列：思い出した保存先の出どころ、X は混ざった）", ""]
    for name, cnts, mends, _ in rows:
        occ = sum(m["occupied"] for m in mends)
        mx = sum(m["mixed"] for m in mends)
        cross = {k: sum(c["cross"][k] for c in cnts) for k in cnts[0]["cross"]}
        pair = {k: sum(c["pair"][k] for c in cnts) for k in cnts[0]["pair"]}
        L.append(f"- {name}：最後に混ざった保存先 {mx}/{occ}（{pct(mx, occ)}）。突き合わせ（古い→出どころ）：" +
                 "、".join(f"{k} {v}" for k, v in cross.items() if v) + "。出どころ（見ている模様←出どころ）：" + "、".join(f"{k} {v}" for k, v in pair.items() if v))
    if have_sealed:
        sj = json.loads(SEALED.read_text(encoding="utf-8"))
        L += ["", "## 封印テストの「害のある取り違え ÷ 正しく思い出した」の訂正（判定は変えない。日付つきの訂正）", "",
              "| 世界 | 古い物差し（封印テストの記録） | 出どころの物差し | 本当の害のある取り違え ÷ 思い出しの総数 |", "|---|---:|---:|---:|"]
        for w in S3.SEALED_WORLDS:
            mw = [sj["results"][w][str(s)]["memory_watch"] for s in S3.SEALED_SEEDS]
            oldr = sum(m["mis_change"] for m in mw) / sum(m["rec_ok"] for m in mw)
            cs = [json.loads((OUT / f"sealed_{w}_s{s}.json").read_text(encoding="utf-8")) for s in S3.SEALED_SEEDS]
            cnts = [c for x in cs for c in x["counts"]]
            harm, ok, r = pool(cnts, "new", "harmful", "2/3"), pool(cnts, "new", "ok", "2/3"), pool(cnts, "recalls")
            L.append(f"| {w} | {oldr:.3f} | {harm / ok if ok else float('nan'):.3f} | {pct(harm, r)} |")
    else:
        L += ["", "（封印の条件はまだ数えていない）"]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("check", "sealed"), default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--world", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--judge", type=int, default=0)
    args = ap.parse_args()
    if args.set == "check":
        run_check(args.noise, args.period)
    elif args.set == "sealed":
        seeds = [args.seed] if args.seed else list(S3.SEALED_SEEDS)
        for s in seeds:
            run_sealed(args.world, s)
    if args.judge:
        judge()


if __name__ == "__main__":
    main()
