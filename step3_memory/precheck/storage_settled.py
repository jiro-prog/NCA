"""
(は) の診断（step3_memory/DESIGN.md 12.5。測り方と読み方は走らせる前に書いた。2026-10-02、そうの判断）。

    .venv/Scripts/python.exe step3_memory/precheck/storage_settled.py [--device cuda]

storage_check.py と同じ回し方（眠る段のそばで保存先の帳簿だけ。セルの動きは眠る段と1ビットも同じ）で、しまう決まりを3つ帳簿づけする：
  (し1)       静かな時期の2回目以降の見直しでしまう（12.3 と同じ。回帰の確かめ：12.3 と同じ数字になるはず）
  (し1)＋(は)  そのうえで、見直しで係数がほとんど動かなかったときだけしまう：(w_前 − w_L)ᵀR_L(w_前 − w_L) ≤ 6σ²（判定に使う）
  (し2)＋(は)  毎回の見直しで、ほとんど動かなかったときだけしまう（記録だけ）
w_前 は見直しの前の係数（寝たセルは驚いて発火していないので、そのステップの前の係数と同じ）、w_L は長い痕跡の最小二乗の解、σ² はそのセルの v（下限つき）。
測るもの：12.3 のものに足して、パターンごとのしまえたセルの数（A・B・C のそれぞれに当たる保存先を最後に持っているセル）と、何も入っていないセルの数。
出力：step3_memory/runs/precheck_storage/settled.md、settled.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import storage_check as SC  # noqa: E402
from storage_check import C, E, W, Physio2, pred_gap, quad  # noqa: E402

VARIANTS = ("し1", "し1+は", "し2+は")
FULL_MAX = 76                      # いっぱいになったことのあるセル ≤ 76（波の区域7680 の1%）
STORED_MIN = 6912                  # パターンごとのしまえたセル ≥ 6912（波の区域の9割）
EMPTY_MAX = 76                     # 何も入っていないセル ≤ 76


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
    books = {v: SC.Book(B, S_, S_, dev) for v in VARIANTS}
    n_settled = torch.zeros(B, S_, S_, device=dev)
    n_review2 = torch.zeros(B, S_, S_, device=dev)
    for t in range(T):
        j = t // plan.dwell
        if t % plan.dwell == plan.dwell - 1 and j + 1 < len(plan.seq):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        prev = st
        st, rec = cell.step(st, sense)
        for b in books.values():
            b.tick()
        sl = rec["sleep"][:, 0]
        k = st.k[:, 0]
        sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
        if bool(sl.any()):
            wL = C.ls_solution(st.RL, st.rL, Physio2.eps)
            settled = quad(st.RL, prev.core.w - wL) <= SC.TAU * sig2
        else:
            settled = torch.zeros_like(sl)
        books["し1"].store(sl & (k >= 2), st.core.w, st.RL, sig2)
        books["し1+は"].store(sl & (k >= 2) & settled, st.core.w, st.RL, sig2)
        books["し2+は"].store(sl & (k >= 1) & settled, st.core.w, st.RL, sig2)
        n_settled += (sl & (k >= 2) & settled).float()
        n_review2 += (sl & (k >= 2)).float()
        ws = W.advance(ws)
    wave = ~W.zone_mask(wcfg, dev)
    out = {}
    for v, b in books.items():
        rows = []
        for gi, grp in enumerate(plan.groups):
            sl_ = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
            n = b.n[sl_][:, :, wave]
            occ = n > 0
            used = occ.sum(1)
            slots = b.slot[sl_].permute(0, 1, 3, 4, 2)[:, :, wave]
            gaps = torch.stack([pred_gap(slots, P) for P in grp], -1)
            best_gap, best_p = gaps.min(-1)
            hit = occ & (best_gap <= SC.MATCH)
            junk = occ & ~hit
            dup = 0
            stored = []
            for p in range(3):
                k_p = (hit & (best_p == p)).sum(1)
                dup += int((k_p - 1).clamp(min=0).sum())
                stored.append(int((k_p >= 1).sum()))
            rows.append(dict(group="/".join(f"{p:g}" for p in grp),
                             used_hist=[int((used == u).sum()) for u in range(SC.K + 1)],
                             ever_full=int(b.ever_full[sl_][:, wave].sum()), evictions=int(b.evict[sl_][:, wave].sum()),
                             new=int(b.new[sl_][:, wave].sum()), merges=int(b.merge[sl_][:, wave].sum()),
                             junk=int(junk.sum()), dup=dup, stored=stored, empty=int((used == 0).sum()), n_cells=int(used.numel()),
                             settled_share=[int(n_settled[sl_][:, wave].sum()), int(n_review2[sl_][:, wave].sum())]))
        out[v] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    res = {}
    for start in ("adult", "grow"):
        for seed in (300, 301):
            res[f"{start}|{seed}"] = run(wcfg, seed, start, args.device)
            print(start, seed, "done", flush=True)
    L = ["# (は) の診断（DESIGN.md 12.5。測り方と読み方は走らせる前に決めた）", "",
         "波の区域のセルは条件ごとに7680。しまえた（A／B／C）＝そのパターンに当たる保存先を最後に持っているセルの数。空＝何も入っていないセルの数。",
         "落ち着いていた見直し＝2回目以降の見直しのうち、係数がほとんど動かなかった（(w_前 − w_L)ᵀR_L(w_前 − w_L) ≤ 6σ²）回数／2回目以降の見直しの回数", "",
         "| しまう決まり | 始め方 | 評価の乱数 | 組 | 使用数の分布（0/1/2/3/4） | いっぱいになったことのあるセル | 入れ替え | 当たらない | 重複 | しまえた（A／B／C） | 空 | 落ち着いていた見直し |",
         "|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---|"]
    verdict = dict(full=0, stored_min=10 ** 9, empty=0)
    for v in VARIANTS:
        for key, out in res.items():
            start, seed = key.split("|")
            for r in out[v]:
                if v == "し1+は":
                    verdict["full"] = max(verdict["full"], r["ever_full"])
                    verdict["stored_min"] = min(verdict["stored_min"], min(r["stored"]))
                    verdict["empty"] = max(verdict["empty"], r["empty"])
                L.append(f"| {v} | {'大人' if start == 'adult' else '育ち'} | {seed} | {r['group']} | {'/'.join(map(str, r['used_hist']))} | {r['ever_full']} | "
                         f"{r['evictions']} | {r['junk']} | {r['dup']} | {'／'.join(map(str, r['stored']))} | {r['empty']} | {r['settled_share'][0]}／{r['settled_share'][1]} |")
    ok_full = verdict["full"] <= FULL_MAX
    ok_store = verdict["stored_min"] >= STORED_MIN and verdict["empty"] <= EMPTY_MAX
    L += ["", "## 読み方（12.5）", "",
          f"- 無駄に埋まらない：(し1)＋(は) のいっぱいになったことのあるセルの最大 {verdict['full']}（基準 ≤ {FULL_MAX}）→ {'満たす' if ok_full else '満たさない'}",
          f"- しまえなさすぎない：パターンごとのしまえたセルの最小 {verdict['stored_min']}（基準 ≥ {STORED_MIN}）、空のセルの最大 {verdict['empty']}（基準 ≤ {EMPTY_MAX}）→ {'満たす' if ok_store else '満たさない'}",
          f"- **判定：{'(は) を採る' if (ok_full and ok_store) else ('重複が減りきらない → そうに出す' if not ok_full else '厳しすぎる → そうに出す')}**"]
    out_dir = SC.ROOT / "step3_memory" / "runs" / "precheck_storage"
    (out_dir / "settled.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "settled.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
