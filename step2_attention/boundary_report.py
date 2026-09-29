"""
境目の実験の結果を、走らせる前に書いた予想と負けの条件に照らす（BOUNDARY.md §6。読み替えは走らせる前に書いた）。

    .venv/Scripts/python.exe step2_attention/boundary_report.py [--run runs/boundary]

- そうの予想「まれで不規則なほど、驚き駆動が有利になる」：m* = 3 で、Δ（2本の平均）は T_c が大きいほど大きく（どの CV でも）、CV が大きいほど大きい（どの T_c でも）
- 私の予想「不規則さ0〜1 では差が小さい。固まって来る変化（不規則さ2）は、時計に有利に働くかもしれない」：
  m* = 3 で、どの T_c でも |Δ(CV=1) − Δ(CV=0)| が引き分けの幅（2ナット）以内、かつ Δ(CV=2) < Δ(CV=1)
- 負けの条件「どれだけまれで不規則でも時計が勝つなら、この形の驚き駆動には効率の利点がない」：
  格子のどの世界でも、決めた予算（1・3・10・30）のどこでも「勝ち」がない（参考に、驚き駆動が使える範囲の細かい予算でも見る）
- 境目：m* で、隣どうしの判定が分かれたところ
出力：<run>/report.md、report.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from boundary import BUDGETS, CVS, M_STAR, TCS, TIE, delta, family_env, verdict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "runs" / "boundary"))
    args = ap.parse_args()
    run = Path(args.run)
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    results = {(w["T_c"], w["cv"]): w["per_seed"] for w in summ["worlds"]}
    grid = [(tc, cv) for tc in TCS for cv in CVS]
    assert all(g in results for g in grid), "24 の世界がそろっていない"
    rep = {}

    def dmean(tc, cv, m=M_STAR):
        ds = [delta(r, m) for r in results[(tc, cv)]]
        return None if any(d is None for d in ds) else float(np.mean(ds))

    # そうの予想
    inc_tc = {cv: [(a, b, dmean(a, cv), dmean(b, cv)) for a, b in zip(TCS[:-1], TCS[1:])] for cv in CVS}
    inc_cv = {tc: [(a, b, dmean(tc, a), dmean(tc, b)) for a, b in zip(CVS[:-1], CVS[1:])] for tc in TCS}
    up = lambda x, y: None if x is None or y is None else y > x
    tc_steps = [up(x, y) for v in inc_tc.values() for _, _, x, y in v]
    cv_steps = [up(x, y) for v in inc_cv.values() for _, _, x, y in v]
    tc_def = [x for x in tc_steps if x is not None]                  # 比べられた組だけで判定する（「届かない」の世界は比べられない）
    cv_def = [x for x in cv_steps if x is not None]
    rep["sou"] = dict(
        tc_increasing_all=bool(tc_def) and all(tc_def), tc_increasing_frac=float(np.mean(tc_def)) if tc_def else None,
        cv_increasing_all=bool(cv_def) and all(cv_def), cv_increasing_frac=float(np.mean(cv_def)) if cv_def else None,
        n_compared=(len(tc_def), len(cv_def)), n_not_compared=(len(tc_steps) - len(tc_def), len(cv_steps) - len(cv_def)),
        by_cv={f"{cv:g}": [(a, b, x, y) for a, b, x, y in v] for cv, v in inc_tc.items()},
        by_tc={str(tc): [(a, b, x, y) for a, b, x, y in v] for tc, v in inc_cv.items()})
    rep["sou"]["held"] = rep["sou"]["tc_increasing_all"] and rep["sou"]["cv_increasing_all"]

    # 私の予想
    mine = []
    for tc in TCS:
        d0, d1, d2 = dmean(tc, 0.0), dmean(tc, 1.0), dmean(tc, 2.0)
        ok01 = None if d0 is None or d1 is None else abs(d1 - d0) <= TIE
        ok2 = None if d1 is None or d2 is None else d2 < d1
        mine.append(dict(T_c=tc, d_cv0=d0, d_cv1=d1, d_cv2=d2, flat_0_to_1=ok01, cv2_below_cv1=ok2))
    f01 = [r["flat_0_to_1"] for r in mine if r["flat_0_to_1"] is not None]
    f2 = [r["cv2_below_cv1"] for r in mine if r["cv2_below_cv1"] is not None]
    rep["mine"] = dict(rows=mine, flat_all=bool(f01) and all(f01), cv2_below_all=bool(f2) and all(f2),
                       n_compared=(len(f01), len(f2)), n_not_compared=(len(mine) - len(f01), len(mine) - len(f2)))
    rep["mine"]["held"] = rep["mine"]["flat_all"] and rep["mine"]["cv2_below_all"]

    # 負けの条件
    wins = [(tc, cv, m) for tc, cv in grid for m in BUDGETS if verdict(results[(tc, cv)], m)[0] == "勝ち"]
    fine_wins = []
    for tc, cv in grid:
        envs = [family_env(r, "驚き駆動") for r in results[(tc, cv)]]
        lo, hi = max(e[0][0] for e in envs), min(e[-1][0] for e in envs)
        if hi <= lo:
            continue
        for m in np.exp(np.linspace(math.log(lo), math.log(hi), 20)):
            if verdict(results[(tc, cv)], float(m))[0] == "勝ち":
                fine_wins.append((tc, cv, float(m)))
                break
    rep["losing_condition"] = dict(triggered=not wins, wins_at_budgets=wins, worlds_with_win_on_fine_grid=len(fine_wins))

    # 境目（m* で隣どうしの判定が分かれたところ）
    v = {(tc, cv): verdict(results[(tc, cv)], M_STAR)[0] for tc, cv in grid}
    edges = []
    for cv in CVS:
        for a, b in zip(TCS[:-1], TCS[1:]):
            if v[(a, cv)] != v[(b, cv)]:
                edges.append(dict(between=[(a, cv), (b, cv)], verdicts=[v[(a, cv)], v[(b, cv)]]))
    for tc in TCS:
        for a, b in zip(CVS[:-1], CVS[1:]):
            if v[(tc, a)] != v[(tc, b)]:
                edges.append(dict(between=[(tc, a), (tc, b)], verdicts=[v[(tc, a)], v[(tc, b)]]))
    rep["edges_m3"] = edges
    # 負けの内訳（BOUNDARY.md §5）：驚き駆動がその予算まで発火を使いきれない（下の包みの最大の m が予算より小さい）ときの負けは「負け（使いきれない）」
    loss_kind = []
    for tc, cv in grid:
        for m in BUDGETS:
            if verdict(results[(tc, cv)], m)[0] != "負け":
                continue
            m_max = [family_env(r, "驚き駆動")[-1][0] for r in results[(tc, cv)]]
            loss_kind.append(dict(T_c=tc, cv=cv, m=m, surprise_m_max=m_max, saturated=all(x < m for x in m_max)))
    rep["losses"] = loss_kind
    rep["counts_m3"] = {k: sum(1 for x in v.values() if x == k) for k in ("勝ち", "引き分け", "負け", "届かない")}
    (run / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    f = lambda x: "—" if x is None else f"{x:+.1f}"
    L = ["# 境目の実験：予想と負けの条件との照らし合わせ（BOUNDARY.md §6）", "",
         f"m* = 3 の判定の数：勝ち {rep['counts_m3']['勝ち']}、引き分け {rep['counts_m3']['引き分け']}、負け {rep['counts_m3']['負け']}、届かない {rep['counts_m3']['届かない']}（24 の世界）", "",
         "## そうの予想（まれで不規則なほど、驚き駆動が有利）", "",
         f"- T_c が大きいほど Δ が大きいか：比べられた隣どうし {rep['sou']['n_compared'][0]} 組のうち "
         f"{0 if rep['sou']['tc_increasing_frac'] is None else rep['sou']['tc_increasing_frac'] * 100:.0f}% で大きくなった（比べられない組 {rep['sou']['n_not_compared'][0]}）",
         f"- CV が大きいほど Δ が大きいか：比べられた隣どうし {rep['sou']['n_compared'][1]} 組のうち "
         f"{0 if rep['sou']['cv_increasing_frac'] is None else rep['sou']['cv_increasing_frac'] * 100:.0f}% で大きくなった（比べられない組 {rep['sou']['n_not_compared'][1]}）",
         f"- **予想どおりか：{'はい' if rep['sou']['held'] else 'いいえ'}**", "",
         "| CV ＼ T_c | " + " | ".join(str(t) for t in TCS) + " |", "|---|" + "---|" * len(TCS)]
    for cv in CVS:
        L.append(f"| {cv:g} | " + " | ".join(f(dmean(tc, cv)) for tc in TCS) + " |")
    L += ["", "（Δ は m* = 3 での、変化1回あたりのナット。2本の平均。正なら驚き駆動が良い）", "",
          "## 私の予想（不規則さ0〜1 では差が小さい、固まって来る変化は時計に有利かも）", "",
          "| T_c | Δ(CV=0) | Δ(CV=1) | Δ(CV=2) | 0〜1 の差が2ナット以内 | Δ(CV=2) ＜ Δ(CV=1) |", "|---:|---:|---:|---:|---|---|"]
    for r in mine:
        L.append(f"| {r['T_c']} | {f(r['d_cv0'])} | {f(r['d_cv1'])} | {f(r['d_cv2'])} | {r['flat_0_to_1']} | {r['cv2_below_cv1']} |")
    L += ["", f"- **予想どおりか：{'はい' if rep['mine']['held'] else 'いいえ'}**（0〜1 が平ら：{rep['mine']['flat_all']}、CV=2 が下：{rep['mine']['cv2_below_all']}。"
          f"比べられた T_c は {rep['mine']['n_compared'][0]}・{rep['mine']['n_compared'][1]}、比べられない T_c は {rep['mine']['n_not_compared'][0]}・{rep['mine']['n_not_compared'][1]}）", "",
          "## 負けの条件", "",
          f"- 決めた予算（1・3・10・30）で「勝ち」があった世界と予算：{len(wins)} 件 → **負けの条件に当たる：{'はい' if rep['losing_condition']['triggered'] else 'いいえ'}**",
          f"- 参考：驚き駆動が使える範囲の細かい予算で「勝ち」がある世界：{len(fine_wins)} / 24", "",
          "## 負けの内訳（BOUNDARY.md §5：驚き駆動がその予算を使いきれない負けは「負け（使いきれない）」）", "",
          "| 予算 m | 負けの数 | そのうち使いきれない | 使える範囲の中での負け |", "|---:|---:|---:|---|"]
    for m in BUDGETS:
        ls = [x for x in loss_kind if x["m"] == m]
        inside = [f"T_c={x['T_c']}・CV={x['cv']:g}" for x in ls if not x["saturated"]]
        L.append(f"| {m:g} | {len(ls)} | {sum(x['saturated'] for x in ls)} | {'、'.join(inside) if inside else '—'} |")
    L += ["", "## 境目（m* = 3 で、隣どうしの判定が分かれたところ）", ""]
    for e in edges:
        (a1, c1), (a2, c2) = e["between"]
        L.append(f"- T_c={a1}・CV={c1:g}（{e['verdicts'][0]}） ｜ T_c={a2}・CV={c2:g}（{e['verdicts'][1]}）")
    (run / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
