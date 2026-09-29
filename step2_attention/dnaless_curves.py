"""
DNA なしの発火のしかたの曲線（DESIGN.md 7.5 の比べ方）。学習は要らない。

    .venv/Scripts/python.exe step2_attention/dnaless_curves.py --eval-seeds 1000 1001 [--noise 0.005] [--sigma-min 0.005]

驚き駆動の家族（候補3 の K ∈ {1,2,5,10,20,40}。K=1 は元の設計の引き金）と、時計・サイコロの曲線を、
書き換え方をすべて DNA なし（η=1、g=1、Δw=0）にそろえて、育つ時期と大人の時期で回す。
驚き駆動の各点が、同じ発火数で時計・サイコロの下の包みを上回るかを、7.5 の compare() で判定する。

2026-09-29：候補A の小さな確かめのデータ（評価の乱数1000）を読み直して、「DNA なしの驚き駆動は、少ない発火では時計に勝っている」
ように見えたので、評価の乱数1001 でも確かめるために作った（EXPERIMENTS.md）。候補B の模擬（precheck/precheck_b.py）も、対照の曲線にこれを使う。
出力：runs/dnaless_curves_<名前>/summary.json、summary.md
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, Physio2  # noqa: E402
from protocol import compare, envelope, run_period  # noqa: E402

FAMILIES = {
    "驚き駆動": [{"kind": "cand3", "K": k} for k in (1, 2, 5, 10, 20, 40)],
    "サイコロ": [{"kind": "dice", "p": p} for p in (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)],
    "時計": [{"kind": "clock", "k": k} for k in (2, 4, 10, 30, 100, 300)],
}
CONTROLS = ("時計", "サイコロ")


def label(pol):
    return {"cand3": lambda p: "元の引き金" if p["K"] == 1 else f"候補3 K={p['K']}",
            "dice": lambda p: f"サイコロ p={p['p']}", "clock": lambda p: f"時計 k={p['k']}"}[pol["kind"]](pol)


@torch.no_grad()
def curves(wcfg, eval_seed, dev, families=FAMILIES):
    """DNA なしで、家族ごと・時期ごとの点（発火数、NLL ほか）を返す"""
    cell = AttentionCell(dna=False).to(dev).eval()
    pts = []
    for fam, pols in families.items():
        for pol in pols:
            for period in ("grow", "adult"):
                r = run_period(cell, pol, wcfg, period, eval_seed=eval_seed, device=dev)
                pts.append(dict(family=fam, policy=label(pol), **{k: v for k, v in r.items() if k != "policy"}))
    return pts


def judge(pts, contestant="驚き駆動", controls=CONTROLS):
    """比べる家族の各点を、対照の下の包みと比べる（7.5 の compare）"""
    out = []
    for period in ("grow", "adult"):
        env = {c: envelope([(p["fires"], p["quality"]) for p in pts if p["period"] == period and p["family"] == c])
               for c in controls}
        for p in sorted([p for p in pts if p["period"] == period and p["family"] == contestant], key=lambda p: p["fires"]):
            row = dict(period=period, policy=p["policy"], fires=p["fires"], quality=p["quality"])
            for c in controls:
                v, qc = compare((p["fires"], p["quality"]), env[c])
                row[c] = dict(verdict=v, control_nll=qc)
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--noise", type=float, default=None, help="世界のノイズ（省くと今の世界の値）")
    ap.add_argument("--sigma-min", type=float, default=None, help="σ の下限（省くと今の値）")
    ap.add_argument("--name", default="now")
    args = ap.parse_args()
    for s in args.eval_seeds:
        if s >= 9000:
            raise SystemExit("評価の乱数 9000 以上は封印テストのもの。開発では使わない（sealed.py）")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wcfg = W.WorldConfig() if args.noise is None else W.WorldConfig(noise=args.noise)
    if args.sigma_min is not None:
        Physio2.sigma_min = args.sigma_min
    out = HERE / "runs" / f"dnaless_curves_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    res = dict(world=wcfg.to_dict(), sigma_min=Physio2.sigma_min, seeds={})
    pj = {"grow": "育つ時期", "adult": "大人の時期"}
    lines = ["# DNA なしの驚き駆動と、時計・サイコロ（7.5 の比べ方）", "",
             f"世界のノイズ {wcfg.noise}、σ の下限 {Physio2.sigma_min}。書き換えはすべて DNA なし。", ""]
    for seed in args.eval_seeds:
        pts = curves(wcfg, seed, dev)
        verdicts = judge(pts)
        res["seeds"][str(seed)] = dict(points=pts, verdicts=verdicts)
        lines += [f"## 評価の乱数 {seed}", "", "| 時期 | 驚き駆動の点 | 発火数（1セル） | NLL | 時計（同じ発火数） | サイコロ（同じ発火数） |",
                  "|---|---|---:|---:|---|---|"]
        for v in verdicts:
            lines.append(f"| {pj[v['period']]} | {v['policy']} | {v['fires']:.2f} | {v['quality']:.3f} | "
                         f"{v['時計']['verdict']}（{v['時計']['control_nll']:.3f}） | {v['サイコロ']['verdict']}（{v['サイコロ']['control_nll']:.3f}） |")
        lines.append("")
        lines += ["対照の点：", "", "| 時期 | 点 | 発火数 | NLL |", "|---|---|---:|---:|"]
        for p in sorted([p for p in pts if p["family"] in CONTROLS], key=lambda p: (p["period"], p["family"], p["fires"])):
            lines.append(f"| {pj[p['period']]} | {p['policy']} | {p['fires']:.2f} | {p['quality']:.3f} |")
        lines.append("")
        print(f"評価の乱数 {seed}：済み", flush=True)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
