"""
候補B を本物のセルで確かめる（DESIGN.md 14章。合否は 2026-09-29 に走らせる前に決めた）。

    .venv/Scripts/python.exe step2_attention/verify_b.py [--runs runs/2b_B_seed0 runs/2b_B_seed1] [--quick 1] [--device cpu]

世界はパッチを外した元の形（ノイズ0.02、σ の下限0.02、大人の時期の組は 12→24・12→20・24→36・24→12）。驚きの信号は4歩先。評価の乱数1000。
- 合否：学習シードの2本とも、育つ時期と大人の時期の両方で、B（学んだ DNA、θ=4）の点が、時計とサイコロの下の包みをどちらも「上回る」（7.5 の compare）。
  対照の書き換え方は {DNA なし、そのシードの B の DNA} の良いほう（対照に有利）。それ以外（決められないを含む）は不合格
- 合否に入れないが出すもの：予算を振った曲線（同じ DNA で θ ∈ {2,3,4,6,8}、評価だけ）、組ごとの結果（とくに 24→36）、DNA なしの B と1歩先の驚き（元の設計）の点、
  近所を隠す診断、大人の時期の中身、S2-10 の確かめ（DNA なしの密な書き換えが 2a の元の数字を満たすか）
出力：runs/verify_b/（summary.md、summary.json、pareto_<時期>_seed<シード>.png）
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, Physio2  # noqa: E402
from protocol import PAIRS_ORIGINAL, Protocol2, compare, envelope, hide_neighbors, run_period, run_protocol2  # noqa: E402

NOISE = 0.02
THETAS = (2.0, 3.0, 4.0, 6.0, 8.0)
CONTROLS = {
    "時計": [{"kind": "clock", "k": k} for k in (2, 4, 10, 30, 100, 300)],
    "サイコロ": [{"kind": "dice", "p": p} for p in (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)],
}
PJ = {"grow": "育つ時期", "adult": "大人の時期"}


def rp(cell, pol, wcfg, period, dev, ahead=4, theta=4.0):
    """驚きの信号の先としきい値を決めて run_period（元の組、評価の乱数1000）"""
    Physio2.surprise_ahead, Physio2.theta = ahead, theta
    try:
        return run_period(cell, pol, wcfg, period, eval_seed=1000, device=dev, pairs=PAIRS_ORIGINAL)
    finally:
        Physio2.surprise_ahead, Physio2.theta = 1, 4.0


def check_s2_10(dev):
    """S2-10 の確かめ：DNA なしの密な書き換え（2a と同じ）をノイズ0.02 で回し、2a の元の数字を満たすか（波の σ ≤ 0.06 が S2-10 で動いた数字）"""
    wcfg = W.WorldConfig(noise=NOISE)
    s, _, _ = run_protocol2(AttentionCell(dna=False).to(dev).eval(), wcfg, Protocol2(), eval_seed=1000, dense=True, device=dev)
    checks = dict(sigma_wave=s["sigma_median_wave"] <= 0.06, sigma_noise=0.25 <= s["sigma_median_noise"] <= 0.33,
                  sigma_ratio=s["sigma_ratio"] >= 5, calib_wave=0.5 <= s["settle_z2_wave"] <= 1.5,
                  calib_noise=0.5 <= s["settle_z2_noise"] <= 1.5, nll_wave=s["settle_nll_wave"] <= s["nll_irreducible"] + 0.2)
    keep = ("sigma_median_wave", "sigma_median_noise", "sigma_ratio", "settle_z2_wave", "settle_z2_noise", "settle_nll_wave", "nll_irreducible")
    return dict(passed=all(checks.values()), checks=checks, values={k: s[k] for k in keep})


def plot(period, seed, ctrl, b_curve, b_pt, refs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, name in enumerate(CONTROLS):
        pts = [(p["fires"], p["quality"]) for p in ctrl if p["family"] == name]
        env = envelope(pts)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=SERIES[i + 1], alpha=0.35, linewidths=0)
        ax.plot([p[0] for p in env], [p[1] for p in env], color=SERIES[i + 1], linewidth=2, label=f"{name}（下の包み）")
    ax.plot([p[0] for p in b_curve], [p[1] for p in b_curve], color=SERIES[0], linewidth=1.2, alpha=0.6, marker="o", markersize=3,
            label="B の曲線（θ を振った。評価だけ）")
    ax.scatter([b_pt[0]], [b_pt[1]], s=70, color=SERIES[0], zorder=5, label="B（学んだ DNA、θ=4）")
    for (name, pt), mk, col in zip(refs.items(), ("s", "D"), (SERIES[4], SERIES[5])):
        ax.scatter([pt[0]], [pt[1]], s=45, marker=mk, color=col, zorder=5, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("発火数（1セルあたり、全セル）", color=INK_2, fontsize=9)
    ax.set_ylabel("NLL（波の区域。低いほど良い）", color=INK_2, fontsize=9)
    ax.set_title(f"{PJ[period]}　ノイズ0.02・元の組・4歩先の驚き　学習シード{seed}", loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[str(HERE / "runs" / "2b_B_seed0"), str(HERE / "runs" / "2b_B_seed1")])
    ap.add_argument("--quick", type=int, default=0, help="1 なら試運転（対照を各2点、θ を2点）。判定には使わない")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Physio2.sigma_min = NOISE
    wcfg = W.WorldConfig(noise=NOISE)
    controls = {k: v[::3] for k, v in CONTROLS.items()} if args.quick else CONTROLS
    thetas = (2.0, 8.0) if args.quick else THETAS
    out = HERE / "runs" / ("smoke_verify_b" if args.quick else "verify_b")
    out.mkdir(parents=True, exist_ok=True)
    res = dict(world=wcfg.to_dict(), sigma_min=NOISE, pairs=[list(p) for p in PAIRS_ORIGINAL], seeds={})

    dnaless = AttentionCell(dna=False).to(dev).eval()
    refs = {}
    for period in ("grow", "adult"):
        refs[f"dnaless_B_{period}"] = rp(dnaless, {"kind": "trigger"}, wcfg, period, dev, ahead=4)
        refs[f"one_step_{period}"] = rp(dnaless, {"kind": "trigger"}, wcfg, period, dev, ahead=1)
    res["references"] = refs
    print("参考の点：済み", flush=True)

    all_pass = True
    for run in map(Path, args.runs):
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        seed = cfg["seed"]
        assert cfg["physio"]["surprise_ahead"] == 4 and cfg["physio"]["sigma_min"] == NOISE and cfg["world"]["noise"] == NOISE
        assert cfg["pairs"] == "original" and cfg["visible_only"] == 0 and cfg["firing"] == "trigger"
        model = AttentionCell(dna=True).to(dev).eval()
        model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
        r = dict(run=run.name, skipped_updates=cfg.get("skipped_updates"))
        for period in ("grow", "adult"):
            r[f"B_{period}"] = rp(model, {"kind": "trigger"}, wcfg, period, dev)
            with hide_neighbors():
                r[f"B_{period}_hide_nbr"] = rp(model, {"kind": "trigger"}, wcfg, period, dev)
            r[f"B_curve_{period}"] = [dict(theta=th, **{k: v for k, v in rp(model, {"kind": "trigger"}, wcfg, period, dev, theta=th).items()
                                                          if k in ("fires", "quality")}) for th in thetas]
        ctrl = []
        for fam, pols in controls.items():
            for pol in pols:
                for wname, cell in (("DNA なし", dnaless), ("B の DNA", model)):
                    for period in ("grow", "adult"):
                        q = rp(cell, pol, wcfg, period, dev)
                        ctrl.append(dict(family=fam, policy=pol, writer=wname, period=period, fires=q["fires"], quality=q["quality"]))
        r["controls"] = ctrl
        verd = {}
        for period in ("grow", "adult"):
            b = r[f"B_{period}"]
            for fam in controls:
                env = envelope([(p["fires"], p["quality"]) for p in ctrl if p["period"] == period and p["family"] == fam])
                v, qc = compare((b["fires"], b["quality"]), env)
                verd[f"{period}_{fam}"] = dict(verdict=v, fires=b["fires"], nll=b["quality"], control_nll=qc)
            plot(period, seed, [p for p in ctrl if p["period"] == period],
                 sorted((c["fires"], c["quality"]) for c in r[f"B_curve_{period}"]), (b["fires"], b["quality"]),
                 {"DNA なしの B": (refs[f"dnaless_B_{period}"]["fires"], refs[f"dnaless_B_{period}"]["quality"]),
                  "1歩先の驚き（元の設計、DNA なし）": (refs[f"one_step_{period}"]["fires"], refs[f"one_step_{period}"]["quality"])},
                 out / f"pareto_{period}_seed{seed}.png")
        r["verdicts"] = verd
        r["passed"] = all(v["verdict"] == "上回る" for v in verd.values())
        all_pass &= r["passed"]
        res["seeds"][str(seed)] = r
        print(f"学習シード {seed}：{'上回る' if r['passed'] else '上回らない'}", flush=True)
    res["final"] = "合格" if all_pass and not args.quick else ("（試運転）" if args.quick else "不合格")
    res["s2_10"] = check_s2_10(dev)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 候補B を本物のセルで確かめる（DESIGN.md 14章）", "",
         "世界：ノイズ0.02、σ の下限0.02、大人の時期の組 12→24・12→20・24→36・24→12、学習の変化は見える組に絞らない。驚きは4歩先。評価の乱数1000。", "",
         f"**事前に決めた合否：{res['final']}**", "",
         "## 合否（B の学んだ DNA、θ=4）", "", "| 学習シード | 時期 | 発火数（1セル） | NLL | 時計（同じ発火数） | サイコロ（同じ発火数） |", "|---|---|---:|---:|---|---|"]
    for seed, r in res["seeds"].items():
        for period in ("grow", "adult"):
            c1, c2 = r["verdicts"][f"{period}_時計"], r["verdicts"][f"{period}_サイコロ"]
            L.append(f"| {seed} | {PJ[period]} | {c1['fires']:.2f} | {c1['nll']:.3f} | {c1['verdict']}（{c1['control_nll']:.3f}） | "
                     f"{c2['verdict']}（{c2['control_nll']:.3f}） |")
    L += ["", "## 予算を振った曲線（同じ DNA、評価だけ θ を変えた）", "", "| 学習シード | 時期 | θ | 発火数 | NLL |", "|---|---|---:|---:|---:|"]
    for seed, r in res["seeds"].items():
        for period in ("grow", "adult"):
            for c in r[f"B_curve_{period}"]:
                L.append(f"| {seed} | {PJ[period]} | {c['theta']:g} | {c['fires']:.2f} | {c['quality']:.3f} |")
    L += ["", "## 組ごとの結果（大人の時期）", "", "| 学習シード | 驚き | 組 | 発火数 | 学び直しの NLL | 10歩以内の反応 | 誤差が戻った | あきらめ |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    rows = [(seed, "B（学んだ DNA）", r["B_adult"]) for seed, r in res["seeds"].items()]
    rows += [("—", "DNA なしの B", refs["dnaless_B_adult"]), ("—", "1歩先の驚き（DNA なし）", refs["one_step_adult"])]
    for seed, who, a in rows:
        for pair, d in a["per_pair"].items():
            L.append(f"| {seed} | {who} | {pair} | {d['fires']:.2f} | {d['quality']:.3f} | {d['react_cols']:.2f} | {d['err_back']:.2f} | {d['gaveup']:.2f} |")
    L += ["", "## 大人の時期の中身と、近所を隠す診断", "",
          "| 学習シード | 驚き | 落ち着いた発火率（波／ノイズ） | 反応 | 戻り | あきらめ | 学び直しの NLL（近所を隠すと） | 育つ時期の NLL（近所を隠すと） |",
          "|---|---|---|---:|---:|---:|---|---|"]
    for seed, r in res["seeds"].items():
        a, g = r["B_adult"], r["B_grow"]
        L.append(f"| {seed} | B（学んだ DNA） | {a['steady_fire_rate_wave']:.4f}／{a['steady_fire_rate_noise']:.4f} | {a['react_cols']:.2f} | {a['err_back']:.2f} | "
                 f"{a['gaveup']:.2f} | {a['quality']:.3f}（{r['B_adult_hide_nbr']['quality']:.3f}） | {g['quality']:.3f}（{r['B_grow_hide_nbr']['quality']:.3f}） |")
    for who, key in (("DNA なしの B", "dnaless_B"), ("1歩先の驚き（DNA なし）", "one_step")):
        a, g = refs[f"{key}_adult"], refs[f"{key}_grow"]
        L.append(f"| — | {who} | {a['steady_fire_rate_wave']:.4f}／{a['steady_fire_rate_noise']:.4f} | {a['react_cols']:.2f} | {a['err_back']:.2f} | "
                 f"{a['gaveup']:.2f} | {a['quality']:.3f} | {g['quality']:.3f} |")
    s = res["s2_10"]
    L += ["", "## S2-10 の確かめ（DNA なしの密な書き換え、ノイズ0.02、2a の元の数字）", "",
          f"判定：{'満たす' if s['passed'] else '満たさない'}　" + "、".join(f"{k} {'○' if v else '×'}" for k, v in s["checks"].items()), "",
          "値：" + "、".join(f"{k} {v:.4f}" for k, v in s["values"].items()), "",
          "図：`pareto_grow_seed<シード>.png`、`pareto_adult_seed<シード>.png`", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
