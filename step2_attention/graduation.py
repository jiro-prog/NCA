"""
Step 2 の卒業の条件 (i) の確かめ（DESIGN.md 7.6・16章）。

    .venv/Scripts/python.exe step2_attention/graduation.py [--writers smallcheck] [--device cpu]

(i)：標準の世界で、B の驚き駆動（学んだ DNA）が、育つ時期と大人の時期の両方で、θ の格子 {2,3,4,6,8} のどの点でも時計とサイコロに負けない
（「上回る」か「決められない」）、かつ見出しの予算（θ=4 の点）では上回る。開発のシード2本（2本とも）。封印テストは読み方を確定してから別に回す。
- 重みは runs/2b_B_seed0・seed1 の最後の重み。θ を変えるのは評価だけ（θ ごとの DNA の学び直しはしない。注意書き）
- 評価：標準の世界、評価の乱数1000、組は元の4つ。大人の時期の学び直しは見える3つの組で数え、24→36 は「発火しない」の確かめとして別に出す
- 対照：時計 k ∈ {2,4,10,30,100,300}、サイコロ p ∈ {0.003,…,0.5}。書き換え方は --writers（今は smallcheck＝{DNA なし、比べる B の DNA} の良いほう。
  7.5 の「対照もそれぞれの発火のしかたで学んだ DNA」は、そうの判断を待っている。16章）
- 近所を隠す診断は θ=4 で
出力：runs/graduation_dev_<writers>/（summary.md、summary.json、pareto_<時期>_seed<シード>.png）
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, Physio2  # noqa: E402
from protocol import PAIRS_ORIGINAL, compare, envelope, hide_neighbors, run_period  # noqa: E402

THETAS = (2.0, 3.0, 4.0, 6.0, 8.0)
HEADLINE = 4.0
CONTROLS = {
    "時計": [{"kind": "clock", "k": k} for k in (2, 4, 10, 30, 100, 300)],
    "サイコロ": [{"kind": "dice", "p": p} for p in (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)],
}
PJ = {"grow": "育つ時期", "adult": "大人の時期"}


def at_theta(theta, fn):
    old = Physio2.theta
    Physio2.theta = theta
    try:
        return fn()
    finally:
        Physio2.theta = old


def evaluate_seed(model, dnaless, wcfg, pairs, eval_seed, dev, writers="smallcheck"):
    """1本の B の DNA について、θ の格子の点と対照の曲線を出し、(i) の判定をする"""
    rp = lambda cell, pol, period: run_period(cell, pol, wcfg, period, eval_seed=eval_seed, device=dev, pairs=pairs)
    r = dict(theta={}, controls=[])
    for th in THETAS:
        for period in ("grow", "adult"):
            r["theta"][f"{period}_{th:g}"] = at_theta(th, lambda: rp(model, {"kind": "trigger"}, period))
    for period in ("grow", "adult"):
        with hide_neighbors():
            r[f"hide_nbr_{period}"] = at_theta(HEADLINE, lambda: rp(model, {"kind": "trigger"}, period))
    if writers != "smallcheck":
        raise ValueError("対照の書き換え方は、そうの判断を待っている（DESIGN.md 16章）。今は smallcheck だけ")
    for fam, pols in CONTROLS.items():
        for pol in pols:
            for wname, cell in (("DNA なし", dnaless), ("B の DNA", model)):
                for period in ("grow", "adult"):
                    q = rp(cell, pol, period)
                    r["controls"].append(dict(family=fam, policy=pol, writer=wname, period=period, fires=q["fires"], quality=q["quality"]))
    verd = {}
    ok = True
    for period in ("grow", "adult"):
        envs = {fam: envelope([(p["fires"], p["quality"]) for p in r["controls"] if p["period"] == period and p["family"] == fam]) for fam in CONTROLS}
        for th in THETAS:
            b = r["theta"][f"{period}_{th:g}"]
            for fam, env in envs.items():
                v, qc = compare((b["fires"], b["quality"]), env)
                verd[f"{period}_{th:g}_{fam}"] = dict(verdict=v, fires=b["fires"], nll=b["quality"], control_nll=qc)
                if v == "負け" or (th == HEADLINE and v != "上回る"):
                    ok = False
    r["verdicts"] = verd
    r["passed"] = ok
    return r


def plot(period, seed, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, fam in enumerate(CONTROLS):
        pts = [(p["fires"], p["quality"]) for p in r["controls"] if p["period"] == period and p["family"] == fam]
        env = envelope(pts)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=SERIES[i + 1], alpha=0.35, linewidths=0)
        ax.plot([p[0] for p in env], [p[1] for p in env], color=SERIES[i + 1], linewidth=2, label=f"{fam}（下の包み）")
    th_pts = sorted((r["theta"][f"{period}_{th:g}"]["fires"], r["theta"][f"{period}_{th:g}"]["quality"], th) for th in THETAS)
    ax.plot([p[0] for p in th_pts], [p[1] for p in th_pts], color=SERIES[0], linewidth=1.2, marker="o", markersize=4, label="B（θ の格子）")
    for f, q, th in th_pts:
        ax.annotate(f"θ={th:g}", (f, q), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK_2)
    hb = r["theta"][f"{period}_{HEADLINE:g}"]
    ax.scatter([hb["fires"]], [hb["quality"]], s=70, color=SERIES[0], zorder=5, label="見出しの予算（θ=4）")
    ax.set_xscale("log")
    ax.set_xlabel("発火数（1セルあたり、全セル。大人の時期は見える3つの組の世界）", color=INK_2, fontsize=9)
    ax.set_ylabel("NLL（波の区域。低いほど良い）", color=INK_2, fontsize=9)
    ax.set_title(f"卒業の条件 (i)　{PJ[period]}　学習シード{seed}（対照の書き換え方は仮）", loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[str(HERE / "runs" / "2b_B_seed0"), str(HERE / "runs" / "2b_B_seed1")])
    ap.add_argument("--writers", default="smallcheck")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert Physio2.surprise_ahead == 4 and Physio2.sigma_min == 0.02 and W.WorldConfig().noise == 0.02, "標準の世界で回す（台帳 S2-39）"
    wcfg = W.WorldConfig()
    out = HERE / "runs" / f"graduation_dev_{args.writers}"
    out.mkdir(parents=True, exist_ok=True)
    dnaless = AttentionCell(dna=False).to(dev).eval()
    res = dict(writers=args.writers, thetas=THETAS, headline=HEADLINE, eval_seed=1000, pairs=[list(p) for p in PAIRS_ORIGINAL], seeds={})
    for run in map(Path, args.runs):
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        for k, v in (("noise", 0.02), ("ahead", 4), ("visible_only", 0), ("pairs", "original"), ("firing", "trigger")):
            assert cfg.get(k) == v, (run, k, cfg.get(k))
        model = AttentionCell(dna=True).to(dev).eval()
        model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
        r = evaluate_seed(model, dnaless, wcfg, PAIRS_ORIGINAL, 1000, dev, args.writers)
        res["seeds"][str(cfg["seed"])] = r
        for period in ("grow", "adult"):
            plot(period, cfg["seed"], r, out / f"pareto_{period}_seed{cfg['seed']}.png")
        print(f"学習シード {cfg['seed']}：{'満たす' if r['passed'] else '満たさない'}", flush=True)
    res["passed"] = all(r["passed"] for r in res["seeds"].values())
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# 卒業の条件 (i)：開発のシード（θ の格子）" + ("（仮：対照の書き換え方は小さな確かめのもの。7.5 の卒業の決まりとは違う）" if args.writers == "smallcheck" else ""), "",
         "標準の世界、評価の乱数1000。θ を変えるのは評価だけ（θ=4 で学んだ DNA）。大人の時期の学び直しは見える3つの組の世界で数える。", "",
         f"**判定：{'2本とも満たす' if res['passed'] else '満たさない'}**（θ の格子のどの点でも負けない、かつ θ=4 で上回る。両方の時期）", ""]
    for seed, r in res["seeds"].items():
        L += [f"## 学習シード {seed}：{'満たす' if r['passed'] else '満たさない'}", "",
              "| 時期 | θ | 発火数（1セル） | NLL | 時計（同じ発火数） | サイコロ（同じ発火数） |", "|---|---:|---:|---:|---|---|"]
        for period in ("grow", "adult"):
            for th in THETAS:
                c1, c2 = r["verdicts"][f"{period}_{th:g}_時計"], r["verdicts"][f"{period}_{th:g}_サイコロ"]
                mark = "**" if th == HEADLINE else ""
                L.append(f"| {PJ[period]} | {mark}{th:g}{mark} | {c1['fires']:.2f} | {c1['nll']:.3f} | {c1['verdict']}（{c1['control_nll']:.3f}） | "
                         f"{c2['verdict']}（{c2['control_nll']:.3f}） |")
        L += ["", "24→36 の「発火しない」の確かめ（大人の時期、窓の中で1回でも発火した波の区域のセルの割合）：" +
              "、".join(f"θ={th:g} で {r['theta'][f'adult_{th:g}']['quiet_check']['24→36']['cells_fired_frac'] * 100:.2f}%" for th in THETAS), "",
              f"近所を隠す診断（θ=4）：育つ時期 {r['theta'][f'grow_{HEADLINE:g}']['quality']:.3f} → {r['hide_nbr_grow']['quality']:.3f}、"
              f"大人の時期 {r['theta'][f'adult_{HEADLINE:g}']['quality']:.3f} → {r['hide_nbr_adult']['quality']:.3f}", ""]
    L += ["図：`pareto_<時期>_seed<シード>.png`", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
