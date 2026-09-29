"""
候補A（目覚まし）の小さな確かめ（DESIGN.md 12章・7.5。合否は 2026-09-29 に走らせる前に決めた）。

    .venv/Scripts/python.exe step2_attention/small_check.py [--alarm-run runs/2b_alarm_lam1_seed0] [--cand3-run runs/2b_cand3_seed0]

1. 一致の確かめ：DNA なし・候補3（上限20）を予測器0から2000ステップ。落ち着いた誤差（模擬の表12 は1.58倍）
2. 目覚まし（学んだ DNA）の点：育つ時期・大人の時期のそれぞれで（発火数、NLL）
3. 対照の曲線：候補3（K ∈ {1,2,5,10,20,40}。K=1 は元の設計）、サイコロ（p ∈ {0.003,…,0.5}）、時計（k ∈ {2,…,300}）。
   書き換え方は {DNA なし、候補3で学んだ DNA、目覚ましの DNA} のすべてを点にし、下の包みをとる（対照に有利）
4. 7.5 の比べ方で判定：両方の時期で、3つの曲線すべてを「上回る」なら合格。「決められない」があれば λ を1点だけ足す（λ÷3）。それ以外は不合格。
   必ず見る比べ：同じくらいの発火数で、候補3より学び直しが良いか（大人の時期）
5. 近所を隠す診断（目覚まし・候補3 の学んだ DNA、両方の時期）
出力：runs/small_check_<名前>/（summary.md、summary.json、pareto_grow.png、pareto_adult.png）
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
from model import AttentionCell, init_state, irreducible  # noqa: E402
from protocol import compare, envelope, hide_neighbors, run_period  # noqa: E402

FAMILIES = {
    "候補3": [{"kind": "cand3", "K": k} for k in (1, 2, 5, 10, 20, 40)],
    "サイコロ": [{"kind": "dice", "p": p} for p in (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)],
    "時計": [{"kind": "clock", "k": k} for k in (2, 4, 10, 30, 100, 300)],
}


def label(pol):
    return {"cand3": lambda p: f"候補3 K={p['K']}", "dice": lambda p: f"サイコロ p={p['p']}",
            "clock": lambda p: f"時計 k={p['k']}", "alarm": lambda p: "目覚まし"}[pol["kind"]](pol)


@torch.no_grad()
def consistency(dev):
    wcfg = W.WorldConfig()
    g = torch.Generator(device=dev).manual_seed(1000)
    ws = W.reset(wcfg, 12.0, n=4, generator=g, device=dev)
    st = init_state(4, wcfg.size, wcfg.size, dev)
    cell = AttentionCell(dna=False)
    wave = ~W.zone_mask(wcfg, dev)
    e2 = []
    for _ in range(2000):
        st, rec = cell.step(st, W.sense(ws, wcfg, g), policy={"kind": "cand3", "K": 20})
        e2.append(float((rec["e"][:, 0] ** 2)[:, wave].mean()))
        ws = W.advance(ws)
    return float(np.sqrt(np.mean(e2[1600:])) / irreducible(0.005, 12.0))


def plot(period, curves, alarm_pt, cand3_pt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (name, pts) in enumerate(curves.items()):
        env = envelope(pts)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=SERIES[i + 1], alpha=0.35, linewidths=0)
        ax.plot([p[0] for p in env], [p[1] for p in env], color=SERIES[i + 1], linewidth=2, label=f"{name}（下の包み）")
    ax.scatter([alarm_pt[0]], [alarm_pt[1]], s=70, color=SERIES[0], zorder=5, label="目覚まし（λ=1）")
    ax.scatter([cand3_pt[0]], [cand3_pt[1]], s=50, marker="s", color=SERIES[4], zorder=5, label="候補3（学んだ DNA、K=20）")
    ax.set_xscale("log")
    ax.set_xlabel("発火数（1セルあたり、全セル）", color=INK_2, fontsize=9)
    ax.set_ylabel("NLL（波の区域。低いほど良い）", color=INK_2, fontsize=9)
    ax.set_title({"grow": "育つ時期（予測器0から、0〜400）", "adult": "大人の時期（学び直し：新しい波が届いてから300ステップ）"}[period],
                 loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarm-run", default=str(HERE / "runs" / "2b_alarm_lam1_seed0"))
    ap.add_argument("--cand3-run", default=str(HERE / "runs" / "2b_cand3_seed0"))
    ap.add_argument("--eval-seed", type=int, default=1000)
    ap.add_argument("--quick", type=int, default=0, help="1 なら試運転（対照を各2点、書き換え方は DNA なしだけ）。判定には使わない")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    alarm_run, cand3_run = Path(args.alarm_run), Path(args.cand3_run)
    out = HERE / "runs" / f"small_check_{alarm_run.name}"
    out.mkdir(parents=True, exist_ok=True)
    wcfg = W.WorldConfig()

    alarm = AttentionCell(dna=True, alarm=True).to(dev)
    alarm.load_state_dict(torch.load(alarm_run / "model.pt", map_location=dev))
    c3dna = AttentionCell(dna=True).to(dev)
    c3dna.load_state_dict(torch.load(cand3_run / "model.pt", map_location=dev))
    writers = {"DNA なし": AttentionCell(dna=False).to(dev), "候補3の DNA": c3dna, "目覚ましの DNA": alarm}
    families = FAMILIES
    if args.quick:
        writers = {"DNA なし": writers["DNA なし"]}
        families = {k: v[::3] for k, v in FAMILIES.items()}
        out = HERE / "runs" / f"smoke_small_check_{alarm_run.name}"
        out.mkdir(parents=True, exist_ok=True)
    for m in writers.values():
        m.eval()

    res = dict(consistency_cand3_dnaless=consistency(dev))
    print(f"一致の確かめ：DNA なし・候補3（上限20）の落ち着いた誤差 {res['consistency_cand3_dnaless']:.2f}倍（模擬の表12 は1.58倍）", flush=True)

    # 目覚ましの点と、候補3（学んだ DNA、K=20）の点、近所を隠す診断
    for name, model, pol in (("alarm", alarm, {"kind": "alarm"}), ("cand3", c3dna, {"kind": "cand3", "K": 20})):
        for period in ("grow", "adult"):
            res[f"{name}_{period}"] = run_period(model, pol, wcfg, period, eval_seed=args.eval_seed, device=dev)
            with hide_neighbors():
                res[f"{name}_{period}_hide_nbr"] = run_period(model, pol, wcfg, period, eval_seed=args.eval_seed, device=dev)
        print(f"{name}：育つ時期 発火 {res[name + '_grow']['fires']:.1f}・NLL {res[name + '_grow']['quality']:.3f}｜"
              f"大人の時期 発火 {res[name + '_adult']['fires']:.1f}・NLL {res[name + '_adult']['quality']:.3f}", flush=True)

    # 対照の曲線
    curves = {"grow": {}, "adult": {}}
    points = []
    for fam, pols in families.items():
        for period in ("grow", "adult"):
            curves[period][fam] = []
        for pol in pols:
            for wname, model in writers.items():
                for period in ("grow", "adult"):
                    r = run_period(model, pol, wcfg, period, eval_seed=args.eval_seed, device=dev)
                    curves[period][fam].append((r["fires"], r["quality"]))
                    points.append(dict(family=fam, policy=label(pol), writer=wname, period=period, fires=r["fires"],
                                       quality=r["quality"], noise_fire_rate=r["noise_fire_rate"]))
        print(f"対照 {fam}：済み", flush=True)
    res["control_points"] = points

    # 7.5 の比べ方で判定
    verdicts = {}
    for period in ("grow", "adult"):
        a = res[f"alarm_{period}"]
        for fam in families:
            v, qc = compare((a["fires"], a["quality"]), envelope(curves[period][fam]))
            verdicts[f"{period}_{fam}"] = dict(verdict=v, alarm_fires=a["fires"], alarm_nll=a["quality"], control_nll=qc)
    res["verdicts"] = verdicts
    vals = [v["verdict"] for v in verdicts.values()]
    if all(v == "上回る" for v in vals):
        final = "合格"
    elif "決められない" in vals:
        final = "決められない → λ を1点だけ足す（λ÷3 = 1/3）"
    else:
        final = "不合格"
    res["final"] = final
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for period in ("grow", "adult"):
        plot(period, curves[period], (res[f"alarm_{period}"]["fires"], res[f"alarm_{period}"]["quality"]),
             (res[f"cand3_{period}"]["fires"], res[f"cand3_{period}"]["quality"]), out / f"pareto_{period}.png")

    pj = {"grow": "育つ時期", "adult": "大人の時期"}
    lines = ["# 候補A（目覚まし）の小さな確かめ", "",
             f"目覚まし：`{alarm_run.name}`、候補3（学んだ DNA）：`{cand3_run.name}`。合否は DESIGN.md 12章・7.5（走らせる前に決めた）。", "",
             f"一致の確かめ：DNA なし・候補3（上限20）の落ち着いた誤差 {res['consistency_cand3_dnaless']:.2f}倍（模擬の表12 は1.58倍）", "",
             "## 必ず見る比べ：同じくらいの発火数で、候補3より学び直しが良いか（大人の時期）", "",
             "| | 発火数（1セル） | 学び直しの NLL | 候補3の曲線のその発火数での NLL | 判定 |", "|---|---:|---:|---:|---|",
             f"| 目覚まし（λ=1） | {verdicts['adult_候補3']['alarm_fires']:.1f} | {verdicts['adult_候補3']['alarm_nll']:.3f} | "
             f"{verdicts['adult_候補3']['control_nll']:.3f} | {verdicts['adult_候補3']['verdict']} |",
             f"| 参考：候補3（学んだ DNA、K=20） | {res['cand3_adult']['fires']:.1f} | {res['cand3_adult']['quality']:.3f} | — | — |", "",
             "## 7.5 の比べ方による判定", "", "| 時期 | 対照 | 目覚ましの発火数 | 目覚ましの NLL | 対照の NLL（同じ発火数） | 判定 |",
             "|---|---|---:|---:|---:|---|"]
    for k, v in verdicts.items():
        period, fam = k.split("_", 1)
        lines.append(f"| {pj[period]} | {fam} | {v['alarm_fires']:.1f} | {v['alarm_nll']:.3f} | {v['control_nll']:.3f} | {v['verdict']} |")
    lines += ["", f"**事前に決めた合否**：{final}", "",
              "## 目覚ましと候補3 の中身（大人の時期）と、近所を隠す診断", "",
              "| | 落ち着いた発火率（波／ノイズ） | 10歩以内の反応 | 誤差が戻った | あきらめ | 学び直しの NLL（近所を隠すと） | 育つ時期の NLL（近所を隠すと） |",
              "|---|---|---:|---:|---:|---|---|"]
    for name, jp in (("alarm", "目覚まし"), ("cand3", "候補3（学んだ DNA）")):
        a, g = res[f"{name}_adult"], res[f"{name}_grow"]
        lines.append(f"| {jp} | {a['steady_fire_rate_wave']:.4f}／{a['steady_fire_rate_noise']:.4f} | {a['react_cols']:.2f} | {a['err_back']:.2f} | "
                     f"{a['gaveup']:.2f} | {a['quality']:.3f}（{res[name + '_adult_hide_nbr']['quality']:.3f}） | "
                     f"{g['quality']:.3f}（{res[name + '_grow_hide_nbr']['quality']:.3f}） |")
    lines += ["", f"目覚ましの d の平均（大人の時期の終わり）：{res['alarm_adult']['d_mean_end']:.3f}（しきい値は4）", "",
              "図：`pareto_grow.png`、`pareto_adult.png`（対照の点と下の包み、目覚まし、候補3）", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
