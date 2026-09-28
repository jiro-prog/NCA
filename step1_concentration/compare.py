"""
Step 1 の比べ方：ヌル対照（baseline/runs）と濃度発火（step1_concentration/runs）の評価を集めて、
表とパレート図を作る。CLAUDE.md の原則6（比ではなく「ロス vs 発火数」の曲線で、ヌル対照と比べる）。

    .venv/Scripts/python.exe step1_concentration/compare.py
    → step1_concentration/report/compare.md, pareto.png

図は2枚：
- 全体：横軸＝800ステップの総発火数、縦軸＝成長後（200〜800）の平均ロス
- 傷：  横軸＝回復期（400〜800）の発火数、縦軸＝回復期の最後（800）のロス
どちらも左下ほど良い。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common import viz  # noqa: E402


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# EXPERIMENTS.md で使っている Run の呼び名（図のラベルを短くするため）
RUN_NAMES = {
    "seed0_lam0_it2000": "λ0 ずれなし学習",
    "seed0_lam0_tj0.3_noc_it2000": "Run A",
    "seed0_lam0.01_noc": "Run B",
    "seed0_lam0.01_tj0.3_db0": "Run C",
    "seed0_lam0_tj0.3_seec_it2000": "Run D",
    "seed0_lam0.01_tj0.3_db0_mean": "Run E",
    "seed1_lam0_tj0.3_noc_it2000": "Run A（シード1）",
    "seed1_lam0_tj0.3_seec_it2000": "Run D（シード1）",
}


def collect():
    items = []
    for d in sorted((ROOT / "baseline" / "runs").iterdir()):
        if not (d / "config.json").exists() or d.name.startswith("smoke"):
            continue
        cfg = load(d / "config.json")
        pol = cfg["policy"]
        if pol.startswith("dice"):
            group, label = "dice", f"サイコロ{cfg['fire_rate']:g}"
        else:
            group, label = "period", f"周期{cfg['period']}"
        if (d / "eval" / "summary.json").exists():
            items.append(dict(group=group, label=label, key=cfg.get("fire_rate") if group == "dice" else 0,
                              run=d.name, s=load(d / "eval" / "summary.json")))
        if pol == "dice0.5" and (d / "eval_timerstop" / "summary.json").exists():
            items.append(dict(group="timerstop", label="時限停止", key=0, run=d.name + "/timerstop",
                              s=load(d / "eval_timerstop" / "summary.json")))
    runs = HERE / "runs"
    if runs.exists():
        for d in sorted(runs.iterdir()):
            if not (d / "config.json").exists() or d.name.startswith("smoke"):
                continue
            cfg = load(d / "config.json")
            # 最後の重みの評価（eval_j*）と、途中の重みの評価（eval_iterN_j*）
            for ev in sorted(list(d.glob("eval_j*")) + list(d.glob("eval_iter*_j*"))):
                if not (ev / "summary.json").exists():
                    continue
                s = load(ev / "summary.json")
                jp = s.get("jitter_p", 0.0)
                ckpt = ev.name.split("_")[1] if ev.name.startswith("eval_iter") else None
                tj = cfg.get("train_jitter", 0.0)
                old = "see_c" not in cfg        # 2026-09-28 の修正より前の run：濃度を入力に入れ、ずれにバグがあった
                ph = cfg.get("physio", {})
                db = ph.get("d_bias", 2.0)
                cp = ph.get("coupling", "sum")
                label = (f"濃度 λ={cfg['lam']:g}" + ("" if cfg.get("see_c", True) else " 濃度入力なし")
                         + (f" d_bias={db:g}" if db != 2.0 else "")
                         + ({"mean": " イベント平均", "cap": f" イベント上限{ph.get('coupling_cap')}"}.get(cp, ""))
                         + (f" 学習ずれ〜{tj:g}" + ("（バグのあるずれ）" if old else "") if tj else "")
                         + (f" @{ckpt}" if ckpt else "")
                         + (f" 評価ずれ{jp:g}" if jp else ""))
                group = ("ckpt" if ckpt else "step1") + ("_jitter" if jp else "")
                name = RUN_NAMES.get(d.name)
                if name:
                    label = f"{name}：{label}"
                short = ((name or label.replace("濃度 ", "").split(" 評価ずれ")[0]) + (f" @{ckpt}" if ckpt and name else "")
                         if jp == 0 else f"ずれ{jp:g}")
                items.append(dict(group=group, label=label, short=short,
                                  key=cfg["lam"], run=f"{d.name}/{ev.name}", model=d.name + (ckpt or ""), s=s))
    return items


def metrics(s):
    m, r = s["maintain"], s["recover"]
    after = (m["loss_mean"] * 200 + r["loss_mean"] * 400) / 600
    hit = s["recover_first_hit"]
    # 治った割合：（傷の直後の最悪 − 最後）÷（傷の直後の最悪 − 傷の前）。古い summary にはないので計算する
    heal = s.get("heal_fraction")
    if heal is None:
        heal = (s["damage_loss_peak"] - r["loss_end"]) / max(s["damage_loss_peak"] - m["loss_mean"], 1e-9)
    return dict(
        total_fires=s["total_fires"], after_loss=after,
        grow_end=s["grow"]["loss_end"], maint_fps=m["fires_per_step"], maint_loss=m["loss_mean"],
        rec_fires=r["fires_total"], rec_end=r["loss_end"],
        r005=hit.get("0.005"), r002=hit.get("0.002"), heal=heal,
        birth_share=s.get("birth_share"),       # 誕生の発火（固定のルール）の割合。数えていない run は None
    )


def fmt_hit(h):
    return "—" if h is None else f"{h['steps']} / {h['fires']:,}"


def table(items):
    lines = ["| 名前 | 総発火 | うち誕生 | 成長期末ロス | 維持期 発火/ステップ | 維持期 平均ロス | 成長後 平均ロス | "
             "傷→0.005（ステップ / 発火） | 傷→0.002（ステップ / 発火） | 回復期 発火 | 800のロス | 治った割合 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for it in items:
        k = metrics(it["s"])
        birth = "—" if k["birth_share"] is None else f"{k['birth_share']:.1%}"
        lines.append(f"| {it['label']} | {k['total_fires']:,} | {birth} | {k['grow_end']:.5f} | {k['maint_fps']:.1f} | "
                     f"{k['maint_loss']:.5f} | {k['after_loss']:.5f} | {fmt_hit(k['r005'])} | {fmt_hit(k['r002'])} | "
                     f"{k['rec_fires']:,} | {k['rec_end']:.5f} | {k['heal']:.2f} |")
    return "\n".join(lines)


FAILED_LOSS = 0.05   # 成長後の平均ロスがこれ以上（空のキャンバスは約0.08）なら、花になっていないので図から外す


def plot(items, path):
    items = [it for it in items if metrics(it["s"])["after_loss"] < FAILED_LOSS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    style = {   # 色は3つまで（散布図で見分けられる上限）。対照は形と文字で見分ける
        "dice": dict(color=viz.SERIES[0], marker="o", name="サイコロ（発火率を変えて学習し直したもの）"),
        "step1": dict(color=viz.SERIES[1], marker="D", name="濃度発火（Step 1）"),
        "step1_jitter": dict(color=viz.SERIES[1], marker="D", name="濃度発火・タイミングのずれあり", hollow=True),
        "ckpt": dict(color=viz.SERIES[1], marker="*", name="濃度発火・途中の重み", size=110),
        "ckpt_jitter": dict(color=viz.SERIES[1], marker="*", name="途中の重み・ずれあり", hollow=True, size=110),
        "period": dict(color=viz.SERIES[2], marker="s", name="一定周期の時計"),
        "timerstop": dict(color=viz.SERIES[2], marker="X", name="時限停止（200で発火を止める）"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=viz.SURFACE)
    views = [("total_fires", "after_loss", "全体：総発火数（800ステップ）", "成長後（200〜800）の平均ロス"),
             ("rec_fires", "rec_end", "傷：回復期（400〜800）の発火数", "800ステップ目のロス")]
    for ax, (xk, yk, xt, yt) in zip(axes, views):
        ax.set_facecolor(viz.SURFACE)
        # 同じモデルの「ずれなし」と「ずれあり」を細い線でつなぐ（ずれでどれだけ動くか）
        for it in [i for i in items if i["group"] in ("step1", "ckpt")]:
            for jt in [j for j in items if j["group"] == it["group"] + "_jitter" and j.get("model") == it["model"]]:
                a, b = metrics(it["s"]), metrics(jt["s"])
                ax.plot([a[xk], b[xk]], [a[yk], b[yk]], color=viz.AXIS, linewidth=0.8, zorder=0)
        for g, st in style.items():
            pts = sorted([it for it in items if it["group"] == g], key=lambda it: it["key"])
            if not pts:
                continue
            xs = [metrics(it["s"])[xk] for it in pts]
            ys = [metrics(it["s"])[yk] for it in pts]
            if g == "dice" and len(pts) > 1:
                ax.plot(xs, ys, color=st["color"], linewidth=1.5, zorder=1)
            face = viz.SURFACE if st.get("hollow") else st["color"]
            ax.scatter(xs, ys, s=st.get("size", 48), marker=st["marker"], facecolor=face, edgecolor=st["color"],
                       linewidth=1.5, zorder=3, label=st["name"])
            for x, y, it in zip(xs, ys, pts):
                ax.annotate(it.get("short", it["label"]), (x, y), xytext=(5, 4), textcoords="offset points",
                            fontsize=7 if g.endswith("_jitter") else 8, color=viz.INK_2)
        ax.set_yscale("log")
        ax.set_xlabel(xt, color=viz.INK_2, fontsize=9)
        ax.set_title(yt, loc="left", fontsize=10, color=viz.INK)
        ax.grid(True, color=viz.GRID, linewidth=0.8)
        ax.tick_params(which="both", colors=viz.MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        for s in ax.spines.values():
            s.set_color(viz.AXIS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Step 1：ロス vs 発火数（左下ほど良い）", x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=110, facecolor=viz.SURFACE)
    plt.close(fig)


def plot_low_end(items, path, max_fires=200_000):
    """低発火の端だけを拡大した図。花にならなかったもの（成長後ロス ≥ FAILED_LOSS）も × で載せる"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    pts = [it for it in items if metrics(it["s"])["total_fires"] <= max_fires
           and it["group"] in ("dice", "timerstop", "ckpt", "ckpt_jitter", "step1", "step1_jitter")]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=viz.SURFACE)
    ax.set_facecolor(viz.SURFACE)
    dice = sorted([it for it in pts if it["group"] == "dice" and metrics(it["s"])["after_loss"] < FAILED_LOSS],
                  key=lambda it: metrics(it["s"])["total_fires"])
    if len(dice) > 1:
        ax.plot([metrics(it["s"])["total_fires"] for it in dice], [metrics(it["s"])["after_loss"] for it in dice],
                color=viz.SERIES[0], linewidth=1.5, zorder=1)
    for it in pts:
        k = metrics(it["s"])
        failed = k["after_loss"] >= FAILED_LOSS
        g = it["group"]
        color = viz.SERIES[0] if g == "dice" else (viz.SERIES[2] if g == "timerstop" else viz.SERIES[1])
        marker = "x" if failed else {"dice": "o", "timerstop": "X", "ckpt": "*", "ckpt_jitter": "*",
                                     "step1": "D", "step1_jitter": "D"}[g]
        hollow = g.endswith("_jitter") and not failed
        ax.scatter([k["total_fires"]], [k["after_loss"]], s=110 if marker == "*" else 48, marker=marker,
                   facecolor=viz.SURFACE if hollow else color, edgecolor=color, color=color, linewidth=1.5, zorder=3)
        text = it.get("short", it["label"]) + ("（花にならず）" if failed else "") + f"  治った割合 {k['heal']:.2f}"
        ax.annotate(text, (k["total_fires"], k["after_loss"]), xytext=(6, 3), textcoords="offset points",
                    fontsize=7.5, color=viz.INK_2)
    ax.set_yscale("log")
    ax.set_xlim(0, max_fires)
    ax.set_xlabel("総発火数（800ステップ）", color=viz.INK_2, fontsize=9)
    ax.set_title("低発火の端：成長後（200〜800）の平均ロス（左下ほど良い。× は花にならなかったもの）",
                 loc="left", fontsize=10, color=viz.INK)
    ax.grid(True, color=viz.GRID, linewidth=0.8)
    ax.tick_params(which="both", colors=viz.MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    for s in ax.spines.values():
        s.set_color(viz.AXIS)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=viz.SURFACE)
    plt.close(fig)


def main():
    items = collect()
    out = HERE / "report"
    out.mkdir(exist_ok=True)
    md = ["# Step 1 の比較", "", "`step1_concentration/compare.py` が作る表。数字は各 run の summary.json から。",
          "「傷→0.005」は、傷のあとロスが 0.005 以下に戻るまでのステップ数と、そこまでの発火数。",
          f"図（pareto.png）には、成長後の平均ロスが {FAILED_LOSS} 未満のもの（花になったもの）だけを載せる。"
          "空のキャンバスのロスは約0.08。", "",
          table(items), ""]
    (out / "compare.md").write_text("\n".join(md), encoding="utf-8")
    plot(items, out / "pareto.png")
    plot_low_end(items, out / "low_fire_end.png")
    print("\n".join(md))
    print("図:", out / "pareto.png")


if __name__ == "__main__":
    main()
