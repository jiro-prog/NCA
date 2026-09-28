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
            for ev in sorted(d.glob("eval_j*")):
                if not (ev / "summary.json").exists():
                    continue
                s = load(ev / "summary.json")
                jp = s.get("jitter_p", 0.0)
                tj = cfg.get("train_jitter", 0.0)
                old = "see_c" not in cfg        # 2026-09-28 の修正より前の run：濃度を入力に入れ、ずれにバグがあった
                db = cfg.get("physio", {}).get("d_bias", 2.0)
                label = (f"濃度 λ={cfg['lam']:g}" + (" 濃度入力あり" if cfg.get("see_c", True) else "")
                         + (f" d_bias={db:g}" if db != 2.0 else "")
                         + (f" 学習ずれ〜{tj:g}" + ("（バグのあるずれ）" if old else "") if tj else "")
                         + (f" 評価ずれ{jp:g}" if jp else ""))
                items.append(dict(group="step1" if jp == 0 else "step1_jitter", label=label,
                                  short=label.replace("濃度 ", "").split(" 評価ずれ")[0] if jp == 0 else f"ずれ{jp:g}",
                                  key=cfg["lam"], run=f"{d.name}/{ev.name}", model=d.name, s=s))
    return items


def metrics(s):
    m, r = s["maintain"], s["recover"]
    after = (m["loss_mean"] * 200 + r["loss_mean"] * 400) / 600
    hit = s["recover_first_hit"]
    return dict(
        total_fires=s["total_fires"], after_loss=after,
        grow_end=s["grow"]["loss_end"], maint_fps=m["fires_per_step"], maint_loss=m["loss_mean"],
        rec_fires=r["fires_total"], rec_end=r["loss_end"],
        r005=hit.get("0.005"), r002=hit.get("0.002"),
        birth_share=s.get("birth_share"),       # 誕生の発火（固定のルール）の割合。数えていない run は None
    )


def fmt_hit(h):
    return "—" if h is None else f"{h['steps']} / {h['fires']:,}"


def table(items):
    lines = ["| 名前 | 総発火 | うち誕生 | 成長期末ロス | 維持期 発火/ステップ | 維持期 平均ロス | 成長後 平均ロス | "
             "傷→0.005（ステップ / 発火） | 傷→0.002（ステップ / 発火） | 回復期 発火 | 800のロス |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for it in items:
        k = metrics(it["s"])
        birth = "—" if k["birth_share"] is None else f"{k['birth_share']:.1%}"
        lines.append(f"| {it['label']} | {k['total_fires']:,} | {birth} | {k['grow_end']:.5f} | {k['maint_fps']:.1f} | "
                     f"{k['maint_loss']:.5f} | {k['after_loss']:.5f} | {fmt_hit(k['r005'])} | {fmt_hit(k['r002'])} | "
                     f"{k['rec_fires']:,} | {k['rec_end']:.5f} |")
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
        "period": dict(color=viz.SERIES[2], marker="s", name="一定周期の時計"),
        "timerstop": dict(color=viz.SERIES[2], marker="X", name="時限停止（200で発火を止める）"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=viz.SURFACE)
    views = [("total_fires", "after_loss", "全体：総発火数（800ステップ）", "成長後（200〜800）の平均ロス"),
             ("rec_fires", "rec_end", "傷：回復期（400〜800）の発火数", "800ステップ目のロス")]
    for ax, (xk, yk, xt, yt) in zip(axes, views):
        ax.set_facecolor(viz.SURFACE)
        # 同じモデルの「ずれなし」と「ずれあり」を細い線でつなぐ（ずれでどれだけ動くか）
        for it in [i for i in items if i["group"] == "step1"]:
            for jt in [j for j in items if j["group"] == "step1_jitter" and j.get("model") == it["model"]]:
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
            ax.scatter(xs, ys, s=48, marker=st["marker"], facecolor=face, edgecolor=st["color"],
                       linewidth=1.5, zorder=3, label=st["name"])
            for x, y, it in zip(xs, ys, pts):
                ax.annotate(it.get("short", it["label"]), (x, y), xytext=(5, 4), textcoords="offset points",
                            fontsize=7 if g == "step1_jitter" else 8, color=viz.INK_2)
        ax.set_yscale("log")
        ax.set_xlabel(xt, color=viz.INK_2, fontsize=9)
        ax.set_title(yt, loc="left", fontsize=10, color=viz.INK)
        ax.grid(True, color=viz.GRID, linewidth=0.8)
        ax.tick_params(which="both", colors=viz.MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        for s in ax.spines.values():
            s.set_color(viz.AXIS)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Step 1：ロス vs 発火数（左下ほど良い）", x=0.01, ha="left", fontsize=11, color=viz.INK)
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
    print("\n".join(md))
    print("図:", out / "pareto.png")


if __name__ == "__main__":
    main()
