"""
次の手に進む条件（EXPERIMENTS.md で走らせる前に決めたもの）を、学習ログから判定する。

    .venv/Scripts/python.exe step1_concentration/criteria.py runs/<名前> [runs/<名前> ...]

1.「花を覚える前に眠った」：目覚ましの平均（直近20回の移動平均）が0.1を下回った時点で、
   直近100回の見た目のロスの中央値がまだ0.02より大きい（平均だと爆発の跳ね上がりに引っぱられるので中央値。
   2026-09-28、Run F／C' の結果が出る前に平均から直した）
2.「合計で連鎖が戻った」：眠り始めた（目覚ましの移動平均が0.2を下回った）あと、濃度の最大2.5以上が何度も出る
   （ここでは10回以上を「何度も」とする）
あわせて、数値が壊れた（nan）か、区間ごとの状態の絶対値の最大（爆発の目安）も出す。
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(run):
    rows = list(csv.DictReader(open(Path(run) / "train_log.csv", encoding="utf-8")))

    def f(r, k):
        v = r.get(k, "nan")
        return float("nan") if v in ("", "nan") else float(v)
    return [dict(i=int(r["iter"]), rgba=f(r, "rgba_loss"), d=f(r, "d_mean"), c=f(r, "c_max"), x=f(r, "x_absmax"))
            for r in rows]


def rolling(vals, k):
    out = []
    for i in range(len(vals)):
        w = [v for v in vals[max(0, i - k + 1):i + 1] if not math.isnan(v)]
        out.append(sum(w) / len(w) if w else float("nan"))
    return out


def judge(run):
    rows = load(run)
    d = rolling([r["d"] for r in rows], 20)
    res = dict(run=Path(run).name, iters=len(rows))
    nan_at = next((r["i"] for r in rows if math.isnan(r["rgba"])), None)
    windows = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    xmax = {f"{a}-{b - 1}": max((r["x"] for r in rows if a <= r["i"] < b and not math.isnan(r["x"])), default=None)
            for a, b in windows}
    res.update(nan_at=nan_at, xmax=xmax)

    # 1. 花を覚える前に眠った
    k = next((j for j, v in enumerate(d) if v < 0.1), None)
    if k is None:
        res.update(sleep_iter=None, early_sleep=False)
    else:
        w = [r["rgba"] for r in rows[max(0, k - 99):k + 1] if not math.isnan(r["rgba"])]
        mean = sum(w) / len(w)
        med = sorted(w)[len(w) // 2]
        res.update(sleep_iter=rows[k]["i"], loss_mean_before=mean, loss_median_before=med, early_sleep=med > 0.02)

    # 2. 合計で連鎖が戻った
    s = next((j for j, v in enumerate(d) if v < 0.2), None)
    if s is None:
        res.update(sleep_begin=None, cascade_count=0, cascade=False)
    else:
        n = sum(1 for r in rows[s:] if not math.isnan(r["c"]) and r["c"] >= 2.5)
        res.update(sleep_begin=rows[s]["i"], cascade_count=n, cascade=n >= 10)
    return res


def main():
    for run in sys.argv[1:]:
        p = Path(run) if Path(run).exists() else HERE / run
        r = judge(p)
        print(f"== {r['run']}（{r['iters']}回）")
        xm = "  ".join(f"{k}回:{v:.3g}" for k, v in r["xmax"].items() if v is not None)
        print(f"  nan：{r['nan_at'] if r['nan_at'] is not None else 'なし'}　状態の絶対値の最大（区間ごと）：{xm}")
        if r["sleep_iter"] is None:
            print("  1. 花を覚える前に眠った：いいえ（目覚ましの移動平均が0.1を下回らなかった）")
        else:
            print(f"  1. 花を覚える前に眠った：{'はい' if r['early_sleep'] else 'いいえ'}"
                  f"（目覚まし<0.1 になった回 {r['sleep_iter']}、直近100回のロス 中央値 {r['loss_median_before']:.4f}"
                  f"／平均 {r['loss_mean_before']:.4f}）")
        if r["sleep_begin"] is None:
            print("  2. 合計で連鎖が戻った：いいえ（眠り始めなかった）")
        else:
            print(f"  2. 合計で連鎖が戻った：{'はい' if r['cascade'] else 'いいえ'}"
                  f"（眠り始めた回 {r['sleep_begin']} 以降、濃度≥2.5 が {r['cascade_count']} 回）")


if __name__ == "__main__":
    main()
