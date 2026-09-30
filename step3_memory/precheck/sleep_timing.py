"""
いつ寝るかを決める模擬（step3_memory/DESIGN.md 9.7。案と読み方は走らせる前に書いた。2026-10-01、そうの判断）。

    .venv/Scripts/python.exe step3_memory/precheck/sleep_timing.py [--device cuda] [--smoke 1]

睡眠の段（③：長い痕跡で係数を解き直し、σ₄ を1歩先の σ と係数から置き直す。γ_L=0.99）の寝る規則だけを変えて比べる：
  (い) 静かな時期に1回だけ（最後の驚きの発火から100ステップ目）
  (ろ) 静かな間は100ステップごと
  (は) 間隔を100、200、400…と伸ばす（最後の驚きの発火から100、300、700、1500…ステップ目）。上限なし
判定は実数（LESSONS.md 進め方9）：8つの条件（始め方2×評価の乱数2×組2）のすべてで (は) の見逃しの数 ≤ (ろ)＋100、
かつ (は) の書き込み（驚き＋睡眠、全セルの1セル・1切り替えあたり）≤ (い)＋1.5 なら (は)。そうでなければ (い)。
出力：step3_memory/runs/precheck_sleep/timing.md、timing.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import headroom as H  # noqa: E402
import sleep as S  # noqa: E402
from headroom import W  # noqa: E402

MISS_MARGIN = 100
WRITE_MARGIN = 1.5


def variants():
    base = dict(kind="both", g=0.99)
    return [dict(base, name="(い) 1回だけ"), dict(base, name="(ろ) 100ごと", periodic=True), dict(base, name="(は) 2倍ずつ", doubling=True)]


def measure(res, start, gi):
    sl = slice(gi * H.N_PER, (gi + 1) * H.N_PER)
    first = 1 if start == "adult" else 0
    wave = res["wave"]
    r = res["react"][first:, sl][:, :, wave]                                     # (判定する切り替え, 世界, 波の区域のセル)
    miss = ~r
    w_all = (res["fires_ph"][first:, sl] + res["sleeps_ph"][first:, sl])          # (切り替え, 世界, H, W)
    return dict(
        n_miss=int(miss.sum()), ever=int(miss.any(0).sum()), blind_end=int(miss[-1].sum()), n_cells=int(r.shape[1] * r.shape[2]),
        writes_all=float(w_all.mean()), sleeps_all=float(res["sleeps_ph"][first:, sl].mean()),
        writes_wave=float(w_all[:, :, wave].mean()), fires_wave=float(res["fires_ph"][first:, sl][:, :, wave].mean()),
        sleeps_wave=float(res["sleeps_ph"][first:, sl][:, :, wave].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    seeds = H.EVAL_SEEDS
    out_dir = H.ROOT / "step3_memory" / "runs" / "precheck_sleep"
    tag = "timing"
    if args.smoke:
        H.DWELL = 160
        seeds = (300,)
        out_dir = H.ROOT / "step3_memory" / "runs" / "smoke_precheck_sleep"
        out_dir.mkdir(parents=True, exist_ok=True)
    wcfg = W.WorldConfig(noise=0.02)
    rows, allres = [], {}
    for start in S.STARTS:
        for seed in seeds:
            for v in variants():
                res = S.run(v, wcfg, seed, args.device, start)
                allres[f"{start}|{seed}|{v['name']}"] = dict(groups=S.summarize(res, wcfg))
                for gi, grp in enumerate(H.GROUPS):
                    m = measure(res, start, gi)
                    rows.append(dict(start=start, seed=seed, group="/".join(f"{p:g}" for p in grp), var=v["name"], **m))
                print(f"{start} seed={seed} {v['name']}: {res['elapsed']:.1f}s", flush=True)
    names = [v["name"] for v in variants()]
    gorder = ["/".join(f"{p:g}" for p in grp) for grp in H.GROUPS]
    conds = sorted({(r["start"], r["seed"], r["group"]) for r in rows}, key=lambda c: (S.STARTS.index(c[0]), c[1], gorder.index(c[2])))
    get = lambda c, n: [r for r in rows if (r["start"], r["seed"], r["group"]) == c and r["var"] == n][0]
    L = ["# いつ寝るかを決める模擬（DESIGN.md 9.7。読み方は走らせる前に決めた）", "",
         "見逃しの数＝波の区域のセル×判定する切り替えのうち、窓の中で一度も驚いて発火しなかった数（条件ごと、実数）。",
         "書き込み＝驚きの発火＋睡眠の書き込み。全セル（ノイズの区域を含む）の1セル・1切り替え（滞在600ステップ）あたり", "",
         "| 始め方 | 評価の乱数 | 組 | " + " | ".join(f"{n}：見逃しの数（見逃したことのあるセル／見逃したまま）" for n in names) + " |",
         "|---|---:|---|" + "---:|" * len(names)]
    for c in conds:
        L.append(f"| {'大人' if c[0] == 'adult' else '育ち'} | {c[1]} | {c[2]} | " +
                 " | ".join(f"{get(c, n)['n_miss']}（{get(c, n)['ever']}／{get(c, n)['blind_end']}）" for n in names) + " |")
    L += ["", "| 版 | 書き込み（全セル） | うち睡眠（全セル） | 書き込み（波の区域：驚き＋睡眠） | 反応（最小／平均） | 窓の NLL |", "|---|---:|---:|---|---|---:|"]
    W_all = {}
    for n in names:
        rs = [r for r in rows if r["var"] == n]
        W_all[n] = float(np.mean([r["writes_all"] for r in rs]))
        j = S.judge(allres, n, seeds)
        L.append(f"| {n} | {W_all[n]:.2f} | {np.mean([r['sleeps_all'] for r in rs]):.2f} | "
                 f"{np.mean([r['writes_wave'] for r in rs]):.2f}（{np.mean([r['fires_wave'] for r in rs]):.2f}＋{np.mean([r['sleeps_wave'] for r in rs]):.2f}） | "
                 f"{j['react_min']:.2f}／{j['react_mean']:.2f} | {j['quality']:.3f} |")
    i_, ro, ha = names
    miss_ok = all(get(c, ha)["n_miss"] <= get(c, ro)["n_miss"] + MISS_MARGIN for c in conds)
    write_ok = W_all[ha] <= W_all[i_] + WRITE_MARGIN
    worst = max(get(c, ha)["n_miss"] - get(c, ro)["n_miss"] for c in conds)
    L += ["", "## 読み方（DESIGN.md 9.7）", "",
          f"- 見逃しが (ろ) に近いか：(は) − (ろ) の見逃しの数の最大 {worst}（基準 ≤ {MISS_MARGIN}）→ {'満たす' if miss_ok else '満たさない'}",
          f"- 書き込みが (い) に近いか：(は) {W_all[ha]:.2f}、(い) {W_all[i_]:.2f}、差 {W_all[ha] - W_all[i_]:.2f}（基準 ≤ {WRITE_MARGIN}）→ {'満たす' if write_ok else '満たさない'}",
          f"- **判定：{'(は) を採用' if (miss_ok and write_ok) else '(い) を採用し、見逃しを既知の弱さとして記録'}**"]
    (out_dir / f"{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / f"{tag}.json").write_text(json.dumps(dict(rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
