"""
Step 1 の卒業条件を判定する（CLAUDE.md「Step 1 の卒業条件」）。

    .venv/Scripts/python.exe step1_concentration/graduation.py

卒業条件：シード4本中3本以上で、「眠って傷で起きて治す花」の重みが得られる。
重みは「最後の重み」か「良い重みの前線（best/front.json）」のどちらでもよい。報告では分けて書く。

「眠って傷で起きて治す花」の数字（案。すべて、ずれ0.1の評価で判定）：
    維持期の発火 < 10回/ステップ、成長後の平均ロス ≤ 0.0113（サイコロ0.1）、治った割合 ≥ 0.60（サイコロ0.1）、nan なし
最後の重みは eval_j0.1/summary.json、前線は best/front.json（検証はずれ0.1）で見る。
途中の重みの評価（eval_iterN_j*）は、前線ではないので数えない。
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_MAINT_FPS = 10.0
MAX_AFTER_LOSS = 0.0113
MIN_HEAL = 0.60


def ok(maint_fps, after_loss, heal):
    vals = (maint_fps, after_loss, heal)
    if any(v is None or not math.isfinite(v) for v in vals):
        return False
    return maint_fps < MAX_MAINT_FPS and after_loss <= MAX_AFTER_LOSS and heal >= MIN_HEAL


def final_metrics(s):
    m, r = s["maintain"], s["recover"]
    after = s.get("after_loss", (m["loss_mean"] * 200 + r["loss_mean"] * 400) / 600)
    heal = s.get("heal_fraction")
    if heal is None:
        heal = (s["damage_loss_peak"] - r["loss_end"]) / max(s["damage_loss_peak"] - m["loss_mean"], 1e-9)
    return m["fires_per_step"], after, heal


def judge_run(d):
    res = dict(final=None, front=None)
    p = d / "eval_j0.1" / "summary.json"
    if p.exists():
        fps, after, heal = final_metrics(json.loads(p.read_text(encoding="utf-8")))
        res["final"] = dict(passed=ok(fps, after, heal), fps=fps, after=after, heal=heal)
    p = d / "best" / "front.json"
    if p.exists():
        good = [a for a in json.loads(p.read_text(encoding="utf-8")) if ok(a["maint_fps"], a["after_loss"], a["heal"])]
        res["front"] = dict(passed=bool(good), best=min(good, key=lambda a: a["after_loss"]) if good else None)
    return res


def main():
    groups = defaultdict(list)
    for d in sorted((HERE / "runs").iterdir()):
        cfg_p = d / "config.json"
        if not cfg_p.exists() or d.name.startswith("smoke"):
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        key = re.sub(r"^seed\d+_", "", d.name)
        groups[key].append((cfg.get("seed"), d.name, judge_run(d)))

    print(f"基準（ずれ0.1）：維持期の発火 < {MAX_MAINT_FPS:g}/ステップ、成長後ロス ≤ {MAX_AFTER_LOSS}、治った割合 ≥ {MIN_HEAL}")
    for key, runs in groups.items():
        passed = [r for r in runs if (r[2]["final"] and r[2]["final"]["passed"]) or (r[2]["front"] and r[2]["front"]["passed"])]
        print(f"\n== {key}：{len(passed)}/{len(runs)} 本で得られた")
        for seed, name, r in sorted(runs, key=lambda t: t[0]):
            f = r["final"]
            fs = ("—（評価なし）" if f is None else
                  f"{'○' if f['passed'] else '×'}（成長後 {f['after']:.4f}、維持期 {f['fps']:.1f}/ステップ、治った {f['heal']:.2f}）")
            b = r["front"]
            bs = ("—（前線なし）" if b is None else
                  ("×" if not b["passed"] else
                   f"○ iter{b['best']['iter']}（成長後 {b['best']['after_loss']:.4f}、維持期 {b['best']['maint_fps']:.1f}/ステップ、"
                   f"治った {b['best']['heal']:.2f}、総発火 {b['best']['total_fires']:,}）"))
            print(f"  シード{seed}：最後の重み {fs}")
            print(f"          前線     {bs}")


if __name__ == "__main__":
    main()
