"""
候補B を本物のセルに入れたときの、実装の確かめ（DESIGN.md 14章「実装の確かめ」。学習の前に。通らなければ止める）。

    .venv/Scripts/python.exe step2_attention/check_b_impl.py

(a) 驚きの信号が1歩先（これまでどおり）なら、DNA なしの曲線（評価の乱数1000）が、runs/dnaless_curves_now/summary.json と発火数・NLL で完全に一致する
(b) 4歩先の DNA なしの本物のセル（ノイズ0.02、σ の下限0.02、今の組）が、13章の模擬 precheck_b.run_period_b("z4") と、両方の時期で完全に一致する
出力：runs/check_b_impl/summary.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "precheck"))

import torch  # noqa: E402

import world as W  # noqa: E402
from dnaless_curves import FAMILIES, curves  # noqa: E402
from model import AttentionCell, Physio2  # noqa: E402
from protocol import run_period  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = HERE / "runs" / "check_b_impl"
    out.mkdir(parents=True, exist_ok=True)
    res = {}

    # (a) 1歩先はこれまでと同じ
    assert Physio2.surprise_ahead == 1 and Physio2.sigma_min == 0.005
    ref = json.loads((HERE / "runs" / "dnaless_curves_now" / "summary.json").read_text(encoding="utf-8"))["seeds"]["1000"]["points"]
    new = curves(W.WorldConfig(), 1000, dev, FAMILIES)
    key = lambda p: (p["family"], p["policy"], p["period"])
    ref_d = {key(p): p for p in ref}
    diffs = []
    for p in new:
        r = ref_d[key(p)]
        diffs.append(dict(point=" ".join(key(p)), fires=(r["fires"], p["fires"]), quality=(r["quality"], p["quality"]),
                          same=r["fires"] == p["fires"] and r["quality"] == p["quality"]))
    res["a_one_step_unchanged"] = dict(same=all(d["same"] for d in diffs), n=len(diffs), diffs=[d for d in diffs if not d["same"]])
    print(f"(a) 1歩先はこれまでと同じ：{res['a_one_step_unchanged']['same']}（{len(diffs)} 点）", flush=True)

    # (b) 4歩先の本物のセル ＝ 13章の模擬
    import precheck_b                                          # 模擬（記録のまま。書き換えない）
    Physio2.sigma_min = 0.02
    wcfg = W.WorldConfig(noise=0.02)
    cons = {}
    for period in ("grow", "adult"):
        Physio2.surprise_ahead = 4
        cell = AttentionCell(dna=False).to(dev).eval()
        a = run_period(cell, {"kind": "trigger"}, wcfg, period, eval_seed=1000, device=dev)
        b = precheck_b.run_period_b("z4", wcfg, period, 1000, dev)
        Physio2.surprise_ahead = 1
        same = a["fires"] == b["fires"] and a["quality"] == b["quality"]
        close = abs(a["fires"] - b["fires"]) < 1e-6 and abs(a["quality"] - b["quality"]) < 1e-5
        cons[period] = dict(real_cell=(a["fires"], a["quality"]), precheck=(b["fires"], b["quality"]), same=same, close=close)
        if period == "adult":
            cons[period].update(real=dict(react=a["react_cols"], back=a["err_back"], gaveup=a["gaveup"]),
                                pre=dict(react=b["react_cols"], back=b["err_back"], gaveup=b["gaveup"]))
    res["b_four_step_matches_precheck"] = cons
    print("(b) 4歩先の本物のセル ＝ 模擬：", json.dumps(cons, ensure_ascii=False), flush=True)
    res["passed"] = res["a_one_step_unchanged"]["same"] and all(c["close"] for c in cons.values())
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("実装の確かめ：", "通った" if res["passed"] else "通らない（学習の前に止める）", flush=True)


if __name__ == "__main__":
    main()
