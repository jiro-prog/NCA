"""
9.6 の追加の診断の補い：③周期（静かな間は100ステップごとに見直す）で、判定する切り替えを見逃したセルの正確な数（sleep_followup.py の表は割合を丸めるため）。

    .venv/Scripts/python.exe step3_memory/precheck/periodic_counts.py [--device cuda]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import headroom as H  # noqa: E402
import sleep as S  # noqa: E402
from headroom import W  # noqa: E402
from sleep_followup import absorb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    v3 = [x for x in S.variants() if x["name"] == "③(γ=0.99)"][0]
    v3p = dict(v3, name="③周期(γ=0.99)", periodic=True)
    L = ["", "## (a) の補い：③周期で見逃したセルの正確な数（periodic_counts.py）", "", "| 始め方 | 評価の乱数 | 組 | 見逃したセル | そのうち後で反応したセル |", "|---|---:|---|---:|---:|"]
    for start in S.STARTS:
        for seed in H.EVAL_SEEDS:
            r = S.run(v3p, wcfg, seed, args.device, start)
            for gi, grp in enumerate(H.GROUPS):
                e, b, n = absorb(r, start, gi)
                L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {'/'.join(f'{p:g}' for p in grp)} | {e}/{n} | {b} |")
    path = H.ROOT / "step3_memory" / "runs" / "precheck_sleep" / "followup.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
