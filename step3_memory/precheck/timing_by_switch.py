"""
9.7 の補い：(は)（間隔を2倍ずつ）の見逃しが、どの切り替えで起きているか（読み方は事前に決めていない。判定には使わない）。

    .venv/Scripts/python.exe step3_memory/precheck/timing_by_switch.py [--device cuda]
出力：step3_memory/runs/precheck_sleep/timing_by_switch.md
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
from sleep_timing import variants  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    v = [x for x in variants() if x["name"].startswith("(は)")][0]
    heads = [f"{H.SEQ[sw]}→{H.SEQ[sw + 1]}" for sw in range(len(H.SEQ) - 1)]
    L = ["# (は) の見逃しの数を切り替えごとに（9.7 の補い。読み方は事前に決めていない）", "",
         "| 始め方 | 評価の乱数 | 組 | " + " | ".join(heads) + " | 合計（判定する切り替え） |", "|---|---:|---|" + "---:|" * (len(heads) + 1)]
    for start in S.STARTS:
        for seed in H.EVAL_SEEDS:
            res = S.run(v, wcfg, seed, args.device, start)
            first = 1 if start == "adult" else 0
            for gi, grp in enumerate(H.GROUPS):
                sl = slice(gi * H.N_PER, (gi + 1) * H.N_PER)
                m = (~res["react"][:, sl][:, :, res["wave"]]).sum((1, 2)).tolist()
                cells = [("（" + str(int(x)) + "）" if sw < first else str(int(x))) for sw, x in enumerate(m)]
                L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {'/'.join(f'{p:g}' for p in grp)} | " + " | ".join(cells) + f" | {int(sum(m[first:]))} |")
    L += ["", "（かっこ）は判定に入れない切り替え（大人から始めたときの1回目）"]
    path = H.ROOT / "step3_memory" / "runs" / "precheck_sleep" / "timing_by_switch.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
